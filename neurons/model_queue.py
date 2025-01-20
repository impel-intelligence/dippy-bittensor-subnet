# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 const

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import datetime as dt
import time
import argparse
import requests

from common.data import ModelId

import math
import torch
import typing
import constants
import traceback
import bittensor as bt

from common.scores import StatusEnum, Scores
from utilities.local_metadata import LocalMetadata
import os

from utilities.event_logger import EventLogger
from utilities.validation_utils import regenerate_hash

l = LocalMetadata(commit="x", btversion="x")
SKIP_BLOCK = 4200000

import requests
from huggingface_hub.utils import build_hf_headers, hf_raise_for_status
import os

ENDPOINT = "https://huggingface.co"

REPO_TYPES = ["model", "dataset", "space"]

hf_token = os.environ["HF_ACCESS_TOKEN"]


def duplicate(repo_namespace: str, repo_name: str):
    destination = f"DippyAI/{repo_namespace}-{repo_name}"
    r = requests.post(
        f"https://huggingface.co/api/models/{repo_namespace}/{repo_name}/duplicate",
        headers=build_hf_headers(token=hf_token),
        json={"repository": destination, "private": True},
    )
    hf_raise_for_status(r)

    repo_url = r.json().get("url")

    return (f"{repo_url}",)


def push_minerboard(
    hash: str,
    uid: int,
    hotkey: str,
    block: int,
    config,
    local_metadata: LocalMetadata,
    retryWithRemote: bool = False,
) -> None:
    if config.use_local_validation_api and not retryWithRemote:
        validation_endpoint = f"http://localhost:{config.local_validation_api_port}/minerboard_update"
    else:
        validation_endpoint = f"{constants.VALIDATION_SERVER}/minerboard_update"

    # Construct the payload with the model name and chat template type
    payload = {
        "hash": hash,
        "uid": uid,
        "hotkey": hotkey,
        "block": block,
    }

    headers = {
        "Git-Commit": str(local_metadata.commit),
        "Bittensor-Version": str(local_metadata.btversion),
        "UID": str(local_metadata.uid),
        "Hotkey": str(local_metadata.hotkey),
        "Coldkey": str(local_metadata.coldkey),
    }
    if os.environ.get("ADMIN_KEY", None) not in [None, ""]:
        payload["admin_key"] = os.environ["ADMIN_KEY"]

    # Make the POST request to the validation endpoint
    try:
        response = requests.post(validation_endpoint, json=payload, headers=headers)
        response.raise_for_status()  # Raise an exception for HTTP errors
    except Exception as e:
        print(e)


class ModelQueue:
    @staticmethod
    def config():
        parser = argparse.ArgumentParser()
        parser.add_argument("--netuid", type=str, default=constants.SUBNET_UID, help="The subnet UID.")
        parser.add_argument(
            "--use-local-validation-api",
            action="store_true",
            help="Use a local validation api",
        )
        parser.add_argument(
            "--local-validation-api-port",
            type=int,
            default=8000,
            help="Port for local validation api",
        )

        bt.subtensor.add_args(parser)
        bt.wallet.add_args(parser)
        bt.axon.add_args(parser)
        config = bt.config(parser)
        return config

    def __init__(self):
        self.config = ModelQueue.config()
        self.netuid = self.config.netuid or 11

        # === Bittensor objects ====
        self.subtensor = bt.subtensor(config=self.config)
        self.metagraph = self.subtensor.metagraph(self.config.netuid)
        logfilepath = "/tmp/modelq/{time:UNIX}.log"
        self.logger = EventLogger(
            filepath=logfilepath,
            level="INFO",
            stderr=True,
        )
        self.logger.info(f"Starting model queue with config: {self.config}")

    # Every x minutes
    def forever(self):
        while True:
            now = dt.datetime.now()
            # Calculate the next 5 minute mark
            minutes_until_next_epoch = 5 - (now.minute % 5)
            next_epoch_minute_mark = now + dt.timedelta(minutes=minutes_until_next_epoch)
            next_epoch_minute_mark = next_epoch_minute_mark.replace(second=0, microsecond=0)
            sleep_time = (next_epoch_minute_mark - now).total_seconds()
            self.logger.info(f"sleeping for {sleep_time}")
            time.sleep(sleep_time)

            try:
                self.load_latest_metagraph()
            except Exception as e:
                self.logger.error(f"failed to queue {e}")

    def load_latest_metagraph(self):
        metagraph = self.subtensor.metagraph(self.netuid)
        all_uids = metagraph.uids.tolist()

        substrate_client = self.subtensor.substrate
        all_commitments = substrate_client.query_map(
            module="Commitments",
            storage_function="CommitmentOf",
            params=[self.config.netuid],
            block_hash=None,
        )
        commitments = {}
        for key, value in all_commitments:
            hotkey = key.value
            commitment_info = value.value.get("info", {})
            fields = commitment_info.get("fields", [])
            if not fields or not isinstance(fields[0], dict):
                continue
            field_value = next(iter(fields[0].values()))
            if field_value.startswith("0x"):
                field_value = field_value[2:]
            try:
                chain_str = bytes.fromhex(field_value).decode("utf-8").strip()
                commitments[str(hotkey)] = {"block": value["block"].value, "chain_str": chain_str}
            except Exception as e:
                self.logger.error(f"Failed to decode commitment for hotkey {hotkey}: {e}")
                continue

        queued = 0
        failed = 0
        no_metadata = 0
        completed = 0
        for uid in all_uids:
            try:
                hotkey = metagraph.hotkeys[uid]
                commit_data = None
                commit_data = commitments[hotkey] if hotkey in commitments else None
                if commit_data is None:
                    no_metadata += 1
                    self.logger.info(f"NO_METADATA : uid: {uid} hotkey : {hotkey}")
                    continue

                model_id = ModelId.from_compressed_str(commit_data["chain_str"])
                block = commit_data["block"]
                if block < SKIP_BLOCK:
                    continue

                result = self.check_model_score(
                    namespace=model_id.namespace,
                    name=model_id.name,
                    hash=model_id.hash,
                    template=model_id.chat_template,
                    block=block,
                    hotkey=hotkey,
                    config=self.config,
                    retryWithRemote=True,
                )
                stats = f"{result.status} : uid: {uid} hotkey : {hotkey} block: {block} model_metadata : {model_id}"
                self.logger.info(stats)
                if result.status == StatusEnum.FAILED:
                    failed += 1

                if result.status == StatusEnum.QUEUED:
                    try:
                        self.logger.info(f"QUEUED: {hotkey}")
                    except Exception as e:
                        self.logger.error(f"could not duplicate repo : {e}")
                    queued += 1

                if result.status == StatusEnum.COMPLETED:
                    completed += 1

                push_minerboard(
                    hash=model_id.hash,
                    uid=uid,
                    hotkey=hotkey,
                    block=block,
                    local_metadata=l,
                    config=self.config,
                    retryWithRemote=True,
                )

            except Exception as e:
                self.logger.error(f"exception for uid {uid} : {e}")
                continue
        self.logger.info(f"no_metadata {no_metadata} queued {queued} failed {failed} completed {completed}")

    def check_model_score(
        self,
        namespace,
        name,
        hash,
        template,
        block,
        hotkey,
        config,
        retryWithRemote: bool = False,
    ) -> Scores:
        # Status:
        # QUEUED, RUNNING, FAILED, COMPLETED
        if config.use_local_validation_api and not retryWithRemote:
            validation_endpoint = f"http://localhost:{config.local_validation_api_port}/check_model"
        else:
            validation_endpoint = f"{constants.VALIDATION_SERVER}/check_model"

        # Construct the payload with the model name and chat template type
        payload = {
            "repo_namespace": namespace,
            "repo_name": name,
            "hash": hash,
            "chat_template_type": template,
            "block": block,
            "hotkey": hotkey,
        }
        score_data = Scores()

        if os.environ.get("ADMIN_KEY", None) not in [None, ""]:
            payload["admin_key"] = os.environ["ADMIN_KEY"]
        result = None
        # Make the POST request to the validation endpoint
        try:
            response = requests.post(validation_endpoint, json=payload)
            response.raise_for_status()  # Raise an exception for HTTP errors
            # Parse the response JSON
            result = response.json()
            if result is None:
                raise RuntimeError(f"no leaderboard entry exists at this time for {payload}")

            status = StatusEnum.from_string(result["status"])
            score_data.status = status
            print(result)
        except Exception as e:
            self.logger.error(f"Failed to get score and status from API for {namespace}/{name} {result} {e}")
            score_data.status = StatusEnum.FAILED
        return score_data


if __name__ == "__main__":
    q = ModelQueue()
    q.forever()
