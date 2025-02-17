# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 const

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import datetime as dt
import time
import argparse
import requests

from common.data import ModelId

import random
import torch
from typing import cast, Any, Dict
import constants
import traceback
import bittensor as bt
from bittensor import Subtensor
from bittensor.core.chain_data import (
    decode_account_id,
)

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
def extract_raw_data(data):
        try:
            # Navigate to the fields tuple
            fields = data.get('info', {}).get('fields', ())
            
            # The first element should be a tuple containing a dictionary
            if fields and isinstance(fields[0], tuple) and isinstance(fields[0][0], dict):
                # Find the 'Raw' key in the dictionary
                raw_dict = fields[0][0]
                raw_key = next((k for k in raw_dict.keys() if k.startswith('Raw')), None)
                
                if raw_key and raw_dict[raw_key]:
                    # Extract the inner tuple of integers
                    numbers = raw_dict[raw_key][0]
                    # Convert to string
                    result = ''.join(chr(x) for x in numbers)
                    return result
                
        except (IndexError, AttributeError):
            pass
        
        return None

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
            "--immediate",
            action="store_true",
            help="Trigger queue immediately",
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
            if not self.config.immediate:
                time.sleep(sleep_time)

            try:
                self.load_latest_metagraph()
            except Exception as e:
                self.logger.error(f"failed to queue {e}")

    def build_commit_data(self) -> Dict[str, Any]:
        max_retries = 10
        base_delay = 1.5  # seconds
        commitments = {}
        raw_commmitments = None
        for attempt in range(max_retries):
            try:
                # First try using self.subtensor
                try:
                    raw_commmitments = self.subtensor.query_map(
                        module="Commitments",
                        name="CommitmentOf",
                        params=[self.config.netuid])
                except Exception as e:
                    bt.logging.warning(f"Failed to fetch metadata with self.subtensor: {e}, trying dedicated subtensor")
                    # Fall back to dedicated subtensor
                    dedicated_subtensor = None
                    try:
                        network = random.choice(["finney", "subvortex", "latent-lite"])
                        dedicated_subtensor = Subtensor(network=network)
                        bt.logging.warning(f"Created dedicated subtensor for metadata fetch: {dedicated_subtensor} ")
                        raw_commmitments = dedicated_subtensor.query_map(
                        module="Commitments",
                        name="CommitmentOf",
                        params=[self.config.netuid])
                    finally:
                        # Ensure we close the dedicated subtensor
                        if dedicated_subtensor is not None:
                            try:
                                dedicated_subtensor.close()
                            except Exception as close_error:
                                bt.logging.error(f"Error closing dedicated subtensor: {close_error}")
            except Exception as e:
                delay = base_delay**attempt
                if attempt < max_retries - 1:  # Don't log "retrying" on the last attempt
                    bt.logging.error(f"Attempt {attempt + 1}/{max_retries} failed to fetch data : {e}")
                    bt.logging.info(f"Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)
                else:
                    bt.logging.error(f"All attempts failed to fetch data : {e}")
                    raise e

        if raw_commmitments is None:
            raise Exception("Failed to fetch raw commitments from chain")
        commitments = {}
        for key, value in raw_commmitments:
            try:
                hotkey = decode_account_id(key[0])
                body = cast(dict, value.value)
                chain_str = extract_raw_data(body)
                commitments[str(hotkey)] = {"block": body["block"], "chain_str": chain_str}
            except Exception as e:
                bt.logging.error(f"Failed to decode commitment for hotkey {hotkey}: {e}")
                continue

        return commitments

    

    def load_latest_metagraph(self):
        metagraph = self.subtensor.metagraph(self.netuid)
        all_uids = metagraph.uids.tolist()

        commitments = self.build_commit_data()

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
                    self.logger.info(f"QUEUED: {hotkey}")

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
