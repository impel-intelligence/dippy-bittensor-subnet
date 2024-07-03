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

from model.data import ModelId

import math
import torch
import typing
import constants
import traceback
import bittensor as bt

from model.scores import StatusEnum
from neurons.validator import get_model_score, LocalMetadata
import os

l = LocalMetadata(commit="x",btversion="x")

class ModelQueue:
    @staticmethod
    def config():
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--device",
            type=str,
            default="cuda",
            help="Device name.",
        )
        parser.add_argument(
            "--netuid", type=str, default=constants.SUBNET_UID, help="The subnet UID."
        )
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
        bt.logging.add_args(parser)
        bt.wallet.add_args(parser)
        bt.axon.add_args(parser)
        config = bt.config(parser)
        return config

    def __init__(self):
        self.config = ModelQueue.config()
        bt.logging(config=self.config)
        self.netuid = self.config.netuid or 11

        bt.logging.info(f"Starting model queue with config: {self.config}")

        # === Bittensor objects ====
        self.subtensor = bt.subtensor(config=self.config)
        self.metagraph: bt.metagraph = self.subtensor.metagraph(self.config.netuid)

    # Every hour
    def forever(self):
        while True:
            now = dt.datetime.now()
            # Calculate the next 15-minute mark
            minutes_until_next_15 = 15 - (now.minute % 15)
            next_15_minute_mark = now + dt.timedelta(minutes=minutes_until_next_15)
            next_15_minute_mark = next_15_minute_mark.replace(second=0, microsecond=0)
            sleep_time = (next_15_minute_mark - now).total_seconds()
            time.sleep(sleep_time)
            self.load_latest_metagraph()

    def load_latest_metagraph(self):
        metagraph = self.subtensor.metagraph(self.netuid)
        all_uids = metagraph.uids.tolist()
        queued = 0
        failed = 0
        completed = 0
        for uid in all_uids:
            try:
                hotkey = metagraph.hotkeys[uid]
                metadata = bt.extrinsics.serving.get_metadata(
                    self.subtensor, self.netuid, hotkey
                )
                if metadata is None:
                    continue
                commitment = metadata["info"]["fields"][0]
                hex_data = commitment[list(commitment.keys())[0]][2:]
                chain_str = bytes.fromhex(hex_data).decode()
                model_id = ModelId.from_compressed_str(chain_str)

                result = get_model_score(
                    namespace=model_id.namespace,
                    name=model_id.name,
                    hash=model_id.hash,
                    template=model_id.chat_template,
                    config=self.config,
                    local_metadata=l,
                    retryWithRemote=True,
                )

                if result.status == StatusEnum.QUEUED:
                    stats = f"uid: {uid} hotkey : {hotkey} model_metadata : {model_id} \n result: {result}"
                    bt.logging.info(f"QUEUED : {stats}")
                    queued += 1
                if result.status == StatusEnum.FAILED:
                    stats = f"uid: {uid} hotkey : {hotkey} model_metadata : {model_id} \n result: {result}"
                    bt.logging.info(f"FAILED : {stats}")
                    failed += 1
                if result.status == StatusEnum.COMPLETED:
                    completed += 1

            except Exception as e:
                bt.logging.error(e)
                continue
        bt.logging.info(f"queued {queued} failed {failed} completed {completed}")




if __name__ == "__main__":
    q = ModelQueue()
    q.forever()
