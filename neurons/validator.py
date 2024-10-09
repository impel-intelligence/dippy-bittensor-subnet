# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 const
import copy

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

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Optional
import datetime as dt
import os
import math
import time
import torch
import random
import shutil
import asyncio
import subprocess
import argparse
from typing import Tuple
from threadpoolctl import threadpool_limits
import requests
from importlib.metadata import version
from shlex import split

import constants
from model.data import ModelMetadata, ModelId
from huggingface_hub import get_safetensors_metadata
from model.scores import Scores, StatusEnum
from model import wandb_logger
import traceback
import threading
import multiprocessing
from rich.table import Table
from rich.console import Console

from utilities.compete import iswin
from utilities.event_logger import EventLogger
from utilities import utils
from utilities.miner_registry import MinerEntry

import constants
import traceback
import bittensor as bt

import os
import numpy as np
import torch
from scipy import optimize

from utilities.validation_utils import regenerate_hash

os.environ["TOKENIZERS_PARALLELISM"] = "false"
INVALID_BLOCK_START = 3840700
INVALID_BLOCK_END = 3933300
PRE_HOTKEY_BLOCKS = 4005701

def compute_wins(
    miner_registry: Dict[int, MinerEntry],
) -> Tuple[Dict[int, int], Dict[int, float]]:
    """
    Computes the wins and win rate for each model based on loss comparison.

    This function iterates through each miner in the registry, comparing losses to determine the number of wins
    and then calculates the win rate for each miner.

    Parameters:
        miner_registry (Dict[int, MinerEntry]): A dictionary with miner UIDs as keys and MinerEntry objects as values.

    Returns:
        Tuple[Dict[int, int], Dict[int, float]]: A tuple containing two dictionaries:
            - The first dictionary maps miner IDs to their respective number of wins.
            - The second dictionary maps miner IDs to their win rate, calculated as the number of wins divided by the total comparisons.
    """

    uids = miner_registry.keys()
    wins = {uid: 0 for uid in uids}
    win_rate = {uid: 0 for uid in uids}
    for i, uid_i in enumerate(uids):
        total_matches = 0
        block_i = miner_registry[uid_i].block
        for j, uid_j in enumerate(uids):
            if i == j:
                continue
            block_j = miner_registry[uid_j].block
            score_i = miner_registry[uid_i].total_score
            score_j = miner_registry[uid_j].total_score
            wins[uid_i] += 1 if iswin(score_i, score_j, block_i, block_j) else 0
            total_matches += 1
        # Calculate win rate for uid i
        win_rate[uid_i] = wins[uid_i] / total_matches if total_matches > 0 else 0
        if miner_registry[uid_i].invalid or miner_registry[uid_i].total_score == 0:
            win_rate[uid_i] = float("-inf")

    return wins, win_rate


@dataclass
class LocalMetadata:
    """Metadata associated with the local validator instance"""

    commit: str
    btversion: str
    uid: int = 0
    coldkey: str = ""
    hotkey: str = ""


def local_metadata() -> LocalMetadata:
    """Extract the version as current git commit hash"""
    commit_hash = ""
    try:
        result = subprocess.run(
            split("git rev-parse HEAD"),
            check=True,
            capture_output=True,
            cwd=constants.ROOT_DIR,
        )
        commit = result.stdout.decode().strip()
        assert len(commit) == 40, f"Invalid commit hash: {commit}"
        commit_hash = commit[:8]
    except Exception as e:
        commit_hash = "unknown"

    bittensor_version = version("bittensor")
    return LocalMetadata(
        commit=commit_hash,
        btversion=bittensor_version,
    )


class Validator:
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
            "--blocks_per_epoch",
            type=int,
            default=100,
            help="Number of blocks to wait before setting weights.",
        )
        parser.add_argument(
            "--dont_set_weights",
            action="store_true",
            help="Validator does not set weights on the chain.",
        )
        parser.add_argument(
            "--wait_for_inclusion",
            action="store_true",
            help="Validator does not set weights on the chain.",
        )
        parser.add_argument(
            "--offline",
            action="store_true",
            help="Does not launch a wandb run, does not set weights, does not check that your key is registered.",
        )
        parser.add_argument(
            "--immediate",
            action="store_true",
            help="Triggers run step immediately. NOT RECOMMENDED FOR PRODUCTION",
        )
        parser.add_argument("--netuid", type=str, default=constants.SUBNET_UID, help="The subnet UID.")
        parser.add_argument(
            "--genesis",
            action="store_true",
            help="Don't sync to consensus, rather start evaluation from scratch",
        )
        parser.add_argument(
            "--dtype",
            type=str,
            default="bfloat16",
            help="datatype to load model in, either bfloat16 or float16",
        )
        parser.add_argument(
            "--do_sample",
            action="store_true",
            help="Sample a response from each model (for leaderboard)",
        )
        parser.add_argument(
            "--num_samples_per_eval",
            type=int,
            default=64,
            help="Number of samples to evaluate per UID",
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
        parser.add_argument(
            "--wandb-key",
            type=str,
            default="",
            help="A WandB API key for logging purposes",
        )

        bt.subtensor.add_args(parser)
        bt.logging.add_args(parser)
        bt.wallet.add_args(parser)
        bt.axon.add_args(parser)
        config = bt.config(parser)
        return config

    def state_path(self) -> str:
        """
        Constructs a file path for storing validator state.

        Returns:
        str: A string representing the file path.
        """
        return os.path.expanduser(
            "{}/{}/{}/netuid{}/{}".format(
                bt.logging.config().logging.logging_dir,
                self.wallet.name,
                self.wallet.hotkey_str,
                self.config.netuid,
                "vali-state",
            )
        )

    def __init__(self, local_metadata: LocalMetadata):
        self.config = Validator.config()
        bt.logging(config=self.config)

        bt.logging.info(f"Starting validator with config: {self.config}")

        # === Bittensor objects ====
        self.wallet = bt.wallet(config=self.config)
        self.subtensor = bt.subtensor(config=self.config)
        self.metagraph: bt.metagraph = self.subtensor.metagraph(self.config.netuid)
        torch.backends.cudnn.benchmark = True

        # Dont check registration status if offline.
        if not self.config.offline:
            self.uid = utils.assert_registered(self.wallet, self.metagraph)

        # Track how may run_steps this validator has completed.
        self.run_step_count = 0

        # === Running args ===
        self.weights = torch.zeros_like(self.metagraph.S)
        self.alt_weights = torch.zeros_like(self.metagraph.S)
        self.epoch_step = 0
        self.global_step = 0
        self.last_epoch = self.metagraph.block.item()

        # Sync to consensus
        if not self.config.genesis:
            self.weights.copy_(self.metagraph.C)

        validator_uid = 0
        if not self.config.offline:
            validator_uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)

        # Set up local metadata for stats collection
        self.local_metadata = LocalMetadata(
            commit=local_metadata.commit,
            btversion=local_metadata.btversion,
            hotkey=self.wallet.hotkey.ss58_address,
            coldkey=self.wallet.coldkeypub.ss58_address,
            uid=validator_uid,
        )
        bt.logging.info(f"dumping localmetadata: {self.local_metadata}")

        # Initialize wandb
        if self.config.wandb_key:
            wandb_logger.safe_login(api_key=self.config.wandb_key)
            bt.logging.info(f"wandb locked in")
        wandb_logger.safe_init(
            "Validator",
            self.wallet,
            self.metagraph,
            self.config,
        )
        wandb_logger.safe_log(
            {
                "log_success": 1,
            }
        )
        # eventlog_path = "/tmp/sn11_event_logs/event_{time}.log"
        eventlog_path = "/dev/null"
        self.use_event_logger = False
        if os.getenv("SN11_LOG_PATH") is not None:
            eventlog_path = os.getenv("SN11_LOG_PATH")
        try:
            self.event_logger = EventLogger(filepath=eventlog_path)
            self.use_event_logger = True
        except Exception as e:
            bt.logging.error(
                f"Could not initialize event logger: {e}. Event logging is optional and used for diagnostic purposes. If you do not know what this is for, that's ok."
            )

        # == Initialize the update thread ==
        self.stop_event = threading.Event()

    def __del__(self):
        if hasattr(self, "stop_event"):
            self.stop_event.set()

    def _event_log(self, msg: str, **kwargs):
        if self.use_event_logger:
            self.event_logger.info(msg, **kwargs)
        return

    def _with_decoration(self, metadata: LocalMetadata, keypair, payload):
        signature = sign_request(
                    keypair,
                    payload = metadata.hotkey,
                )
        combined_payload = {
            "signature": signature,
            "payload": payload,
            "commit": str(metadata.commit),
            "btversion": str(metadata.btversion),
            "uid": str(metadata.uid),
            "hotkey": str(metadata.hotkey),
            "coldkey": str(metadata.coldkey),
        }
        return combined_payload
    def _remote_log(self, payload):
        event_report_endpoint = f"{constants.VALIDATION_SERVER}/event_report"
        try:
            response = requests.post(event_report_endpoint, json=payload)
            response.raise_for_status()  # Raise an exception for HTTP errors
        except Exception as e:
            bt.logging.error(f"could not remote log: {e}. This error is ok to ignore if you are a validator")

    @staticmethod
    def adjust_for_vtrust(weights: np.ndarray, consensus: np.ndarray, vtrust_min: float = 0.5):
        """
        Interpolate between the current weight and the normalized consensus weights so that the
        vtrust does not fall below vturst_min, assuming the consensus does not change.
        """
        vtrust_loss_desired = 1 - vtrust_min

        # If the predicted vtrust is already above vtrust_min, then just return the current weights.
        orig_vtrust_loss = np.maximum(0.0, weights - consensus).sum()
        if orig_vtrust_loss <= vtrust_loss_desired:
            bt.logging.info("Weights already satisfy vtrust_min. {} >= {}.".format(1 - orig_vtrust_loss, vtrust_min))
            return weights

        # If maximum vtrust allowable by the current consensus is less that vtrust_min, then choose the smallest lambda
        # that still maximizes the predicted vtrust. Otherwise, find lambda that achieves vtrust_min.
        vtrust_loss_min = 1 - np.sum(consensus)
        if vtrust_loss_min > vtrust_loss_desired:
            bt.logging.info(
                "Maximum possible vtrust with current consensus is less than vtrust_min. {} < {}.".format(
                    1 - vtrust_loss_min, vtrust_min
                )
            )
            vtrust_loss_desired = 1.05 * vtrust_loss_min

        # We could solve this with a LP, but just do rootfinding with scipy.
        consensus_normalized = consensus / np.sum(consensus)

        def fn(lam: float):
            new_weights = (1 - lam) * weights + lam * consensus_normalized
            vtrust_loss = np.maximum(0.0, new_weights - consensus).sum()
            return vtrust_loss - vtrust_loss_desired

        sol = optimize.root_scalar(fn, bracket=[0, 1], method="brentq")
        lam_opt = sol.root

        new_weights = (1 - lam_opt) * weights + lam_opt * consensus_normalized
        vtrust_pred = np.minimum(weights, consensus).sum()
        bt.logging.info(
            "Interpolated weights to satisfy vtrust_min. {} -> {}.".format(1 - orig_vtrust_loss, vtrust_pred)
        )
        return new_weights

    async def try_set_weights(self, ttl: int):
        if self.config.dont_set_weights or self.config.offline:
            return

        wait_for_inclusion = False
        try:
            if self.config.wait_for_inclusion:
                wait_for_inclusion = True
        except Exception as e:
            bt.logging.info(f"wait_for_inclusion not set: {wait_for_inclusion}")

        async def _try_set_weights(wait_for_inclusion=False):
            weights_success = False
            try:
                # Fetch latest metagraph
                metagraph = self.subtensor.metagraph(self.config.netuid)
                consensus = metagraph.C.cpu().numpy()
                cpu_weights = self.weights.cpu().numpy()
                adjusted_weights = self.adjust_for_vtrust(cpu_weights, consensus)
                self.weights = torch.tensor(adjusted_weights, dtype=torch.float32)
                self.weights.nan_to_num(0.0)
                self.subtensor.set_weights(
                    netuid=self.config.netuid,
                    wallet=self.wallet,
                    uids=self.metagraph.uids,
                    weights=adjusted_weights,
                    wait_for_inclusion=wait_for_inclusion,
                    version_key=constants.weights_version_key,
                )
                weights_report = {"weights": {}}
                for uid, score in enumerate(self.weights):
                    weights_report["weights"][uid] = score
                wandb_logger.safe_log(weights_report)
                self._event_log("set_weights_complete", weights=weights_report)
                bt.logging.info(f"successfully_set_weights")
                weights_success = True
            except Exception as e:
                bt.logging.error(f"failed_set_weights {e}")
            # Only dump weight state to console
            ws, ui = self.weights.topk(len(self.weights))
            table = Table(title="All Weights")
            table.add_column("uid", justify="right", style="cyan", no_wrap=True)
            table.add_column("weight", style="magenta")
            for index, weight in list(zip(ui.tolist(), ws.tolist())):
                table.add_row(str(index), str(round(weight, 4)))
            console = Console()
            console.print(table)

            # Weight setting status
            status_table = Table(title="Weight Setting Status")
            status_table.add_column("Status", style="cyan")
            status_table.add_column("Value", style="magenta")
            status_table.add_row("successfully_set_weights", str(weights_success))
            status_table.add_row("failed_set_weights", str(not weights_success))
            status_table.add_row("wait_for_inclusion", str(wait_for_inclusion))
            console.print(status_table)
            return weights_success

        weights_set_success = False
        try:
            bt.logging.debug("Setting weights.")
            weights_set_success = await asyncio.wait_for(_try_set_weights(wait_for_inclusion), ttl)
            payload = {
                "time": dt.datetime.utcnow(),
                "weights_set_success": weights_set_success,
                "wait_for_inclusion": wait_for_inclusion,
            }
            bt.logging.debug("Finished setting weights.")
            logged_payload = self._with_decoration(self.local_metadata, self.wallet.hotkey,payload)
            self._remote_log(logged_payload)
        except asyncio.TimeoutError:
            bt.logging.error(f"Failed to set weights after {ttl} seconds")
        except Exception as e:
            bt.logging.error(f"Error setting weights: {e}")

    async def try_sync_metagraph(self, ttl: int) -> bool:
        def sync_metagraph(endpoint):
            # Update self.metagraph
            self.metagraph = self.subtensor.metagraph(self.config.netuid)
            self.metagraph.save()

        process = multiprocessing.Process(target=sync_metagraph, args=(self.subtensor.chain_endpoint,))
        process.start()
        process.join(timeout=ttl)
        if process.is_alive():
            process.terminate()
            process.join()
            bt.logging.error(f"Failed to sync metagraph after {ttl} seconds")
            return False

        self.metagraph.load()
        bt.logging.info("Synced metagraph")
        self._event_log("metagraph_sync_success")

        return True

    async def try_run_step(self, ttl: int) -> Optional[bool]:
        async def _try_run_step():
            success = await self.run_step()
            logged_payload = self._with_decoration(self.local_metadata, self.wallet.hotkey,{"success": success})
            self._remote_log(logged_payload)
            return success

        try:
            bt.logging.warning("Running step.")
            step_success = await asyncio.wait_for(_try_run_step(), ttl)
            bt.logging.warning("Finished running step.")
            return step_success
        except asyncio.TimeoutError:
            bt.logging.error(f"Failed to run step after {ttl} seconds")
            return False
        except Exception as e:
            bt.logging.error(f"Failed to run step : {e} {traceback.format_exc()}")
            return False

    def fetch_model_data(self, hotkey: str) -> Optional[MinerEntry]:
        try:
            metadata = bt.extrinsics.serving.get_metadata(self.subtensor, self.config.netuid, hotkey)
            if metadata is None:
                return None
            
            commitment = metadata["info"]["fields"][0]
            hex_data = commitment[list(commitment.keys())[0]][2:]
            chain_str = bytes.fromhex(hex_data).decode()
     
            model_id = ModelId.from_compressed_str(chain_str) # --
            submission_hash = regenerate_hash(model_id.namespace, model_id.name, model_id.chat_template, hotkey)
            if int(submission_hash) != int(model_id.hash):
                bt.logging.error(f"Submission Hash {submission_hash} -- Original Hash {model_id.hash}")
               
            block = metadata["block"] # --
            bt.logging.error(f"BLOCK -- {metadata['block']}")
            entry = MinerEntry()
            entry.block = block
            entry.model_id = model_id
            return entry
        except Exception as e:
            bt.logging.error(f"could not fetch data for {hotkey} : {e}")
            return None

    async def run_step(self):
        """
        Executes a step in the evaluation process of models. This function performs several key tasks:
        1. Iterate through blockchain state to find miner entries for models.
        2. Fetches model scoring data from separate evaluation instance.
        3. Applies elimination logic to better calulate model scoring.
        4. Calculates wins and win rates for each model to determine their performance relative to others.
        5. Updates the weights of each model based on their performance and applies a softmax normalization.
        6. Logs all relevant data for the step, including model IDs, scores, and win rates.
        """

        # Update self.metagraph
        synced = await self.try_sync_metagraph(ttl=60)
        if not synced:
            return False
        current_block = self.metagraph.block.item()
        competition_parameters = constants.COMPETITION_SCHEDULE[0]
        telemetry_report(self.local_metadata)
        all_uids = self.metagraph.uids.tolist()
        # Avoid biasing lower value uids when making calls
        random.shuffle(all_uids)
        # Prepare evaluation
        bt.logging.debug(
            f"Computing metrics on {len(all_uids)} for competition {competition_parameters.competition_id}"
        )
        miner_registry: Dict[int, MinerEntry] = {uid: MinerEntry() for uid in all_uids}
        invalid_uids = []
        for uid in miner_registry.keys():
            hotkey = self.metagraph.hotkeys[uid]
            miner_registry[uid].hotkey = hotkey
            try:
                model_data = self.fetch_model_data(hotkey)
                if model_data is None:
                    invalid_uids.append(uid)
                    bt.logging.info(f"skip {uid} no model_data")
                    continue
                # Skip model submitted after run step has begun
                if model_data.block > current_block:
                    invalid_uids.append(uid)
                    bt.logging.info(f"skip {uid} submitted on {model_data.block} after {current_block}")
                    continue

                if model_data.block > INVALID_BLOCK_START and model_data.block < INVALID_BLOCK_END:
                    invalid_uids.append(uid)
                    bt.logging.info(f"skip {uid} submitted on {model_data.block} given range {INVALID_BLOCK_START} - {INVALID_BLOCK_END}")
                    continue

                if model_data.block < PRE_HOTKEY_BLOCKS:
                    invalid_uids.append(uid)
                    bt.logging.info(f"skip {uid} submitted on {model_data.block} before {PRE_HOTKEY_BLOCKS} Pre Hotkey Check Blocks")
                    continue

                
                if model_data.model_id is None:
                    invalid_uids.append(uid)
                    bt.logging.info(f"skip {uid} no model_id available")
                    continue

                miner_registry[uid].block = model_data.block
                miner_registry[uid].model_id = model_data.model_id

                signed_payload = sign_request(
                    self.wallet.hotkey,
                    hotkey,
                )
                _score_data = _get_model_score(
                    miner_registry[uid].model_id,
                    self.config,
                    self.local_metadata,
                    signed_payload,
                )
                if _score_data.status != StatusEnum.COMPLETED:
                    _score_data = _get_model_score(
                        miner_registry[uid].model_id,
                        self.config,
                        self.local_metadata,
                        signed_payload,
                        True,
                    )
                bt.logging.info(
                    f"_score_data for {uid} on block {miner_registry[uid].block} : {miner_registry[uid].model_id} is {_score_data}"
                )
                if _score_data.status == StatusEnum.QUEUED or _score_data.status == StatusEnum.RUNNING:
                    invalid_uids.append(uid)
                    bt.logging.info(f"skip {uid} status is {_score_data.status}")
                    continue
                if _score_data.status == StatusEnum.COMPLETED:
                    miner_registry[uid].total_score = _score_data.calculate_total_score()
                elif _score_data.status == StatusEnum.FAILED:
                    miner_registry[uid].total_score = 0
            except Exception as e:
                bt.logging.error(f"could not update for {uid}:{hotkey} {e}")
                invalid_uids.append(uid)
                continue

        bt.logging.info(
            f"all_uids : {len(miner_registry)} invalid uids: {len(invalid_uids)} cutoff_block : {current_block}"
        )
        # Mark uids that do not have a proper score
        for uid in invalid_uids:
            miner_registry[uid].invalid = True
            miner_registry[uid].total_score = 0

        # Compute wins and win rates per uid.
        wins, win_rate = compute_wins(miner_registry)
        sorted_uids = sorted(miner_registry.keys())

        # Compute softmaxed weights based on win rate.
        model_weights = torch.tensor([win_rate[uid] for uid in sorted_uids], dtype=torch.float32)
        step_weights = torch.softmax(model_weights / constants.temperature, dim=0)

        # Update weights based on moving average.
        self.weights = torch.zeros_like(self.metagraph.S)
        new_weights = torch.zeros_like(self.metagraph.S)
        for i, uid_i in enumerate(sorted_uids):
            new_weights[uid_i] = step_weights[i]
        new_weights *= 1 / new_weights.sum()
        if new_weights.shape[0] < self.weights.shape[0]:
            self.weights = self.weights[: new_weights.shape[0]]
        elif new_weights.shape[0] > self.weights.shape[0]:
            self.weights = torch.cat(
                [
                    self.weights,
                    torch.zeros(new_weights.shape[0] - self.weights.shape[0]),
                ]
            )
        self.weights = constants.alpha * self.weights + (1 - constants.alpha) * new_weights
        self.weights = self.weights.nan_to_num(0.0)

        # Log to screen.
        self.log_step(
            miner_registry,
            wins,
            win_rate,
        )
        return True

    def log_step(
        self,
        miner_registry: Dict[int, MinerEntry],
        wins,
        win_rate,
    ):
        sorted_uids = sorted(miner_registry.keys())
        # Build step log
        step_log = {
            "timestamp": time.time(),
            "uids": sorted_uids,
            "uid_data": {},
            "step": self.epoch_step,
        }
        for i, uid in enumerate(sorted_uids):
            step_log["uid_data"][str(uid)] = {
                "uid": uid,
                "block": miner_registry[uid].block,
                "score": miner_registry[uid].total_score,
                "win_rate": win_rate[uid],
                "win_total": wins[uid],
                "weight": self.weights[uid].item(),
            }
        table = Table(title="Step")
        table.add_column("uid", justify="right", style="cyan", no_wrap=True)
        table.add_column("score", style="magenta")
        table.add_column("win_rate", style="magenta")
        table.add_column("win_total", style="magenta")
        table.add_column("weights", style="magenta")
        table.add_column("block", style="magenta")
        for uid in sorted_uids:
            try:
                table.add_row(
                    str(uid),
                    str(round(step_log["uid_data"][str(uid)]["score"], 4)),
                    str(round(step_log["uid_data"][str(uid)]["win_rate"], 4)),
                    str(step_log["uid_data"][str(uid)]["win_total"]),
                    str(round(self.weights[uid].item(), 4)),
                    str(step_log["uid_data"][str(uid)]["block"]),
                )
            except:
                pass
        console = Console()
        console.print(table)

        ws, ui = self.weights.topk(len(self.weights))
        table = Table(title="Weights > 0.001")
        table.add_column("uid", justify="right", style="cyan", no_wrap=True)
        table.add_column("weight", style="magenta")
        for index, weight in list(zip(ui.tolist(), ws.tolist())):
            if weight > 0.001:
                table.add_row(str(index), str(round(weight, 4)))
        console = Console()
        console.print(table)

        # Sink step log.
        bt.logging.debug(f"Step results: {step_log}")
        scores_per_uid = {}
        for uid in sorted_uids:
            scores_per_uid[uid] = miner_registry[uid].total_score
        wandb_logger.safe_log({"miner_scores/scored_per_uid": scores_per_uid})
        self._event_log("log_scores", scores=scores_per_uid, step=self.epoch_step)

    async def run(self):
        while True:
            try:
                current_time = dt.datetime.utcnow()
                minutes = current_time.minute

                # Check if we're at a 20-minute mark
                if minutes % 20 == 0 or self.config.immediate:
                    bt.logging.info(f"Running step at {current_time.strftime('%H:%M')}")
                    success = await self.try_run_step(ttl=60 * 20)
                    self.global_step += 1
                    if success:
                        await self.try_set_weights(ttl=120)
                    await self.try_sync_metagraph(ttl=120)
                    if self.config.immediate:
                        await asyncio.sleep(3600)
                    # Wait for 1 minute to avoid running multiple times within the same minute
                    await asyncio.sleep(60)
                else:
                    # Calculate minutes until next 20-minute mark
                    minutes_until_next = 20 - (minutes % 20)
                    next_run = current_time + dt.timedelta(minutes=minutes_until_next)
                    bt.logging.info(
                        f"Waiting {minutes_until_next} minutes until next run at {next_run.strftime('%H:%M')}"
                    )

                    # Wait until the next minute before checking again
                    await asyncio.sleep(60)

            except KeyboardInterrupt:
                bt.logging.info("KeyboardInterrupt caught")
                exit()
            except Exception as e:
                bt.logging.error(f"Error in validator loop \n {e} \n {traceback.format_exc()}")
                # Add a small delay before retrying in case of continuous errors
                await asyncio.sleep(5)


def telemetry_report(local_metadata: LocalMetadata, payload=None):
    telemetry_endpoint = f"{constants.VALIDATION_SERVER}/telemetry_report"

    headers = {
        "Git-Commit": str(local_metadata.commit),
        "Bittensor-Version": str(local_metadata.btversion),
        "UID": str(local_metadata.uid),
        "Hotkey": str(local_metadata.hotkey),
        "Coldkey": str(local_metadata.coldkey),
    }

    # Make the POST request to the validation endpoint
    try:
        response = requests.post(telemetry_endpoint, json=payload, headers=headers)
        response.raise_for_status()  # Raise an exception for HTTP errors
    except Exception as e:
        bt.logging.error(e)
    return


import base64
def sign_request(
        keypair,
        payload: str
):

    signed_payload = keypair.sign(data=payload)
    signed_payload_base64 = base64.b64encode(signed_payload).decode('utf-8')

    return {
        "payload_signed": signed_payload_base64,
        "payload": payload,
    }


def _get_model_score(
    model_id: ModelId,
    config,
    local_metadata: LocalMetadata,
    signatures: Dict[str, str],
    retryWithRemote: bool = False,
) -> Scores:

    return get_model_score(
        namespace=model_id.namespace,
        name=model_id.name,
        hash=model_id.hash,
        template=model_id.chat_template,
        config=config,
        local_metadata=local_metadata,
        signatures=signatures,
        retryWithRemote=retryWithRemote,
    )


def get_model_score(
    namespace,
    name,
    hash,
    template,
    config,
    local_metadata: LocalMetadata,
    signatures: Dict[str, str],
    retryWithRemote: bool = False,
    debug: bool = False,
) -> Scores:
    # Status:
    # QUEUED, RUNNING, FAILED, COMPLETED
    # return (score, status)
    if config.use_local_validation_api and not retryWithRemote:
        validation_endpoint = f"http://localhost:{config.local_validation_api_port}/evaluate_model"
    else:
        validation_endpoint = f"{constants.VALIDATION_SERVER}/evaluate_model"

    # Construct the payload with the model name and chat template type
    payload = {
        "repo_namespace": namespace,
        "repo_name": name,
        "hash": hash,
        "chat_template_type": template,
    }
    headers = {
        "Git-Commit": str(local_metadata.commit),
        "Bittensor-Version": str(local_metadata.btversion),
        "UID": str(local_metadata.uid),
        "Hotkey": str(local_metadata.hotkey),
        "Coldkey": str(local_metadata.coldkey),
    }
    headers.update(signatures)
    if os.environ.get("ADMIN_KEY", None) not in [None, ""]:
        payload["admin_key"] = os.environ["ADMIN_KEY"]

    score_data = Scores()
    # Make the POST request to the validation endpoint
    try:
        response = requests.post(validation_endpoint, json=payload, headers=headers)
        response.raise_for_status()  # Raise an exception for HTTP errors
        # Parse the response JSON
        result = response.json()
        if debug:
            console = Console()
            console.print(f"Payload: {payload}")
        if result is None or "status" not in result:
            score_data.status = StatusEnum.FAILED
            return score_data
        status = StatusEnum.from_string(result["status"])
        score_data.status = status
        if "score" in result:
            score_data.from_response(result["score"])
    except Exception as e:
        score_data.status = StatusEnum.FAILED
        bt.logging.error(e)
        bt.logging.error(f"Failed to get score and status for {namespace}/{name}")

    bt.logging.debug(f"Model {namespace}/{name} has score data {score_data}")
    return score_data


if __name__ == "__main__":
    metadata = local_metadata()
    asyncio.run(Validator(metadata).run())
