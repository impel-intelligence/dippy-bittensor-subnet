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

from collections import defaultdict
import datetime as dt
import os
import math
import time
import torch
import random
import shutil
import asyncio
import argparse
import typing
from threadpoolctl import threadpool_limits
import requests


import constants
from model.data import ModelMetadata
from model.model_tracker import ModelTracker
from model.model_updater import ModelUpdater
from model.storage.chain.chain_model_metadata_store import ChainModelMetadataStore
from model.storage.disk.disk_model_store import DiskModelStore
from model.storage.disk.utils import get_hf_download_path
from model.storage.hugging_face.hugging_face_model_store import HuggingFaceModelStore
import traceback
import threading
import multiprocessing
from rich.table import Table
from rich.console import Console

from utilities.miner_iterator import MinerIterator
from utilities import utils
from utilities.perf_monitor import PerfMonitor

import math
import torch
import typing
import constants
import traceback
import bittensor as bt

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def iswin(score_i, score_j, block_i, block_j):
    """
    Determines the winner between two models based on the epsilon adjusted loss.

    Parameters:
        loss_i (float): Loss of uid i on batch
        loss_j (float): Loss of uid j on batch.
        block_i (int): Block of uid i.
        block_j (int): Block of uid j.
    Returns:
        bool: True if loss i is better, False otherwise.
    """
    # Adjust score based on timestamp and pretrain epsilon
    score_i = (1 - constants.timestamp_epsilon) * score_i if block_i > block_j else score_i
    score_j = (1 - constants.timestamp_epsilon) * score_j if block_j > block_i else score_j
    return score_i > score_j

def compute_wins(
    uids: typing.List[int],
    scores_per_uid: typing.Dict[int, float],
    uid_to_block: typing.Dict[int, int],
):
    """
    Computes the wins and win rate for each model based on loss comparison.

    Parameters:
        uids (list): A list of uids to compare.
        scores_per_uid (dict): A dictionary of losses for each uid by batch.
        batches (List): A list of data batches.
        uid_to_block (dict): A dictionary of blocks for each uid.

    Returns:
        tuple: A tuple containing two dictionaries, one for wins and one for win rates.
    """
    wins = {uid: 0 for uid in uids}
    win_rate = {uid: 0 for uid in uids}
    for i, uid_i in enumerate(uids):
        total_matches = 0
        block_i = uid_to_block[uid_i]
        for j, uid_j in enumerate(uids):
            if i == j:
                continue
            block_j = uid_to_block[uid_j]
            score_i = scores_per_uid[uid_i]
            score_j = scores_per_uid[uid_j]
            wins[uid_i] += 1 if iswin(score_i, score_j, block_i, block_j) else 0
            total_matches += 1
        # Calculate win rate for uid i
        win_rate[uid_i] = wins[uid_i] / total_matches if total_matches > 0 else 0

    return wins, win_rate

def best_uid(metagraph: bt.metagraph) -> int:
    """Returns the best performing UID in the metagraph."""
    return max(range(metagraph.n), key=lambda uid: metagraph.I[uid].item())

def nearest_tempo(start_block, tempo, block):
    start_num = start_block + tempo
    intervals = (block - start_num) // tempo
    nearest_num = start_num + intervals * tempo
    if nearest_num >= block:
        nearest_num -= tempo
    return nearest_num

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
            "--blocks_per_epoch",
            type=int,
            default=100,
            help="Number of blocks to wait before setting weights.",
        )
        parser.add_argument(
            "--sample_min",
            type=int,
            default=15,
            help="Number of uids to eval each step.",
        )
        parser.add_argument(
            "--dont_set_weights",
            action="store_true",
            help="Validator does not set weights on the chain.",
        )
        parser.add_argument(
            "--offline",
            action="store_true",
            help="Does not launch a wandb run, does not set weights, does not check that your key is registered.",
        )
        parser.add_argument(
            "--model_dir",
            default=os.path.join(constants.ROOT_DIR, "model-store/"),
            help="Where to store downloaded models",
        )
        parser.add_argument(
            "--netuid", type=str, default=constants.SUBNET_UID, help="The subnet UID."
        )
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
            "--clean_period_minutes",
            type=int,
            default=1,
            help="How often to delete unused models",
        )
        parser.add_argument(
            "--update_delay_minutes",
            type=int,
            default=5,
            help="Period between checking for new models from each UID",
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

    def __init__(self):
        self.config = ModelQueue.config()
        bt.logging(config=self.config)

        bt.logging.info(f"Starting validator with config: {self.config}")

        # === Bittensor objects ====
        self.wallet = bt.wallet(config=self.config)
        self.subtensor = bt.subtensor(config=self.config)
        self.dendrite = bt.dendrite(wallet=self.wallet)
        self.metagraph: bt.metagraph = self.subtensor.metagraph(self.config.netuid)
        torch.backends.cudnn.benchmark = True

        # Dont check registration status if offline.
        if not self.config.offline:
            self.uid = utils.assert_registered(self.wallet, self.metagraph)

        # Track how may run_steps this validator has completed.
        self.run_step_count = 0

        # === Running args ===
        self.weights = torch.zeros_like(self.metagraph.S)
        self.epoch_step = 0
        self.global_step = 0
        self.last_epoch = self.metagraph.block.item()

        self.uids_to_eval: typing.Dict[str, typing.Set] = {}

        # Create a set of newly added uids that should be evaluated on the next loop.
        self.pending_uids_to_eval_lock = threading.RLock()
        self.pending_uids_to_eval: typing.Dict[str, typing.Set] = {}

        # Setup a model tracker to track which miner is using which model id.
        self.model_tracker = ModelTracker()

        # Setup a miner iterator to ensure we update all miners.
        # This subnet does not differentiate between miner and validators so this is passed all uids.
        self.miner_iterator = MinerIterator(self.metagraph.uids.tolist())

        # Setup a ModelMetadataStore
        self.metadata_store = ChainModelMetadataStore(
            self.subtensor, self.config.netuid, self.wallet
        )

        # Setup a RemoteModelStore
        self.remote_store = HuggingFaceModelStore()

        # Setup a LocalModelStore
        self.local_store = DiskModelStore(base_dir=self.config.model_dir)

        # Setup a model updater to download models as needed to match the latest provided miner metadata.
        self.model_updater = ModelUpdater(
            metadata_store=self.metadata_store,
            remote_store=self.remote_store,
            local_store=self.local_store,
            model_tracker=self.model_tracker,
        )

        # Sync to consensus
        if not self.config.genesis:
            competition_ids: typing.Dict[int, typing.Optional[str]] = {}
            for uid, hotkey in enumerate(list(self.metagraph.hotkeys)):
                try:
                    metadata: typing.Optional[ModelMetadata] = asyncio.run(
                        self.metadata_store.retrieve_model_metadata(hotkey)
                    )
                    competition_ids[uid] = (
                        (
                            metadata.id.competition_id
                            if metadata.id.competition_id is not None
                            else constants.ORIGINAL_COMPETITION_ID
                        )
                        if metadata is not None
                        else None
                    )
                except Exception as e:
                    bt.logging.error(
                        f"Unable to get metadata for consensus UID {uid} with hotkey {hotkey}"
                    )
                    bt.logging.error(e)
                    competition_ids[uid] = None

            self.weights.copy_(self.metagraph.C)

            for competition in constants.COMPETITION_SCHEDULE:
                bt.logging.warning(
                    f"Building consensus state for competition {competition.competition_id}"
                )
                consensus = [
                    x[0]
                    for x in sorted(
                        [
                            (i, val.nan_to_num(0).item())
                            for (i, val) in enumerate(list(self.metagraph.consensus))
                            if competition_ids[i] == competition.competition_id
                        ],
                        key=lambda x: x[1],
                        reverse=True,
                    )[: self.config.sample_min]
                ]

                self.uids_to_eval[competition.competition_id] = set(consensus)
                self.pending_uids_to_eval[competition.competition_id] = set()

                consensus_map = {uid: self.weights[uid].item() for uid in consensus}
                bt.logging.info(
                    f"Consensus for competition {competition.competition_id}: {consensus_map}"
                )

                for uid in consensus:
                    hotkey = self.metagraph.hotkeys[uid]
                    try:
                        asyncio.run(self.model_updater.sync_model(hotkey))
                        if (
                            self.model_tracker.get_model_metadata_for_miner_hotkey(
                                hotkey
                            )
                            is None
                        ):
                            bt.logging.warning(
                                f"Unable to get metadata for consensus UID {uid} with hotkey {hotkey}"
                            )
                    except Exception as e:
                        bt.logging.warning(
                            f"Unable to sync model for consensus UID {uid} with hotkey {hotkey}"
                        )

        # Touch all models, starting a timer for them to be deleted if not used
        self.model_tracker.touch_all_miner_models()
        
        # == Initialize the update thread ==
        self.stop_event = threading.Event()
        self.update_thread = threading.Thread(
            target=self.update_models,
            args=(self.config.update_delay_minutes,),
            daemon=True,
        )
        self.update_thread.start()

        # == Initialize the cleaner thread to remove outdated models ==
        self.clean_thread = threading.Thread(
            target=self.clean_models,
            args=(self.config.clean_period_minutes,),
            daemon=True,
        )
        self.clean_thread.start()

    def __del__(self):
        if hasattr(self, "stop_event"):
            self.stop_event.set()
            self.update_thread.join()
            self.clean_thread.join()

    def update_models(self, update_delay_minutes):
        # Track how recently we updated each uid
        uid_last_checked = dict()

        # The below loop iterates across all miner uids and checks to see
        # if they should be updated.
        while not self.stop_event.is_set():
            try:
                # Get the next uid to check
                next_uid = next(self.miner_iterator)

                # Confirm that we haven't checked it in the last `update_delay_minutes` minutes.
                time_diff = (
                    dt.datetime.now() - uid_last_checked[next_uid]
                    if next_uid in uid_last_checked
                    else None
                )

                if time_diff and time_diff < dt.timedelta(minutes=update_delay_minutes):
                    # If we have seen it within `update_delay_minutes` minutes then sleep until it has been at least `update_delay_minutes` minutes.
                    time_to_sleep = (
                        dt.timedelta(minutes=update_delay_minutes) - time_diff
                    ).total_seconds()
                    bt.logging.info(
                        f"Update loop has already processed all UIDs in the last {update_delay_minutes} minutes. Sleeping {time_to_sleep} seconds."
                    )
                    time.sleep(time_to_sleep)

                uid_last_checked[next_uid] = dt.datetime.now()
                bt.logging.info(f"Updating model for UID={next_uid}")

                # Get their hotkey from the metagraph.
                hotkey = self.metagraph.hotkeys[next_uid]

                # Compare metadata and tracker, syncing new model from remote store to local if necessary.
                updated = asyncio.run(self.model_updater.sync_model(hotkey))
                
                # Ensure we eval the new model on the next loop.
                if updated:
                    metadata = self.model_tracker.get_model_metadata_for_miner_hotkey(
                        hotkey
                    )
                    if metadata is not None:
                        bt.logging.warning(
                            f"Updated model for UID={next_uid}. Was new = {updated}"
                        )
                        with self.pending_uids_to_eval_lock:
                            self.pending_uids_to_eval[metadata.id.competition_id].add(
                                next_uid
                            )
                            bt.logging.debug(
                                f"Found a new model for UID={next_uid} for competition {metadata.id.competition_id}. It will be evaluated on the next loop."
                            )
                    else:
                        bt.logging.warning(
                            f"Unable to sync model for consensus UID {next_uid} with hotkey {hotkey}"
                        )

            except Exception as e:
                bt.logging.error(f"Error in update loop: {e}")

        bt.logging.info("Exiting update models loop.")

    def clean_models(self, clean_period_minutes: int):
        # The below loop checks to clear out all models in local storage that are no longer referenced.
        while not self.stop_event.is_set():
            try:
                old_models = self.model_tracker.get_and_clear_old_models()

                if len(old_models) > 0:
                    bt.logging.info("Starting cleanup of stale models. Removing {}...".format(len(old_models)))

                for hotkey, model_metadata in old_models:
                    local_path = self.local_store.get_path(hotkey)
                    model_dir = get_hf_download_path(local_path, model_metadata.id)
                    shutil.rmtree(model_dir, ignore_errors=True)

                if len(old_models) > 0:
                    bt.logging.info("Starting cleanup of stale models. Removing {}... Done!".format(len(old_models)))

            except Exception as e:
                bt.logging.error(f"Error in clean loop: {e}")
                print(traceback.format_exc())

            time.sleep(dt.timedelta(minutes=clean_period_minutes).total_seconds())

        bt.logging.info("Exiting clean models loop.")

    async def try_set_weights(self, ttl: int):
        async def _try_set_weights():
            try:
                self.weights.nan_to_num(0.0)
                self.subtensor.set_weights(
                    netuid=self.config.netuid,
                    wallet=self.wallet,
                    uids=self.metagraph.uids,
                    weights=self.weights,
                    wait_for_inclusion=False,
                    version_key=constants.weights_version_key,
                )
            except:
                pass
            ws, ui = self.weights.topk(len(self.weights))
            table = Table(title="All Weights")
            table.add_column("uid", justify="right", style="cyan", no_wrap=True)
            table.add_column("weight", style="magenta")
            for index, weight in list(zip(ui.tolist(), ws.tolist())):
                table.add_row(str(index), str(round(weight, 4)))
            console = Console()
            console.print(table)

        try:
            bt.logging.debug("Setting weights.")
            await asyncio.wait_for(_try_set_weights(), ttl)
            bt.logging.debug("Finished setting weights.")
        except asyncio.TimeoutError:
            bt.logging.error(f"Failed to set weights after {ttl} seconds")

    async def try_sync_metagraph(self, ttl: int):
        def sync_metagraph(endpoint):
            # Update self.metagraph
            self.metagraph = bt.subtensor(endpoint).metagraph(self.config.netuid)
            self.metagraph.save()

        process = multiprocessing.Process(
            target=sync_metagraph, args=(self.subtensor.chain_endpoint,)
        )
        process.start()
        process.join(timeout=ttl)
        if process.is_alive():
            process.terminate()
            process.join()
            bt.logging.error(f"Failed to sync metagraph after {ttl} seconds")
            return

        bt.logging.info("Synced metagraph")
        self.metagraph.load()
        self.miner_iterator.set_miner_uids(self.metagraph.uids.tolist())

    async def try_run_step(self, ttl: int):
        async def _try_run_step():
            await self.run_step()

        try:
            bt.logging.warning("Running step.")
            await asyncio.wait_for(_try_run_step(), ttl)
            bt.logging.warning("Finished running step.")
        except asyncio.TimeoutError:
            bt.logging.error(f"Failed to run step after {ttl} seconds")

    async def run_step(self):
        """
        Executes a step in the evaluation process of models. This function performs several key tasks:
        1. Identifies valid models for evaluation (top sample_min from last run + newly updated models).
        2. Generates random pages for evaluation and prepares batches for each page from the dataset.
        3. Computes the scoring for each model based on the losses incurred on the evaluation batches.
        4. Calculates wins and win rates for each model to determine their performance relative to others.
        5. Updates the weights of each model based on their performance and applies a softmax normalization.
        6. Implements a blacklist mechanism to remove underperforming models from the evaluation set.
        7. Logs all relevant data for the step, including model IDs, pages, batches, wins, win rates, and losses.
        """

        # Update self.metagraph
        await self.try_sync_metagraph(ttl=60)
        competition_parameters = constants.COMPETITION_SCHEDULE[
            self.global_step % len(constants.COMPETITION_SCHEDULE)
        ]

        # Add uids with newly updated models to the upcoming batch of evaluations.
        with self.pending_uids_to_eval_lock:
            self.uids_to_eval[competition_parameters.competition_id].update(
                self.pending_uids_to_eval[competition_parameters.competition_id]
            )
            self.pending_uids_to_eval[competition_parameters.competition_id].clear()

        # Pull relevant uids for step. If they aren't found in the model tracker on eval they will be skipped.
        uids = list(self.uids_to_eval[competition_parameters.competition_id])

        if not uids:
            if self.config.genesis:
                bt.logging.debug(
                    f"No uids to eval for competition {competition_parameters.competition_id}. Waiting 5 minutes to download some models."
                )
                time.sleep(300)
            else:
                bt.logging.debug(
                    f"No uids to eval for competition {competition_parameters.competition_id}."
                )
            return

        # Keep track of which block this uid last updated their model.
        # Default to an infinite block if we can't retrieve the metadata for the miner.
        uid_to_block = defaultdict(lambda: math.inf)

        # Prepare evaluation
        bt.logging.debug(
            f"Computing metrics on {uids} for competition {competition_parameters.competition_id}"
        )
        scores_per_uid = {muid: None for muid in uids}
        sample_per_uid = {muid: None for muid in uids}

        load_model_perf = PerfMonitor("Eval: Load model")
        compute_loss_perf = PerfMonitor("Eval: Compute loss")

        # Queue up model metadata
        self.model_tracker.release_all()
        uid_to_hotkey_and_model_metadata: typing.Dict[
            int, typing.Tuple[str, typing.Optional[ModelMetadata]]
        ] = {}
        for uid_i in uids:
            # Check that the model is in the tracker.
            hotkey = self.metagraph.hotkeys[uid_i]
            model_i_metadata = self.model_tracker.take_model_metadata_for_miner_hotkey(
                hotkey
            )
            bt.logging.info(f"Model metadata for {uid_i} is {model_i_metadata}")
            if model_i_metadata is not None:
                for other_uid, (
                    other_hotkey,
                    other_metadata,
                ) in uid_to_hotkey_and_model_metadata.items():
                    if (
                        other_metadata
                        and model_i_metadata.id.hash == other_metadata.id.hash
                    ):
                        if model_i_metadata.block < other_metadata.block:
                            bt.logging.error(
                                f"Perferring duplicate of {other_uid} with {uid_i} since it is older"
                            )
                            # Release the other model since it is not in use.
                            self.model_tracker.release_model_metadata_for_miner_hotkey(other_hotkey, other_metadata)
                            uid_to_hotkey_and_model_metadata[other_uid] = (
                                other_hotkey,
                                None,
                            )
                        else:
                            bt.logging.error(
                                f"Perferring duplicate of {uid_i} with {other_uid} since it is newer"
                            )
                            # Release own model since it is not in use.
                            self.model_tracker.release_model_metadata_for_miner_hotkey(hotkey, model_i_metadata)
                            model_i_metadata = None
                        break

            uid_to_hotkey_and_model_metadata[uid_i] = (hotkey, model_i_metadata)
            
        bt.logging.info("Looking at model metadata", uid_to_hotkey_and_model_metadata)

        # Compute the score for each model.
        # if status is QUEUED or RUNNING, set score to None
        for uid_i, (
            hotkey,
            model_i_metadata,
        ) in uid_to_hotkey_and_model_metadata.items():
            score = None
            if model_i_metadata is not None:
                if (
                    model_i_metadata.id.competition_id
                    == competition_parameters.competition_id
                ):
                    # self.model_tracker.touch_miner_model(hotkey)
                    # Update the block this uid last updated their model.
                    uid_to_block[uid_i] = model_i_metadata.block
                    while True:
                        try:
                            _score, status = get_model_score(
                                model_i_metadata.id.namespace,
                                model_i_metadata.id.name,
                                model_i_metadata.id.hash,
                                model_i_metadata.id.chat_template,
                                self.config
                            )
                            bt.logging.info(f"Score for {model_i_metadata} is {_score}")
                            bt.logging.info(f"Status for {model_i_metadata} is {status}")
                            bt.logging.debug(f"Queried for score for {model_i_metadata.id} Current status: {status}")
                            if status != 'QUEUED':
                                break
                        except:
                            bt.logging.error(f"Failed to get score for {model_i_metadata.id}")
                            break


        for uid_i, (
            hotkey,
            model_i_metadata,
        ) in uid_to_hotkey_and_model_metadata.items():
            score = None
            if model_i_metadata is not None:
                if (
                    model_i_metadata.id.competition_id
                    == competition_parameters.competition_id
                ):
                    try:
                        self.model_tracker.touch_miner_model(hotkey)
                        # Update the block this uid last updated their model.
                        uid_to_block[uid_i] = model_i_metadata.block

                        while True:
                            try:
                                _score, status = get_model_score(
                                    model_i_metadata.id.namespace,
                                    model_i_metadata.id.name,
                                    model_i_metadata.id.hash,
                                    model_i_metadata.id.chat_template,
                                    self.config
                                )
                                bt.logging.info(f"Score for {model_i_metadata} is {_score}")
                                bt.logging.info(f"Status for {model_i_metadata} is {status}")
                                if status == 'COMPLETED':
                                    score = _score
                                    break
                                elif status == 'FAILED':
                                    score = 0
                                    break
                                else:
                                    bt.logging.debug(f"Waiting for score for {model_i_metadata.id} Current status: {status}")
                                    time.sleep(10)
                            except:
                                bt.logging.error(f"Failed to get score for {model_i_metadata.id}")
                                break
                    except Exception as e:
                        bt.logging.error(
                            f"Error in eval loop: {e}. Setting score for uid: {uid_i} to 0."
                        )
                    finally:
                        # After we are done with the model, release it.
                        self.model_tracker.release_model_metadata_for_miner_hotkey(hotkey, model_i_metadata)
                else:
                    bt.logging.debug(
                        f"Skipping {uid_i}, submission is for a different competition ({model_i_metadata.id.competition_id}). Setting loss to 0."
                    )
            else:
                bt.logging.debug(
                    f"Unable to load the model for {uid_i} (perhaps a duplicate?). Setting loss to 0."
                )
            if not score:
                bt.logging.error(f"Failed to get score for {model_i_metadata}")
                score = 0

            scores_per_uid[uid_i] = score

            bt.logging.warning(
                f"Computed model score for uid: {uid_i}: {score}"
            )
            bt.logging.debug(f"Computed model losses for uid: {uid_i}: {score}")

        # Compute wins and win rates per uid.
        wins, win_rate = compute_wins(uids, scores_per_uid, uid_to_block)
        # Compute softmaxed weights based on win rate.
        model_weights = torch.tensor(
            [win_rate[uid] for uid in uids], dtype=torch.float32
        )
        step_weights = torch.softmax(model_weights / constants.temperature, dim=0)
        # Update weights based on moving average.
        new_weights = torch.zeros_like(self.metagraph.S)
        for i, uid_i in enumerate(uids):
            new_weights[uid_i] = step_weights[i]
        scale = (
            len(constants.COMPETITION_SCHEDULE)
            * competition_parameters.reward_percentage
        )
        new_weights *= scale / new_weights.sum()
        if new_weights.shape[0] < self.weights.shape[0]:
            self.weights = self.weights[: new_weights.shape[0]]
        elif new_weights.shape[0] > self.weights.shape[0]:
            self.weights = torch.cat(
                [
                    self.weights,
                    torch.zeros(new_weights.shape[0] - self.weights.shape[0]),
                ]
            )
        self.weights = (
            constants.alpha * self.weights + (1 - constants.alpha) * new_weights
        )
        self.weights = self.weights.nan_to_num(0.0)

        # Filter based on win rate removing all by the sample_min best models for evaluation.
        self.uids_to_eval[competition_parameters.competition_id] = set(
            sorted(win_rate, key=win_rate.get, reverse=True)[: self.config.sample_min]
        )

        # Log the performance of the eval loop.
        bt.logging.debug(load_model_perf.summary_str())
        bt.logging.debug(compute_loss_perf.summary_str())

        # Log to screen.
        self.log_step(
            competition_parameters.competition_id,
            uids,
            uid_to_block,
            wins,
            win_rate,
            scores_per_uid,
            sample_per_uid,
        )

        # Increment the number of completed run steps by 1
        self.run_step_count += 1

    def log_step(
        self,
        competition_id,
        uids,
        uid_to_block,
        wins,
        win_rate,
        scores_per_uid,
        sample_per_uid,
    ):
        # Build step log
        step_log = {
            "timestamp": time.time(),
            "competition_id": competition_id,
            "uids": uids,
            "uid_data": {},
        }
        for i, uid in enumerate(uids):
            step_log["uid_data"][str(uid)] = {
                "uid": uid,
                "block": uid_to_block[uid],
                "score": scores_per_uid[uid],
                "win_rate": win_rate[uid],
                "win_total": wins[uid],
                "weight": self.weights[uid].item(),
                "sample_prompt": (
                    sample_per_uid[uid][0] if sample_per_uid[uid] is not None else None
                ),
                "sample_response": (
                    sample_per_uid[uid][1] if sample_per_uid[uid] is not None else None
                ),
                "sample_truth": (
                    sample_per_uid[uid][2] if sample_per_uid[uid] is not None else None
                ),
            }
        table = Table(title="Step")
        table.add_column("uid", justify="right", style="cyan", no_wrap=True)
        table.add_column("score", style="magenta")
        table.add_column("win_rate", style="magenta")
        table.add_column("win_total", style="magenta")
        table.add_column("weights", style="magenta")
        table.add_column("block", style="magenta")
        for uid in uids:
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
        bt.logging.warning(f"Step results: {step_log}")

    async def run(self):
        while True:
            try:
                while (
                    self.metagraph.block.item() - self.last_epoch
                    < self.config.blocks_per_epoch
                ):
                    await self.try_run_step(ttl=60 * 20)
                    bt.logging.debug(
                        f"{self.metagraph.block.item() - self.last_epoch } / {self.config.blocks_per_epoch} blocks until next epoch."
                    )
                    self.global_step += 1
                self.last_epoch = self.metagraph.block.item()
                self.epoch_step += 1

            except KeyboardInterrupt:
                bt.logging.info(
                    "KeyboardInterrupt caught"
                )
                exit()

            except Exception as e:
                bt.logging.error(
                    f"Error in validator loop \n {e} \n {traceback.format_exc()}"
                )

def get_model_score(namespace, name, hash, template, config):
    # Status:
    # QUEUED, RUNNING, FAILED, COMPLETED
    # return (score, status)
    if config.use_local_validation_api:
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

    if os.environ.get('ADMIN_KEY', None) not in [None, '']:
        payload['admin_key'] = os.environ['ADMIN_KEY']

    console = Console()
    console.print(f"Payload: {payload}")

    # Make the POST request to the validation endpoint
    try:
        response = requests.post(validation_endpoint, json=payload)
        response.raise_for_status()  # Raise an exception for HTTP errors
        # Parse the response JSON
        result = response.json()
        console = Console()
        console.print(f"Payload: {payload}")
        status = result['status']
        if status == 'COMPLETED':
            score = result['score']['total_score']
        elif status == "FAILED":
            bt.logging.warning(f"Model {namespace}/{name} is in status {status}")
            score = 0
        else:
            return None, status
    except Exception as e:
        score = 0
        status = 'FAILED'
        bt.logging.error(e)
        bt.logging.error(f"Failed to get score and status for {namespace}/{name}")

    bt.logging.info(f"Model {namespace}/{name} has score {score} and status {status}")
    return score, status

if __name__ == "__main__":
    asyncio.run(ModelQueue().run())
