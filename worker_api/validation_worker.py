import gc
import time
import os
import multiprocessing
import logging
import traceback
from typing import List, Optional

import uvicorn
from tqdm.auto import tqdm

import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, Header, Request, Response
from huggingface_hub.hf_api import HfApi, RepositoryNotFoundError, GatedRepoError
from dotenv import load_dotenv
import random
from pydantic import BaseModel
from typing import Dict, Any

from worker_api.evaluator import EvaluationScore, Evaluator, InferenceScore, RunError
from worker_api.persistence import SupabaseState
from common.scores import StatusEnum, Scores
from utilities.validation_utils import (
    regenerate_hash,
)
from utilities.repo_details import (
    get_model_size,
    check_model_repo_details,
    ModelRepo,
)
from utilities.event_logger import EventLogger
from scoring.common import EvaluateModelRequest, chat_template_mappings
from dotenv import load_dotenv
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import disable_progress_bars
from maintenance import clean_old_folders
disable_progress_bars()


load_dotenv()

# Constants
MAX_GENERATION_LEEWAY = 0.5  # should be between 0 and 1. This is the percentage of tokens that the model can generate more than the last user message
MAX_GENERATION_LENGTH = 200  # maximum number of tokens that the model can generate
LENGTH_DIFF_PENALTY_STEEPNESS = 2  # the steepness of the exponential decay of the length difference penalty
MAX_AVG_LATENCY = 10000  # in milliseconds

MAX_MODEL_SIZE = 72 * 1024 * 1024 * 1024  # in bytes
MIN_REPO_SIZE = 40 * 1024 * 1024  # in bytes
MAX_REPO_SIZE = 80 * 1024 * 1024 * 1024  #  in bytes
SAMPLE_SIZE = 1024  # number of samples to evaluate the model from the dataset
BATCH_SIZE = 4  # batch size for evaluation
VOCAB_TRUNCATION = 1000  # truncate the vocab to top n tokens
PROB_TOP_K = 10  # the correct token should be in the top n tokens, else a score of 0 is given to that token
# TODO: this will truncate the sequence to MAX_SEQ_LEN tokens. This is a temporary fix to make the evaluation faster.
MAX_SEQ_LEN = (
    4096  # maximum sequence length that should be allowed because eval gets really slow with longer sequences than this
)


SAVE_LEADERBOARD_EVERY = 60  # save the leaderboard every 60 seconds


BLOCK_RATE_LIMIT = 28800  # Every 14400 blocks = 48 hours
app = FastAPI()
supabaser = SupabaseState()

logger = logging.getLogger("uvicorn")

logging.basicConfig(level=logging.ERROR)


app.state.leaderboard_update_time = None
app.state.leaderboard = None

admin_key = os.environ["ADMIN_KEY"]
HF_TOKEN = os.environ.get("HF_ACCESS_TOKEN", "x")
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

hf_api = HfApi()


def model_evaluation_queue(queue_id):
    try:
        while True:
            _model_evaluation_step(queue_id)
            time.sleep(5)
    except Exception as e:
        app.state.event_logger.error("queue_error", queue_id=queue_id, error=e)


def start_staggered_queues(num_queues: int, stagger_seconds: int):
    processes: List[multiprocessing.Process] = []
    for i in range(num_queues):
        p = multiprocessing.Process(target=model_evaluation_queue, args=(i,))
        processes.append(p)
        p.start()
        logger.info(f"Started queue {i}")
        time.sleep(stagger_seconds + i)
    return processes


def _model_evaluation_step(queue_id):
    time.sleep(random.random())

    request = get_next_model_to_eval()
    if request is None:  # Sentinel value to exit the process
        logger.info("No more models to evaluate. Sleep for 15 seconds before checking again.")
        return
    queued_message = f"model_eval_queue_start {request} {queue_id}"
    print(queued_message)
    app.state.event_logger.info(queued_message)
    try:
        result = _evaluate_model(request, queue_id)
        if result is None:
            result = {"note": "incoherent model"}
        app.state.event_logger.info("model_eval_queue_complete", result=result, request=request)
    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
        app.state.event_logger.info("model_eval_queue_error", error=e)
    finally:
        model_dir = f"/tmp/modelcache/{request.hash}"
        shutil.rmtree(model_dir)
        print(f"----successfully deleted {model_dir}")

        gc.collect()


def get_next_model_to_eval():
    response = supabaser.get_next_model_to_eval()

    if response is None:
        return None
    request = EvaluateModelRequest(
        repo_namespace=response["repo_namespace"],
        repo_name=response["repo_name"],
        chat_template_type=response["chat_template_type"],
        hash=response["hash"],
    )
    return request


GPU_ID_MAP = {
    0: "0",
    1: "1",
    2: "2",
    3: "3",
}

import shutil


def _evaluate_model(
    request: EvaluateModelRequest,
    queue_id: int,
):
    """
    Evaluate a model based on the model size and the quality of the model.
    """
    supabaser.update_leaderboard_status(
        request.hash,
        StatusEnum.RUNNING,
        "Model evaluation in progress starting with inference score",
    )

    model_dir = f"/tmp/modelcache/{request.hash}"
    try:
        snapshot_download(repo_id=f"{request.repo_namespace}/{request.repo_name}", local_dir=model_dir, token=HF_TOKEN)
    except Exception as e:
        print(e)
        error_string = f"snapshot_download_error: could not download model from huggingface"
        supabaser.update_leaderboard_status(request.hash, StatusEnum.FAILED, error_string)
        raise RuntimeError(error_string)
    evaluator = Evaluator(gpu_ids=GPU_ID_MAP[queue_id], model_dir=model_dir, trace=False)

    try:
        inference_response = evaluator.inference_score(request)
        if isinstance(inference_response, RunError):
            raise Exception(inference_response.error)
        coherence_score = 1
        judge_score = inference_response.judge_score

        upsert_row_supabase(
            {
                "hash": request.hash,
                "judge_score": judge_score,
                "coherence_score": coherence_score,
                "notes": f"Inference score complete",
            }
        )
    except Exception as e:
        error_string = f"inference_score_error with message: {e}"
        supabaser.update_leaderboard_status(request.hash, StatusEnum.FAILED, error_string)
        raise RuntimeError(error_string)

    eval_score = 0
    latency_score = 0
    model_size_score = 0

    if eval_score is None or latency_score is None or model_size_score is None or judge_score is None:
        raise HTTPException(
            status_code=500,
            detail="Error calculating scores, one or more scores are None",
        )

    full_score_data = Scores()
    full_score_data.judge_score = judge_score
    try:
        upsert_row_supabase(
            {
                "hash": request.hash,
                "total_score": full_score_data.judge_score,
                "status": StatusEnum.COMPLETED,
                "notes": f"scoring_status_complete given worker {queue_id}",
            }
        )
        logger.info(f"update_entry_complete now deleting directory {model_dir}")
    except Exception as e:
        failure_reason = str(e)
        logger.error(f"Updating leaderboard to FAILED: {failure_reason}")
        supabaser.update_leaderboard_status(request.hash, StatusEnum.FAILED, failure_reason)
        raise RuntimeError("Error updating leaderboard: " + failure_reason)
    result = {
        "full_score_data": full_score_data,
    }
    return result


def repository_exists(repo_id):
    for attempt in range(3):
        try:
            hf_api.repo_info(repo_id)  # 'username/reponame'
            return True
        except RepositoryNotFoundError:
            if attempt == 2:  # Last attempt
                return False
        except GatedRepoError:
            # If we get a GatedRepoError, it means the repo exists but is private
            if attempt == 2:  # Last attempt
                return False
        except Exception as e:
            app.state.event_logger.error("hf_repo_error", error=e)
            if attempt == 2:  # Last attempt
                return False


class MinerboardRequest(BaseModel):
    uid: int
    hotkey: str
    hash: str
    block: int
    admin_key: Optional[str] = "admin_key"



def hash_check(request: EvaluateModelRequest) -> bool:
    hotkey_hash_matches = int(request.hash) == regenerate_hash(
        request.repo_namespace,
        request.repo_name,
        request.chat_template_type,
        request.hotkey,
    )
    return hotkey_hash_matches


"""
curl -X GET "http://URL.com/model_submission_details?repo_namespace=my-org&repo_name=my-model&chat_template_type=chatml&hash=12345678&competition_id=comp-1&hotkey=xxxx"
Used by validators to get the model details
"""


def update_failure(new_entry, failure_notes):
    # noop if already marked failed
    if new_entry["status"] == StatusEnum.FAILED:
        return new_entry
    new_entry["status"] = StatusEnum.FAILED
    new_entry["notes"] = failure_notes
    return new_entry


def update_completed(new_entry, failure_notes):
    # noop if already marked failed
    if new_entry["status"] == StatusEnum.FAILED:
        return new_entry
    new_entry["status"] = StatusEnum.COMPLETED
    new_entry["notes"] = failure_notes
    return new_entry


INVALID_BLOCK_START = 3840700
INVALID_BLOCK_END = 5112345

def upsert_row_supabase(row):
    app.state.supabase_client.table("leaderboard").upsert(row).execute()


def start():
    # add command line arguments for the ports of the two apis
    import argparse

    parser = argparse.ArgumentParser(description="Run the server")
    parser.add_argument("--main-api-port", type=int, default=8000, help="Port for the main API")
    parser.add_argument(
        "--queues",
        type=int,
        default=0,
        help="Specify the number of queues to start (default: 1)",
    )
    parser.add_argument(
        "--worker",
        action="store_true",
        help="Run only the worker processes without the API server",
    )
    args = parser.parse_args()
    num_queues = args.queues
    MAIN_API_PORT = args.main_api_port
    app.state.event_logger_enabled = False
    try:
        event_logger = EventLogger()
        app.state.event_logger = event_logger
        app.state.event_logger_enabled = True
    except Exception as e:
        logger.warning(f"Failed to create event logger: {e}")

    try:
        app.state.supabase_client = supabaser.supa_client()
    except Exception as e:
        logger.warning(f"Failed to create Supabase client: {e}")
        supabase_client = None
    import datetime

    processes = []
    stagger_seconds = 2
    try:
        logger.info(f"Starting {num_queues} evaluation threads")
        processes = start_staggered_queues(num_queues, stagger_seconds)
        while True:
            time.sleep(60)
            print(f"Current time: {datetime.datetime.now()}")

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, stopping...")
    except Exception as e:
        logger.error(f"An exception occurred: {e}")
    finally:
        logger.info("Stopping evaluation thread")
        for process in processes:
            process.terminate()
            process.join()


if __name__ == "__main__":
    start()
