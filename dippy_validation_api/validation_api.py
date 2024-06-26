import gc
import time
import os
import multiprocessing
import logging
from typing import List

import uvicorn

import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, Header, Response
from huggingface_hub import HfApi, HfFolder
from huggingface_hub.hf_api import HfApi, RepositoryNotFoundError
from dotenv import load_dotenv

from dippy_validation_api.evaluator import Evaluator, RunError
from dippy_validation_api.persistence import SupabaseState
from utilities.validation_utils import (
    regenerate_hash,
    check_model_repo_size,
    get_model_size,
)
from utilities.event_logger import EventLogger
from model_evaluation.common import EvaluateModelRequest
from dotenv import load_dotenv

load_dotenv()

# Constants
MAX_GENERATION_LEEWAY = 0.5  # should be between 0 and 1. This is the percentage of tokens that the model can generate more than the last user message
MAX_GENERATION_LENGTH = 200  # maximum number of tokens that the model can generate
LENGTH_DIFF_PENALTY_STEEPNESS = (
    2  # the steepness of the exponential decay of the length difference penalty
)
QUALITATIVE_SCORE_WEIGHT = 0.82  # weight of the qualitative score in the total score
MODEL_SIZE_SCORE_WEIGHT = 0.06  # weight of the model size score in the total score
LATENCY_SCORE_WEIGHT = 0.06  # weight of the latency score in the total score
VIBE_SCORE_WEIGHT = 0.06  # weight of the vibe score in the total score
MAX_AVG_LATENCY = 10000  # in milliseconds

MAX_MODEL_SIZE = 30 * 1024 * 1024 * 1024  # in bytes
MIN_REPO_SIZE = 10 * 1024 * 1024  # in bytes
MAX_REPO_SIZE = 80 * 1024 * 1024 * 1024  #  in bytes
SAMPLE_SIZE = 1024  # number of samples to evaluate the model from the dataset
BATCH_SIZE = 4  # batch size for evaluation
VOCAB_TRUNCATION = 1000  # truncate the vocab to top n tokens
PROB_TOP_K = 10  # the correct token should be in the top n tokens, else a score of 0 is given to that token
# TODO: this will truncate the sequence to MAX_SEQ_LEN tokens. This is a temporary fix to make the evaluation faster.
MAX_SEQ_LEN = 4096  # maximum sequence length that should be allowed because eval gets really slow with longer sequences than this

MAX_SEQ_LEN_VIBE_SCORE = 2048  # maximum sequence length that should be allowed for vibe score calculation because it is slow with longer sequences than this
BATCH_SIZE_VIBE_SCORE = 4  # batch size for vibe score calculation
SAMPLE_SIZE_VIBE_SCORE = 128  # number of samples to evaluate the model from the dataset for vibe score calculation

SAVE_LEADERBOARD_EVERY = 60  # save the leaderboard every 60 seconds

app = FastAPI()
evaluator = Evaluator()
supabaser = SupabaseState()

logger = logging.getLogger("uvicorn")

logging.basicConfig(level=logging.ERROR)

app.state.leaderboard_update_time = None
app.state.leaderboard = None

admin_key = os.environ["ADMIN_KEY"]
hf_api = HfApi()
chat_template_mappings = {
    "vicuna": "prompt_templates/vicuna_prompt_template.jinja",
    "chatml": "prompt_templates/chatml_prompt_template.jinja",
    "mistral": "prompt_templates/mistral_prompt_template.jinja",
    "zephyr": "prompt_templates/zephyr_prompt_template.jinja",
    "alpaca": "prompt_templates/alpaca_prompt_template.jinja",
    "llama2": "prompt_templates/llama2_prompt_template.jinja",
    "llama3": "prompt_templates/llama3_prompt_template.jinja",
}


def model_evaluation_queue(queue_id):
    try:
        while True:
            _model_evaluation_step()
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


def _model_evaluation_step():
    request = get_next_model_to_eval()
    if request is None:  # Sentinel value to exit the process
        logger.info(
            "No more models to evaluate. Sleep for 15 seconds before checking again."
        )
        return
    logger.info(f"Model evaluation queued: {request}")
    try:
        result = _evaluate_model(request)
        logger.info(f"Model evaluation completed: {result}")
        app.state.event_logger.info(
            "model_eval_queue_complete", result=result, request=request
        )
    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
        app.state.event_logger.info("model_eval_queue_error", error=e)
    finally:
        gc.collect()  # Garbage collect
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Empty CUDA cache


def get_next_model_to_eval():
    response = supabaser.get_next_model_to_eval()
    if response is None:
        return None
    return EvaluateModelRequest(
        repo_namespace=response["repo_namespace"],
        repo_name=response["repo_name"],
        chat_template_type=response["chat_template_type"],
        hash=response["hash"],
    )


def _evaluate_model(
    request: EvaluateModelRequest,
):
    """
    Evaluate a model based on the model size and the quality of the model.
    """
    update_supabase_leaderboard_status(
        request.hash,
        "RUNNING",
        "Model evaluation in progress",
    )

    logger.info("Model evaluation in progress")
    try:
        eval_score_result = evaluator.eval_score(request)
        if isinstance(eval_score_result, RunError):
            raise Exception(eval_score_result.error)
    except Exception as e:
        error_string = f"Error calling eval_score job with message: {e}"
        update_supabase_leaderboard_status(
            request.hash,
            "FAILED",
            error_string,
        )
        raise RuntimeError(error_string)

    eval_score = eval_score_result.eval_score
    latency_score = eval_score_result.latency_score
    model_size_score = eval_score_result.eval_model_size_score

    update_row_supabase(
        {
            "hash": request.hash,
            "model_size_score": model_size_score,
            "qualitative_score": eval_score,
            "latency_score": latency_score,
            "notes": "Now computing vibe score",
        }
    )
    try:
        vibe_score_response = evaluator.vibe_score(request)
        if isinstance(vibe_score_response, RunError):
            raise Exception(vibe_score_response.error)

    except Exception as e:
        error_string = f"Error calling vibe_score job with message: {e}"
        supabaser.update_leaderboard_status(request.hash, "FAILED", error_string)
        raise RuntimeError(error_string)

    vibe_score = vibe_score_response.vibe_score

    if (
        eval_score is None
        or latency_score is None
        or model_size_score is None
        or vibe_score is None
    ):
        raise HTTPException(
            status_code=500,
            detail="Error calculating scores, one or more scores are None",
        )

    total_score = model_size_score * MODEL_SIZE_SCORE_WEIGHT
    total_score += eval_score * QUALITATIVE_SCORE_WEIGHT
    total_score += latency_score * LATENCY_SCORE_WEIGHT
    total_score += vibe_score * VIBE_SCORE_WEIGHT

    try:
        update_row_supabase(
            {
                "hash": request.hash,
                "model_size_score": model_size_score,
                "qualitative_score": eval_score,
                "latency_score": latency_score,
                "vibe_score": vibe_score,
                "total_score": total_score,
                "status": "COMPLETED",
                "notes": "",
            }
        )

    except Exception as e:
        failure_reason = str(e)
        logger.error(f"Updating leaderboard to FAILED: {failure_reason}")
        supabaser.update_leaderboard_status(request.hash, "FAILED", failure_reason)
        raise RuntimeError("Error updating leaderboard: " + failure_reason)
    result = {
        "model_size_score": model_size_score,
        "qualitative_score": eval_score,
        "latency_score": latency_score,
        "vibe_score": vibe_score,
        "total_score": total_score,
    }
    return result


def update_supabase_leaderboard_status(
    hash,
    status,
    notes="",
):
    return supabaser.update_leaderboard_status(hash, status, notes)


def get_json_result(hash):
    return supabaser.get_json_result(hash)


def repository_exists(repo_id):
    try:
        hf_api.repo_info(repo_id)  # 'username/reponame'
        return True
    except RepositoryNotFoundError:
        return False
    except Exception as e:
        app.state.event_logger.error("hf_repo_error", error=e)
        return False


@app.post("/telemetry_report")
def telemetry_report(
    git_commit: str = Header(None, alias="Git-Commit"),
    bittensor_version: str = Header(None, alias="Bittensor-Version"),
    uid: str = Header(None, alias="UID"),
    hotkey: str = Header(None, alias="Hotkey"),
    coldkey: str = Header(None, alias="Coldkey"),
):
    request_details = {
        "git_commit": git_commit,
        "bittensor_version": bittensor_version,
        "uid": uid,
        "hotkey": hotkey,
        "coldkey": coldkey,
    }

    # log incoming request details
    if app.state.event_logger_enabled:
        app.state.event_logger.info("telemetry_request", extra=request_details)
    return Response(status_code=200)


@app.post("/evaluate_model")
def evaluate_model(
    request: EvaluateModelRequest,
    git_commit: str = Header(None, alias="Git-Commit"),
    bittensor_version: str = Header(None, alias="Bittensor-Version"),
    uid: str = Header(None, alias="UID"),
    hotkey: str = Header(None, alias="Hotkey"),
    coldkey: str = Header(None, alias="Coldkey"),
):
    request_details = {
        "git_commit": git_commit,
        "bittensor_version": bittensor_version,
        "uid": uid,
        "hotkey": hotkey,
        "coldkey": coldkey,
    }

    # log incoming request details
    if app.state.event_logger_enabled:
        app.state.event_logger.info("incoming_evaluate_request", extra=request_details)

    # verify hash
    if int(request.hash) != regenerate_hash(
        request.repo_namespace,
        request.repo_name,
        request.chat_template_type,
        request.competition_id,
    ):
        raise HTTPException(
            status_code=400, detail="Hash does not match the model details"
        )

    # check if the model already exists in the leaderboard
    # This needs to be a virtually atomic operation
    current_status = supabaser.get_json_result(request.hash)

    if current_status is None and request.admin_key == admin_key:
        logger.error("QUEUING NEW MODEL")
        failure_notes = ""
        # add the model to leaderboard with status QUEUED
        new_entry_dict = {
            "hash": request.hash,
            "repo_namespace": request.repo_namespace,
            "repo_name": request.repo_name,
            "chat_template_type": request.chat_template_type,
            "model_size_score": -1.0,
            "qualitative_score": -1.0,
            "latency_score": -1.0,
            "vibe_score": -1.0,
            "total_score": -1.0,
            "timestamp": pd.Timestamp.utcnow(),
            "status": "QUEUED",
            "coherence_score": -1.0,
            "notes": failure_notes,
        }

        logger.info("QUEUING: " + str(new_entry_dict))

        update_row_supabase(new_entry_dict)

    # validate the request
    if request.chat_template_type not in chat_template_mappings:
        failure_notes = (
            f"Chat template type not supported: {request.chat_template_type}"
        )
        logger.error(failure_notes)
        supabaser.update_leaderboard_status(request.hash, "FAILED", failure_notes)
        return supabaser.get_json_result(request.hash)
    # validate the repo exists
    repo_id = f"{request.repo_namespace}/{request.repo_name}"
    if not repository_exists(repo_id):
        failure_notes = f"Huggingface repo not public: {repo_id}"
        logger.error(failure_notes)
        supabaser.update_leaderboard_status(request.hash, "FAILED", failure_notes)
        return supabaser.get_json_result(request.hash)

    # check repo size of the model to see if it is within the limit
    try:
        model_repo_size = check_model_repo_size(
            request.hash, request.repo_namespace, request.repo_name
        )
        if model_repo_size is None:
            failure_notes = "Error checking model repo size. Make sure the model repository exists and is accessible."
            logger.error(failure_notes)
            supabaser.update_leaderboard_status(request.hash, "FAILED", failure_notes)
            return supabaser.get_json_result(request.hash)

    except Exception as e:
        failure_notes = f"Error checking model repo size: {e}"
        logger.error(failure_notes)
        supabaser.update_leaderboard_status(request.hash, "FAILED", failure_notes)
        return supabaser.get_json_result(request.hash)

    if model_repo_size > MAX_REPO_SIZE or model_repo_size < MIN_REPO_SIZE:
        failure_notes = f"Model repo size is not up to requirments: {model_repo_size} bytes. Should be less than {MAX_REPO_SIZE} bytes and greater than {MIN_REPO_SIZE} bytes"
        logger.error(failure_notes)
        supabaser.update_leaderboard_status(request.hash, "FAILED", failure_notes)
        return supabaser.get_json_result(request.hash)

    # check model size by checking safetensors index
    model_size = get_model_size(request.repo_namespace, request.repo_name)
    if model_size is None:
        failure_notes = "Error getting model size. Make sure the model.index.safetensors.json file exists in the model repository. And it has the metadata->total_size field."
        logger.error(failure_notes)
        supabaser.update_leaderboard_status(request.hash, "FAILED", failure_notes)
        return supabaser.get_json_result(request.hash)

    if (model_size // 4) > MAX_MODEL_SIZE:
        failure_notes = f"Model size is too large: {model_size} bytes. Should be less than {MAX_MODEL_SIZE} bytes"
        logger.error(failure_notes)
        supabaser.update_leaderboard_status(request.hash, "FAILED", failure_notes)
        return supabaser.get_json_result(request.hash)

    return supabaser.get_json_result(request.hash)


def update_row_supabase(row):
    if "timestamp" in row:
        row["timestamp"] = row["timestamp"].isoformat()

    app.state.supabase_client.table("leaderboard").upsert(row).execute()


@app.get("/leaderboard")
def display_leaderboard():
    return supabaser.get_leaderboard()


def start():
    # add command line arguments for the ports of the two apis
    import argparse

    parser = argparse.ArgumentParser(description="Run the server")
    parser.add_argument(
        "--main-api-port", type=int, default=8000, help="Port for the main API"
    )
    parser.add_argument(
        "--save-remote", action="store_true", default=False, help="Enable remote saving"
    )
    parser.add_argument(
        "--queues",
        type=int,
        default=1,
        help="Specify the number of queues to start (default: 1)",
    )
    args = parser.parse_args()
    num_queues = args.queues
    MAIN_API_PORT = args.main_api_port

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

    processes = []
    stagger_seconds = 2
    try:
        logger.info(f"Starting {num_queues} evaluation threads")
        processes = start_staggered_queues(num_queues, stagger_seconds)
        logger.info("Starting API server")
        uvicorn.run(app, host="0.0.0.0", port=MAIN_API_PORT)
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
