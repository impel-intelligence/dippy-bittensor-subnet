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


BLOCK_RATE_LIMIT = 57600  # Every 57600 blocks = 192 hours
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


@app.post("/telemetry_report")
async def telemetry_report(
    request: Request,
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
    if request is not None:
        try:
            payload = await request.json()
            if not isinstance(payload, dict):
                raise ValueError("Invalid JSON payload - must be a dictionary")
            request_details = {**payload, **request_details}
        except ValueError as e:
            if app.state.event_logger_enabled:
                app.state.event_logger.info(
                    "failed_telemetry_request",
                    extra={**request_details, "error": str(e), "traceback": "Invalid JSON payload format"},
                )
        except Exception as e:
            if app.state.event_logger_enabled:
                app.state.event_logger.info(
                    "failed_telemetry_request",
                    extra={**request_details, "error": str(e), "traceback": traceback.format_exc()},
                )

    # log incoming request details
    if app.state.event_logger_enabled:
        app.state.event_logger.info("telemetry_request", extra=request_details)
    return Response(status_code=200)


class EventData(BaseModel):
    commit: str
    btversion: str
    uid: str
    hotkey: str
    coldkey: str
    payload: Dict[Any, Any]
    signature: Dict[str, Any]

    def _payload_to_dict(self) -> Dict[str, Any]:
        """Convert payload to a JSON serializable dictionary."""
        result = {
            "commit": self.commit,
            "btversion": self.btversion,
            "uid": self.uid,
            "hotkey": self.hotkey,
            "coldkey": self.coldkey,
            "payload": {},
        }

        # Convert payload dict key by key
        if isinstance(self.payload, dict):
            for key, value in self.payload.items():
                try:
                    result["payload"][key] = value
                except Exception as e:
                    logger.error(f"Error converting payload key {key}: {e}")

        return result

    def _signature_to_dict(self) -> Dict[str, Any]:
        """Convert signature to a JSON serializable dictionary."""
        result = {"signature": {}}

        # Convert signature dict key by key
        if isinstance(self.signature, dict):
            for key, value in self.signature.items():
                try:
                    result["signature"][key] = value
                except Exception as e:
                    logger.error(f"Error converting signature key {key}: {e}")

        return result

    def to_dict(self) -> Dict[str, Any]:
        """Convert EventData to a JSON serializable dictionary."""
        return {**self._payload_to_dict(), **self._signature_to_dict()}


@app.post("/event_report")
async def event_report(event_data: EventData):
    try:
        if app.state.event_logger_enabled:
            app.state.event_logger.info("event_request", extra=event_data.to_dict())
        return Response(status_code=200)
    except Exception as e:
        if app.state.event_logger_enabled:
            error_details = {"error": str(e), "traceback": traceback.format_exc()}
            app.state.event_logger.error("failed_event_request", extra=error_details)
        return Response(status_code=400, content={"error": str(e)})


class MinerboardRequest(BaseModel):
    uid: int
    hotkey: str
    hash: str
    block: int
    admin_key: Optional[str] = "admin_key"


@app.post("/minerboard_update")
def minerboard_update(
    request: MinerboardRequest,
):
    if request.admin_key != admin_key:
        return Response(status_code=403)

    supabaser.update_minerboard_status(
        minerhash=request.hash,
        uid=request.uid,
        hotkey=request.hotkey,
        block=request.block,
    )
    return Response(status_code=200)


@app.get("/minerboard")
def get_minerboard():
    entries = supabaser.minerboard_fetch()
    if len(entries) < 1:
        return []
    results = []
    for entry in entries:
        flattened_entry = {**entry.pop("leaderboard"), **entry}
        results.append(flattened_entry)
    return results


def hash_check(request: EvaluateModelRequest) -> bool:
    hotkey_hash_matches = int(request.hash) == regenerate_hash(
        request.repo_namespace,
        request.repo_name,
        request.chat_template_type,
        request.hotkey,
    )
    return hotkey_hash_matches


@app.post("/evaluate_model")
def evaluate_model(
    request: EvaluateModelRequest,
    git_commit: str = Header(None, alias="Git-Commit"),
    bittensor_version: str = Header(None, alias="Bittensor-Version"),
    uid: str = Header(None, alias="UID"),
    hotkey: str = Header(None, alias="Hotkey"),
    coldkey: str = Header(None, alias="Coldkey"),
    signed_payload: str = Header(None, alias="signed_payload"),
    miner_hotkey: str = Header(None, alias="miner_hotkey"),
):
    request_details = {
        "git_commit": git_commit,
        "bittensor_version": bittensor_version,
        "uid": uid,
        "hotkey": hotkey,
        "coldkey": coldkey,
        "signed_payload": signed_payload,
        "miner_hotkey": miner_hotkey,
    }
    # verify hash
    hash_verified = hash_check(request)
    if not hash_verified:
        raise HTTPException(status_code=400, detail="Hash does not match the model details")

    return supabaser.get_json_result(request.hash)


"""
curl -X GET "http://URL.com/model_submission_details?repo_namespace=my-org&repo_name=my-model&chat_template_type=chatml&hash=12345678&competition_id=comp-1"
Used by validators to get the model details
"""


@app.get("/model_submission_details")
# Example curl request:
def get_model_submission_details(
    repo_namespace: str,
    repo_name: str,
    chat_template_type: str,
    hash: str,
    competition_id: Optional[str] = None,
    hotkey: Optional[str] = None,
):
    request = EvaluateModelRequest(
        repo_namespace=repo_namespace,
        repo_name=repo_name,
        chat_template_type=chat_template_type,
        hash=hash,
        hotkey=hotkey,
    )
    # verify hash
    hash_verified = hash_check(request)
    if not hash_verified:
        raise HTTPException(status_code=400, detail="Hash does not match the model details")

    return supabaser.get_json_result(request.hash)


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


INVALID_BLOCK_START = 1
INVALID_BLOCK_END = 5207777


@app.post("/check_model")
def check_or_create_model(
    request: EvaluateModelRequest,
):
    # verify hash
    hash_verified = hash_check(request)
    if not hash_verified:
        raise HTTPException(status_code=400, detail="Hash does not match the model details")
    if request.admin_key != admin_key:
        raise HTTPException(status_code=403, detail="invalid key")

    early_failure = False
    failure_notes = ""
    # validate the repo exists
    repo_id = f"{request.repo_namespace}/{request.repo_name}"
    if not repository_exists(repo_id):
        failure_notes = f"Huggingface repo not public: {repo_id}"
        early_failure = True

    # only check if the model already exists in the leaderboard
    # update state if repo not public
    # This needs to be a virtually atomic operation
    try:
        current_entry = supabaser.get_json_result(request.hash)
        if current_entry is not None and early_failure:
            logger.error(failure_notes)
            internal_entry = supabaser.get_internal_result(request.hash)
            internal_entry = update_failure(internal_entry, failure_notes)
            return supabaser.upsert_and_return(internal_entry, request.hash)
        if current_entry is not None:
            return current_entry

    except Exception as e:
        logger.error(f"error while fetching request {request} : {e}")
        return None
    logger.info(f"COULD NOT FIND EXISTING MODEL for {request} : QUEUING NEW MODEL")

    # add the model to leaderboard with status QUEUED
    new_entry_dict = {
        "hash": request.hash,
        "repo_namespace": request.repo_namespace,
        "repo_name": request.repo_name,
        "chat_template_type": request.chat_template_type,
        "model_size_score": 0,
        "qualitative_score": 0,
        "creativity_score": 0,
        "latency_score": 0,
        "vibe_score": 0,
        "judge_score": 0,
        "total_score": 0,
        "timestamp": pd.Timestamp.utcnow(),
        "status": StatusEnum.QUEUED,
        "coherence_score": 0,
        "notes": failure_notes,
    }

    if early_failure:
        logger.error(failure_notes)
        updated_entry = update_failure(new_entry_dict, failure_notes)
        return supabaser.upsert_and_return(updated_entry, request.hash)

    logger.info("QUEUING: " + str(new_entry_dict))

    last_model = supabaser.last_uploaded_model(request.hotkey)
    if last_model is not None:
        last_model_status = StatusEnum.from_string(last_model["leaderboard"]["status"])
        if last_model_status != StatusEnum.FAILED:
            last_block = last_model.get("block", request.block)
            current_block = request.block
            # eg block 3001 - 2001 = 1000
            if abs(current_block - last_block) < BLOCK_RATE_LIMIT and abs(current_block - last_block) > 0:
                failure_notes = f"""
                Exceeded rate limit. 
                Last submitted model was block {last_block} with details {last_model}.
                Block difference between last block {last_block} and current block {current_block} is {(current_block - last_block)}
                Current submission {current_block} exceeds minimum block limit {BLOCK_RATE_LIMIT}"""
                logger.error(failure_notes)
                new_entry_dict = update_failure(new_entry_dict, failure_notes)

    # validate the request
    if request.chat_template_type not in chat_template_mappings:
        failure_notes = f"Chat template type not supported: {request.chat_template_type}"
        logger.error(failure_notes)
        new_entry_dict = update_failure(new_entry_dict, failure_notes)

    # validate the repo exists
    repo_id = f"{request.repo_namespace}/{request.repo_name}"
    if not repository_exists(repo_id):
        failure_notes = f"Huggingface repo not public: {repo_id}"
        logger.error(failure_notes)
        new_entry_dict = update_failure(new_entry_dict, failure_notes)
        return supabaser.upsert_and_return(new_entry_dict, request.hash)

    if INVALID_BLOCK_START < request.block < INVALID_BLOCK_END:
        failure_notes = f"{INVALID_BLOCK_START} < {request.block} < {INVALID_BLOCK_END}"
        logger.error(failure_notes)
        new_entry_dict = update_failure(new_entry_dict, failure_notes)
        return supabaser.upsert_and_return(new_entry_dict, request.hash)

    # check repo size of the model to see if it is within the limit
    try:
        model_repo_details = check_model_repo_details(request.hash, request.repo_namespace, request.repo_name)
        if model_repo_details is None:
            failure_notes = "Error checking model repo size. Make sure the model repository exists and is accessible."
            logger.error(failure_notes)
            new_entry_dict = update_failure(new_entry_dict, failure_notes)
            return supabaser.upsert_and_return(new_entry_dict, request.hash)
        model_repo_size = model_repo_details.repo_size
        new_entry_dict["model_hash"] = model_repo_details.model_hash
        if model_repo_size > MAX_REPO_SIZE or model_repo_size < MIN_REPO_SIZE:
            failure_notes = f"Model repo size is not up to requirements: {model_repo_size} bytes. Should be less than {MAX_REPO_SIZE} bytes and greater than {MIN_REPO_SIZE} bytes"
            logger.error(failure_notes)
            new_entry_dict = update_failure(new_entry_dict, failure_notes)
            return supabaser.upsert_and_return(new_entry_dict, request.hash)
    except Exception as e:
        failure_notes = f"Error checking model repo size: {e}"
        logger.error(failure_notes)
        new_entry_dict = update_failure(new_entry_dict, failure_notes)
        return supabaser.upsert_and_return(new_entry_dict, request.hash)

    # check model size by checking safetensors index
    model_size = get_model_size(request.repo_namespace, request.repo_name)
    if model_size is None:
        failure_notes = "Error getting model size. Make sure the model.index.safetensors.json file exists in the model repository. And it has the metadata->total_size field."
        logger.error(failure_notes)
        new_entry_dict = update_failure(new_entry_dict, failure_notes)
        return supabaser.upsert_and_return(new_entry_dict, request.hash)

    if (model_size // 4) > MAX_MODEL_SIZE:
        failure_notes = f"Model size is too large: {model_size} bytes. Should be less than {MAX_MODEL_SIZE} bytes"
        logger.error(failure_notes)
        new_entry_dict = update_failure(new_entry_dict, failure_notes)
        return supabaser.upsert_and_return(new_entry_dict, request.hash)

    existing_record = supabaser.search_record_with_model_hash(
        new_entry_dict.get("model_hash", ""),
        request.block,
    )

    if "model_hash" in new_entry_dict and existing_record is not None:
        failure_notes = (
            f"Given entry {new_entry_dict} has conflicting model_hash with existing record {existing_record}"
        )
        logger.error(failure_notes)
        new_entry_dict = update_completed(new_entry_dict, failure_notes)
        return supabaser.upsert_and_return(new_entry_dict, request.hash)

    return supabaser.upsert_and_return(new_entry_dict, request.hash)


def upsert_row_supabase(row):
    app.state.supabase_client.table("leaderboard").upsert(row).execute()


@app.get("/hc")
def hc():
    return {"g": True}


def start():
    import argparse

    parser = argparse.ArgumentParser(description="Run the server")
    parser.add_argument("--main-api-port", type=int, default=8000, help="Port for the main API")
    args = parser.parse_args()
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
