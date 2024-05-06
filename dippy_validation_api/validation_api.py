import gc
import time
import requests
from typing import Any, Optional
import os
import multiprocessing
import argparse
from pydantic import BaseModel

import pandas as pd
import uvicorn
import torch
from fastapi import FastAPI, HTTPException
from fastapi.logger import logger
from supabase import create_client

from utilities.validation_utils import regenerate_hash, check_model_repo_size, get_model_size

from dotenv import load_dotenv

load_dotenv()

# Constants
MAX_GENERATION_LEEWAY = 0.5 # should be between 0 and 1. This is the percentage of tokens that the model can generate more than the last user message
MAX_GENERATION_LENGTH = 200 # maximum number of tokens that the model can generate
LENGTH_DIFF_PENALTY_STEEPNESS = 2 # the steepness of the exponential decay of the length difference penalty
QUALITATIVE_SCORE_WEIGHT = 0.82 # weight of the qualitative score in the total score
MODEL_SIZE_SCORE_WEIGHT = 0.06 # weight of the model size score in the total score
LATENCY_SCORE_WEIGHT = 0.06 # weight of the latency score in the total score
VIBE_SCORE_WEIGHT = 0.06 # weight of the vibe score in the total score
MAX_AVG_LATENCY = 10000 # in milliseconds

MAX_MODEL_SIZE = 30 * 1024 * 1024 * 1024 # in bytes
MAX_REPO_SIZE = 80 * 1024 * 1024 * 1024 #  in bytes
SAMPLE_SIZE = 1024 # number of samples to evaluate the model from the dataset
BATCH_SIZE = 4 # batch size for evaluation
VOCAB_TRUNCATION = 1000 # truncate the vocab to top n tokens
PROB_TOP_K = 10 # the correct token should be in the top n tokens, else a score of 0 is given to that token
# TODO: this will truncate the sequence to MAX_SEQ_LEN tokens. This is a temporary fix to make the evaluation faster.
MAX_SEQ_LEN = 4096 # maximum sequence length that should be allowed because eval gets really slow with longer sequences than this

MAX_SEQ_LEN_VIBE_SCORE = 2048 # maximum sequence length that should be allowed for vibe score calculation because it is slow with longer sequences than this
BATCH_SIZE_VIBE_SCORE = 4 # batch size for vibe score calculation
SAMPLE_SIZE_VIBE_SCORE = 128 # number of samples to evaluate the model from the dataset for vibe score calculation

SAVE_REMOTE = True # Save the leaderboard to Supabase 

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase_client = create_client(supabase_url, supabase_key)

leaderboard_file = 'leaderboard.csv'

# if the leaderboard file does not exist, create it with proper columns
columns = ['hash', 'repo_namespace', 'repo_name', 'chat_template_type', 'model_size_score', 'qualitative_score', 'latency_score', 'vibe_score', 'total_score', 'timestamp', 'status', 'notes']
if not os.path.exists(leaderboard_file):
    # fetch from supabase
    try:
        if SAVE_REMOTE:
            leaderboard = supabase_client.table("leaderboard").select("*").execute().get('data')
            if leaderboard:
                leaderboard = pd.DataFrame(leaderboard)
                leaderboard.to_csv(leaderboard_file, index=False)
            else:
                raise Exception("No data found in Supabase")
        else:
            leaderboard = pd.DataFrame(columns=columns)
            leaderboard.to_csv(leaderboard_file, index=False)
    except Exception as e:
        logger.error(f"Error fetching leaderboard from Supabase: {e}")
        leaderboard = pd.DataFrame(columns=columns)
        leaderboard.to_csv(leaderboard_file, index=False)

class EvaluateModelRequest(BaseModel):
    repo_namespace: str
    repo_name: str
    chat_template_type: str
    hash: str
    revision: Optional[str] = "main"
    competition_id: Optional[str] = "d1"

app = FastAPI()

chat_template_mappings = {
    "vicuna": "prompt_templates/vicuna_prompt_template.jinja",
    "chatml": "prompt_templates/chatml_prompt_template.jinja",
    "mistral": "prompt_templates/mistral_prompt_template.jinja",
    "zephyr": "prompt_templates/zephyr_prompt_template.jinja",
    "alpaca": "prompt_templates/alpaca_prompt_template.jinja",
}

def get_leaderboard() -> pd.DataFrame:
    dtype_dict = {
        'hash': str,
        'repo_namespace': str,
        'repo_name': str,
        'chat_template_type': str,
        'model_size_score': 'float64',  # Use 'float64' to allow NaNs
        'qualitative_score': 'float64',  # Use 'float64' to allow NaNs
        'latency_score': 'float64',  # Use 'float64' to allow NaNs
        'vibe_score': 'float64',  # Use 'float64' to allow NaNs
        'total_score': 'float64',  # Use 'float64' to allow NaNs
        'timestamp': str,
        'status': str,
        'notes': str
    }
    leaderboard = pd.read_csv(leaderboard_file, dtype=dtype_dict, parse_dates=['timestamp'])
    # Replace NaN with None for JSON serialization
    leaderboard = leaderboard.where(pd.notnull(leaderboard), None)
    return leaderboard

def save_leaderboard(leaderboard: pd.DataFrame, hash=None, save_remote=SAVE_REMOTE):
    leaderboard.to_csv(leaderboard_file, index=False)
    if hash is not None:
        leaderboard_row = leaderboard[leaderboard['hash'] == hash].iloc[0]
        if save_remote:
            try:
                supabase_client.table("leaderboard").upsert({"hash": leaderboard_row['hash'], "repo_namespace": leaderboard_row['repo_namespace'], "repo_name": leaderboard_row['repo_name'], "chat_template_type": leaderboard_row['chat_template_type'], "model_size_score": leaderboard_row['model_size_score'], "qualitative_score": leaderboard_row['qualitative_score'], "latency_score": leaderboard_row['latency_score'], "total_score": leaderboard_row['total_score'], "status": leaderboard_row['status'], "notes": leaderboard_row['notes']}).execute()
            except Exception as e:
                logger.error(f"Error saving leaderboard row to Supabase: {e}")

def model_evaluation_worker(evaluation_queue):
    while True:
        request = evaluation_queue.get()
        if request is None:  # Sentinel value to exit the process
            break
        try:
            with torch.no_grad():  # Disable gradient calculation
                result = evaluate_model_logic(request)
                logger.info(f"Model evaluation completed: {result}")
        except Exception as e:
            logger.error(f"Error during model evaluation: {e}")
        finally:
            gc.collect()  # Garbage collect
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # Empty CUDA cache

        
def evaluate_model_logic(request: EvaluateModelRequest):
    """
    Evaluate a model based on the model size and the quality of the model.
    """
    leaderboard = get_leaderboard()
    if not (leaderboard['hash'] == request.hash).any():
        logger.debug(leaderboard)
        logger.debug(leaderboard['hash'])
        logger.debug(type(leaderboard['hash']))
        logger.debug(request.hash)
        logger.debug(type(request.hash))
        raise ValueError(f"Model {request.hash} not found in the leaderboard")
    
    # changed status to in progress
    update_leaderboard_status(request.hash, "RUNNING", "Model evaluation in progress")
    logger.info("Model evaluation in progress")

    start_time = time.time()
    while True:
        try:
            eval_score_response = requests.post("http://localhost:8001/eval_score", json=request.model_dump())
            if eval_score_response.status_code == 200:
                logger.info("eval_score API call successful")
                break
            else:
                raise RuntimeError(f"Error calling eval_score API: {eval_score_response.content}")
        except Exception as e:
            if time.time() - start_time > 30:
                # update leaderboard status to failed
                update_leaderboard_status(request.hash, "FAILED", "Error calling eval_score API with message: " + eval_score_response.content)
                try:
                    shutdown_response = requests.post("http://localhost:8001/shutdown", timeout=1)
                except Exception as e:
                    pass
                raise RuntimeError(f"Error calling eval_score API: {eval_score_response.content}")
        
        time.sleep(1)  # Wait for 1 second before retrying
    
    # Call the shutdown endpoint to restart the eval_score_api for the next evaluation to avoid memory leaks that were observed with loading and unloading different models
    logger.info("Shutting down eval_score_api")
    try:
        shutdown_response = requests.post("http://localhost:8001/shutdown", timeout=1)
    except Exception as e:
        pass
    logger.info("vibe_score_api shutdown initiated for restart.")
    
    eval_score_data = eval_score_response.json()
    eval_score = eval_score_data["eval_score"]
    latency_score = eval_score_data["latency_score"]
    model_size_score = eval_score_data["model_size_score"]

    # Call the vibe_score API
    start_time = time.time()
    while True:
        vibe_score_response = requests.post("http://localhost:8002/vibe_match_score", json=request.model_dump())
        if vibe_score_response.status_code == 200:
            break
        elif time.time() - start_time > 30:
            # update leaderboard status to failed
            try:
                shutdown_response = requests.post("http://localhost:8002/shutdown", timeout=1)
            except Exception as e:
                pass
            update_leaderboard_status(request.hash, "FAILED", "Error calling vibe_score API with message: " + vibe_score_response.content)
            raise HTTPException(status_code=500, detail=f"Error calling vibe_score API: {vibe_score_response.content}")
        time.sleep(1)  # Wait for 1 second before retrying
    
    # Call the shutdown endpoint to restart the vibe_score_api for the next evaluation to avoid memory leaks that were observed with loading and unloading different models

    try:
        shutdown_response = requests.post("http://localhost:8002/shutdown", timeout=1)
    except Exception as e:
        pass

    vibe_score = vibe_score_response.json()["vibe_score"]

    if eval_score is None or latency_score is None or model_size_score is None or vibe_score is None:
        raise HTTPException(status_code=500, detail="Error calculating scores, one or more scores are None")
    
    total_score = model_size_score * MODEL_SIZE_SCORE_WEIGHT
    total_score += eval_score * QUALITATIVE_SCORE_WEIGHT
    total_score += latency_score * LATENCY_SCORE_WEIGHT
    total_score += vibe_score * VIBE_SCORE_WEIGHT

    try:
        leaderboard = get_leaderboard()
        leaderboard.loc[leaderboard['hash'] == request.hash, 'model_size_score'] = float(model_size_score)
        leaderboard.loc[leaderboard['hash'] == request.hash, 'qualitative_score'] = float(eval_score)
        leaderboard.loc[leaderboard['hash'] == request.hash, 'latency_score'] = float(latency_score)
        leaderboard.loc[leaderboard['hash'] == request.hash, 'vibe_score'] = float(vibe_score)
        leaderboard.loc[leaderboard['hash'] == request.hash, 'total_score'] = float(total_score)
        leaderboard.loc[leaderboard['hash'] == request.hash, 'status'] = "COMPLETED"
        leaderboard.loc[leaderboard['hash'] == request.hash, 'notes'] = ""
        save_leaderboard(leaderboard, request.hash, SAVE_REMOTE)
    except Exception as e:
        failure_reason = str(e)
        update_leaderboard_status(request.hash, "FAILED", failure_reason)
        raise RuntimeError("Error updating leaderboard: " + failure_reason)
    
    return {
        "model_size_score": model_size_score,
        "qualitative_score": eval_score,
        "latency_score": latency_score,
        "vibe_score": vibe_score,
        "total_score": total_score
    }

def update_leaderboard_status(hash, status, notes=""):
    try:
        leaderboard = get_leaderboard()
        leaderboard.loc[leaderboard['hash'] == hash, 'status'] = status
        if notes:
            leaderboard.loc[leaderboard['hash'] == hash, 'notes'] = notes
        save_leaderboard(leaderboard, hash, SAVE_REMOTE)
    except Exception as e:
        logger.error(f"Error updating leaderboard status for {hash}: {e}")

def get_json_result(hash):
    leaderboard = get_leaderboard()
    if (leaderboard['hash'] == hash).any():
        # if it exists, return score and status
        model_entry = leaderboard[leaderboard['hash'] == hash].iloc[0]
        
        return {
            "score": {
                "model_size_score": model_entry['model_size_score'],
                "qualitative_score": model_entry['qualitative_score'],
                "latency_score": model_entry['latency_score'],
                "vibe_score": model_entry['vibe_score'],
                "total_score": model_entry['total_score']
            },
            "status": model_entry['status']
        }
    else:
        None


@app.post("/evaluate_model")
def evaluate_model(request: EvaluateModelRequest):
    # verfify hash
    if int(request.hash) != regenerate_hash(request.repo_namespace, request.repo_name, request.chat_template_type, request.competition_id):
        logger.error(f"Hash does not match the expected hash: {request.hash} != {regenerate_hash(request.repo_namespace, request.repo_name, request.chat_template_type, request.competition_id)}")
        raise HTTPException(status_code=400, detail="Hash does not match the expected hash")
    # read the leaderboard file

    # check if the model already exists in the leaderboard
    current_status = get_json_result(request.hash)
    if current_status is not None:
        return current_status

    # validate the request
    if request.chat_template_type not in chat_template_mappings:
        logger.error(f"Chat template type not supported: {request.chat_template_type}")
        raise HTTPException(status_code=400, detail="Chat template type not supported")
    
    # check repo size of the model to see if it is within the limit
    try:
        model_repo_size = check_model_repo_size(request.hash, request.repo_namespace, request.repo_name)
        if model_repo_size is None:
            logger.error("Error checking model repo size")
            raise HTTPException(status_code=400, detail="Error occured while checking model repo size on Hugging Face Hub.")
    except Exception as e:
        logger.error(f"Error checking model repo size: {e}")
        raise HTTPException(status_code=400, detail=f'"{request.repo_namespace}/{request.repo_name}" is probably a gated model, or it does not exist on the Hugging Face Hub.')
    
    if model_repo_size > MAX_REPO_SIZE:
        logger.error(f"Model repo size is too large: {model_repo_size} bytes. Should be less than {MAX_REPO_SIZE} bytes")
        raise HTTPException(status_code=400, detail="Model repo size is too large: " + str(model_repo_size) + " bytes. Should be less than " + str(MAX_REPO_SIZE) + " bytes")
    
    # check model size by checking safetensors index
    model_size = get_model_size(request.repo_namespace, request.repo_name)
    if model_size is None:
        model_size = 0
        # raise HTTPException(status_code=400, detail="Error getting model size. Make sure the model.index.safetensors.json file exists in the model repository. And it has the metadata->total_size field.")

    if (model_size // 4) > MAX_MODEL_SIZE:
        logger.error(f"Model size is too large: {model_size} bytes. Should be less than {MAX_MODEL_SIZE} bytes")
        raise HTTPException(status_code=400, detail="Model size is too large: " + str(model_size) + " bytes. Should be less than " + str(MAX_MODEL_SIZE) + " bytes")

    leaderboard = get_leaderboard()
    # add the model to leaderboard with status pending
    new_entry = pd.DataFrame([{
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
        "notes": ""
    }])
    leaderboard = pd.concat([leaderboard, new_entry], ignore_index=True)
    save_leaderboard(leaderboard, request.hash, SAVE_REMOTE)
    
    # Add the evaluation task to the queue
    evaluation_queue.put(request)

    logger.info('returning result')
    return get_json_result(request.hash)

@app.get("/leaderboard")
def display_leaderboard():
    return get_leaderboard().to_dict(orient='records')

if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn', force=True) # need to fo
    except RuntimeError as e:
        logger.warning(f"Warning: multiprocessing context has already been set. Details: {e}")
    
    evaluation_queue = multiprocessing.Queue()

    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Run the server")
    parser.add_argument("--no-remote", action="store_false", help="Disable remote saving")
    args = parser.parse_args()

    SAVE_REMOTE = args.no_remote

    try:
        logger.info("Starting evaluation thread")
        evaluation_process = multiprocessing.Process(target=model_evaluation_worker, args=(evaluation_queue,))
        evaluation_process.start()
        logger.info("Starting API server")
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, stopping...")
    except Exception as e:
        logger.error(f"An exception occurred: {e}")
    finally:
        logger.info("Stopping evaluation thread")
        # empty the queue
        while not evaluation_queue.empty():
            evaluation_queue.get()
        
        # remove any rows with status QUEUED
        leaderboard = get_leaderboard()
        leaderboard = leaderboard[leaderboard['status'] != 'QUEUED']
        save_leaderboard(leaderboard, None, SAVE_REMOTE)
        # add a sentinel to the queue to stop the thread
        evaluation_queue.put(None)
        evaluation_process.join()

        # remove any RUNNING status
        leaderboard = get_leaderboard()
        leaderboard = leaderboard[leaderboard['status'] != 'RUNNING']
        save_leaderboard(leaderboard, None, SAVE_REMOTE)
        logger.info("API server and evaluation thread have been stopped")
