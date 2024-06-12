import gc
import time
import os
import multiprocessing
import logging
from datetime import datetime, timedelta
from typing import Optional
from pydantic import BaseModel
import requests
import uvicorn

import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from supabase import create_client
import huggingface_hub
import shutil
from dotenv import load_dotenv

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
MIN_REPO_SIZE = 10 * 1024 * 1024 # in bytes
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

SAVE_LEADERBOARD_EVERY = 60 # save the leaderboard every 60 seconds

EVAL_SCORE_PORT = 8001 # default port for the eval_score API
VIBE_SCORE_PORT = 8002 # default port for the vibe_score API


leaderboard_file = 'leaderboard.csv'

class EvaluateModelRequest(BaseModel):
    repo_namespace: str
    repo_name: str
    chat_template_type: str
    hash: str
    revision: Optional[str] = "main"
    competition_id: Optional[str] = "d1"
    admin_key: Optional[str] = "admin_key"

app = FastAPI()

logger = logging.getLogger("uvicorn")

logging.basicConfig(level=logging.ERROR)

app.state.leaderboard_update_time = None
app.state.leaderboard = None

admin_key = os.environ['ADMIN_KEY']

chat_template_mappings = {
    "vicuna": "prompt_templates/vicuna_prompt_template.jinja",
    "chatml": "prompt_templates/chatml_prompt_template.jinja",
    "mistral": "prompt_templates/mistral_prompt_template.jinja",
    "zephyr": "prompt_templates/zephyr_prompt_template.jinja",
    "alpaca": "prompt_templates/alpaca_prompt_template.jinja",
    "llama2": "prompt_templates/llama2_prompt_template.jinja",
    "llama3": "prompt_templates/llama3_prompt_template.jinja",
}


def save_leaderboard(leaderboard: pd.DataFrame):
    leaderboard.to_csv(leaderboard_file, index=False)

def model_evaluation_worker():
    while True:
        # request = evaluation_queue.get()
        request = get_next_model_to_eval()
        if request is None:  # Sentinel value to exit the process
            logger.info("No more models to evaluate. Sleep for 5 seconds before checking again.")
            time.sleep(5)
            continue
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

def get_next_model_to_eval():
    try:
        response = app.state.supabase_client.table("leaderboard").select("*").eq("status", "QUEUED").order("timestamp", desc=False).limit(1).execute()
        if len(response.data) == 0:
            return None
    except Exception as e:
        logger.error(f"Error fetching next model to evaluate: {e}")
        return None
    return EvaluateModelRequest(repo_namespace=response.data[0]["repo_namespace"], repo_name=response.data[0]["repo_name"], chat_template_type=response.data[0]["chat_template_type"], hash=response.data[0]["hash"])
        
def evaluate_model_logic(request: EvaluateModelRequest):
    """
    Evaluate a model based on the model size and the quality of the model.
    """
    update_supabase_leaderboard_status(request.hash, "RUNNING", "Model evaluation in progress")
    
    logger.info("Model evaluation in progress")
    start_time = time.time()
    eval_score_response = None
    while True:
        try:
            eval_score_response = requests.post(f"http://localhost:{EVAL_SCORE_PORT}/eval_score", json=request.model_dump())
            if eval_score_response.status_code == 200:
                logger.info("eval_score API call successful")
                break
            else:
                clean_up(request)
                raise RuntimeError(f"Error calling eval_score API: {eval_score_response.content}")
        except Exception as e:
            if time.time() - start_time > 30:
                error_string = f"Error calling eval_score API with message: {eval_score_response.content if eval_score_response else e}"
                
                update_supabase_leaderboard_status(request.hash, "FAILED", error_string)
                
                try:
                    shutdown_response = requests.post(f"http://localhost:{EVAL_SCORE_PORT}/shutdown", timeout=1)
                except Exception as shutdown_error:
                    pass
                
                clean_up(request)
                raise RuntimeError(error_string)
        
        time.sleep(1)  # Wait for 1 second before retrying
    
    # Call the shutdown endpoint to restart the eval_score_api for the next evaluation to avoid memory leaks that were observed with loading and unloading different models
    logger.info("Shutting down eval_score_api")
    try:
        shutdown_response = requests.post(f"http://localhost:{EVAL_SCORE_PORT}/shutdown", timeout=1)
    except Exception as e:
        pass
    logger.info("vibe_score_api shutdown initiated for restart.")
    
    eval_score_data = eval_score_response.json()
    eval_score = eval_score_data["eval_score"]
    latency_score = eval_score_data["latency_score"]
    model_size_score = eval_score_data["model_size_score"]

    update_row_supabase({"hash": request.hash, "model_size_score": model_size_score, "qualitative_score": eval_score, "latency_score": latency_score, "notes": "Now computing vibe score"})


    # Call the vibe_score API
    start_time = time.time()
    vibe_score_response = None
    while True:
        try:
            vibe_score_response = requests.post(f"http://localhost:{VIBE_SCORE_PORT}/vibe_match_score", json=request.model_dump())
            if vibe_score_response.status_code == 200:
                logger.info("vibe_score API call successful")
                break
            else:
                clean_up(request)
                raise RuntimeError(f"Error calling vibe_score API: {vibe_score_response.content}")
        except Exception as e:
            if time.time() - start_time > 30:
                error_string = f"Error calling vibe_score API with message: {vibe_score_response.content if vibe_score_response else e}"
                update_supabase_leaderboard_status(request.hash, "FAILED", error_string)
                
                try:
                    shutdown_response = requests.post(f"http://localhost:{VIBE_SCORE_PORT}/shutdown", timeout=1)
                except Exception as shutdown_error:
                    logger.error(f"Error during vibe_score_api shutdown: {shutdown_error}")
                clean_up(request)
                raise RuntimeError(error_string)
        
        time.sleep(1)  # Wait for 1 second before retrying
    
    # Call the shutdown endpoint to restart the vibe_score_api for the next evaluation to avoid memory leaks that were observed with loading and unloading different models
    logger.info("Shutting down vibe_score_api")
    try:
        shutdown_response = requests.post(f"http://localhost:{VIBE_SCORE_PORT}/shutdown", timeout=1)
    except Exception as e:
        pass

    vibe_score = vibe_score_response.json()["vibe_score"]

    if eval_score is None or latency_score is None or model_size_score is None or vibe_score is None:
        clean_up(request)
        raise HTTPException(status_code=500, detail="Error calculating scores, one or more scores are None")
    
    total_score = model_size_score * MODEL_SIZE_SCORE_WEIGHT
    total_score += eval_score * QUALITATIVE_SCORE_WEIGHT
    total_score += latency_score * LATENCY_SCORE_WEIGHT
    total_score += vibe_score * VIBE_SCORE_WEIGHT

    try:
        update_row_supabase({"hash": request.hash, "model_size_score": model_size_score, "qualitative_score": eval_score, "latency_score": latency_score, "vibe_score": vibe_score, "total_score": total_score, "status": "COMPLETED", "notes": ""})
        
    except Exception as e:
        failure_reason = str(e)
        logger.error(f"Updating leaderboard to FAILED: {failure_reason}")
        update_supabase_leaderboard_status(request.hash, "FAILED", failure_reason)
        
        clean_up(request)
        raise RuntimeError("Error updating leaderboard: " + failure_reason)
    

    clean_up(request)
    return {
        "model_size_score": model_size_score,
        "qualitative_score": eval_score,
        "latency_score": latency_score,
        "vibe_score": vibe_score,
        "total_score": total_score
    }

def clean_up(request):
    repo_id = f"{request.repo_namespace}/{request.repo_name}"
    hf_cache_info = huggingface_hub.scan_cache_dir()
    # delete from huggingface cache
    try:
        for repo_info in hf_cache_info.repos:
            if repo_info.repo_id == repo_id:
                upload_to_hf(repo_info.repo_path, f"{request.repo_namespace}_{request.repo_name}")
                shutil.rmtree(repo_info.repo_path)
                logger.info(f"Deleted {repo_info.repo_path} from huggingface cache")
    except Exception as e:
        logger.error(f"Error cleaning up: {e}")

def update_supabase_leaderboard_status(hash, status, notes=""):
    try:
        response = app.state.supabase_client.table("leaderboard").upsert({"hash": hash, "status": status, "notes": notes}, returning="minimal").execute()
    except Exception as e:
        logger.error(f"Error updating leaderboard status for {hash}: {e}")


def get_json_result(hash):
    try:
        response = app.state.supabase_client.table("leaderboard").select("*").eq("hash", hash).execute()
        if len(response.data) > 0:
            return {
            "score": {
                "model_size_score": response.data[0]['model_size_score'],
                "qualitative_score": response.data[0]['qualitative_score'],
                "latency_score": response.data[0]['latency_score'],
                "vibe_score": response.data[0]['vibe_score'],
                "total_score": response.data[0]['total_score']
            },
            "status": response.data[0]['status']
        }
        raise RuntimeError('No record QUEUED')
    except Exception as e:
        logger.error(f"Error fetching leaderboard from database: {e}")
        return None

@app.post("/evaluate_model")
def evaluate_model(request: EvaluateModelRequest):
    # verify hash
    if int(request.hash) != regenerate_hash(request.repo_namespace, request.repo_name, request.chat_template_type, request.competition_id):
        raise HTTPException(status_code=400, detail="Hash does not match the model details")

    # check if the model already exists in the leaderboard
    # This needs to be a virtually atomic operation
    current_status = get_json_result(request.hash)
    
    if current_status is None and request.admin_key == admin_key:
        logger.error('QUEUING NEW MODEL')
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
            "notes": ""
        }

        logger.info('QUEUING: ' + str(new_entry_dict))

        update_row_supabase(new_entry_dict)
    else:
        return current_status

    # validate the request
    if request.chat_template_type not in chat_template_mappings:
        failure_notes = f"Chat template type not supported: {request.chat_template_type}"
        logger.error(failure_notes)
        update_supabase_leaderboard_status(request.hash, "FAILED", failure_notes)
        return get_json_result(request.hash)

    # check repo size of the model to see if it is within the limit
    try:
        model_repo_size = check_model_repo_size(request.hash, request.repo_namespace, request.repo_name)
        if model_repo_size is None:
            failure_notes = "Error checking model repo size. Make sure the model repository exists and is accessible."
            logger.error(failure_notes)
            update_supabase_leaderboard_status(request.hash, "FAILED", failure_notes)
            return get_json_result(request.hash)
    
    except Exception as e:
        failure_notes = f"Error checking model repo size: {e}"
        logger.error(failure_notes)
        update_supabase_leaderboard_status(request.hash, "FAILED", failure_notes)
        return get_json_result(request.hash)


    if model_repo_size > MAX_REPO_SIZE or model_repo_size < MIN_REPO_SIZE:
        failure_notes = f"Model repo size is not up to requirments: {model_repo_size} bytes. Should be less than {MAX_REPO_SIZE} bytes and greater than {MIN_REPO_SIZE} bytes"
        logger.error(failure_notes)
        update_supabase_leaderboard_status(request.hash, "FAILED", failure_notes)
        return get_json_result(request.hash)

    # check model size by checking safetensors index
    model_size = get_model_size(request.repo_namespace, request.repo_name)
    if model_size is None:
        failure_notes = "Error getting model size. Make sure the model.index.safetensors.json file exists in the model repository. And it has the metadata->total_size field."
        logger.error(failure_notes)
        update_supabase_leaderboard_status(request.hash, "FAILED", failure_notes)
        return get_json_result(request.hash)

    if (model_size // 4) > MAX_MODEL_SIZE:
        failure_notes = f"Model size is too large: {model_size} bytes. Should be less than {MAX_MODEL_SIZE} bytes"
        logger.error(failure_notes)
        update_supabase_leaderboard_status(request.hash, "FAILED", failure_notes)
        return get_json_result(request.hash)

    # Add the evaluation task to the queue
    # evaluation_queue.put(request)

    logger.info('returning result')
    return get_json_result(request.hash)
    
def update_row_supabase(row):
    if 'timestamp' in row:
        row['timestamp'] = row['timestamp'].isoformat()
    
    app.state.supabase_client.table("leaderboard").upsert(row).execute()

def upload_to_hf(local_model_dir, model_name):
    try:
        api = huggingface_hub.HfApi(token=os.getenv("HF_ACCESS_TOKEN"))
        api.create_repo(
            repo_id="DippyAI" + "/" + model_name,
            exist_ok=False,
            private=False,
        )
        commit_info = api.upload_folder(
            repo_id="DippyAI" + "/" + model_name,
            folder_path=local_model_dir,
            commit_message="Upload model.",
            repo_type="model",
        )
    except Exception as e:
        # logger.error(f"Error uploading model to Hugging Face: {e}")
        logger.error(f"Error uploading model to Hugging Face: {e}")
        return None


@app.get("/leaderboard")
def display_leaderboard():
    try:
        response = app.state.supabase_client.table("leaderboard").select("*").execute()
        leaderboard = pd.DataFrame(response.data)
        # sort in descending order by total score
        leaderboard = leaderboard.sort_values(by='total_score', ascending=False)
        # # filter out entries older than two weeks
        # two_weeks_ago = datetime.now() - timedelta(weeks=2)
        # # Convert the 'timestamp' column to datetime format. If parsing errors occur, 'coerce' will replace problematic inputs with NaT (Not a Time)
        # leaderboard['timestamp'] = pd.to_datetime(leaderboard['timestamp'], errors='coerce', utc=True)
        # leaderboard = leaderboard[(leaderboard['timestamp'].dt.tz_convert(None) > two_weeks_ago) | (leaderboard.index < 1000)]
        # leaderboard = leaderboard.sort_values(by='total_score', ascending=False)

        return leaderboard.to_dict(orient='records')
    except Exception as e:
        logger.error(f"Error fetching leaderboard from Supabase: {e}")
        return {"status": "failed"}

@app.get("/load_leaderboard_from_supabase")
def display_leaderboard():
    if not app.state.leaderboard_update_time or time.time() - app.state.leaderboard_update_time > 10 * 60:
        app.state.leaderboard_update_time = time.time()
        try:
            response = app.state.supabase_client.table("leaderboard").select("*").execute()
            app.state.leaderboard = pd.DataFrame(response.data)
        except Exception as e:
            logger.error(f"Error fetching leaderboard from Supabase: {e}")
            return {"status": "failed"}
    return app.state.leaderboard.to_dict(orient='records')


if __name__ == "__main__":
    # add command line arguments for the ports of the two apis
    import argparse
    parser = argparse.ArgumentParser(description="Run the server")
    parser.add_argument("--main-api-port", type=int, default=8000, help="Port for the main API")
    parser.add_argument("--eval-score-port", type=int, default=8001, help="Port for the eval_score API")
    parser.add_argument("--vibe-score-port", type=int, default=8002, help="Port for the vibe_score API")
    parser.add_argument("--save-remote", action="store_true", default=False, help="Enable remote saving")
    args = parser.parse_args()

    MAIN_API_PORT = args.main_api_port
    EVAL_SCORE_PORT = args.eval_score_port
    VIBE_SCORE_PORT = args.vibe_score_port
    
    evaluation_queue = multiprocessing.Queue()


    supabase_url = os.environ["SUPABASE_URL"]
    supabase_key = os.environ["SUPABASE_KEY"]
    try:
        app.state.supabase_client = create_client(supabase_url, supabase_key)
    except Exception as e:
        logger.warning(f"Failed to create Supabase client: {e}. Leaderboard will only be saved locally.")
        supabase_client = None

    
    # Create a global shared namespace for the leaderboard
    manager_instance = multiprocessing.Manager()
    app.state.ns = manager_instance.Namespace()

    try:
        logger.info("Starting evaluation thread")
        evaluation_process = multiprocessing.Process(target=model_evaluation_worker)
        evaluation_process.start()
        logger.info("Starting API server")
        uvicorn.run(app, host="0.0.0.0", port=MAIN_API_PORT)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, stopping...")
    except Exception as e:
        logger.error(f"An exception occurred: {e}")
    finally:
        logger.info("Stopping evaluation thread")
        evaluation_process.join()
