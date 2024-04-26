import gc
from typing import Any, Optional
import requests
import subprocess
import os
from queue import Queue
from threading import Thread
from tqdm import tqdm
import shutil
from pydantic import BaseModel

import pandas as pd
import uvicorn
import torch
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer, BitsAndBytesConfig
import huggingface_hub
from fastapi import FastAPI, HTTPException

from dataset import PippaDataset
from supabase import create_client
import hashlib
import argparse

from dotenv import load_dotenv

load_dotenv()

# TODO: Add a timeout to the evaluation API to prevent it from running for too long

# Constants
QUALITATIVE_SCORE_WEIGHT = 0.8
MODEL_SIZE_SCORE_WEIGHT = 0.1
LATENCY_SCORE_WEIGHT = 0.1
MAX_AVG_LATENCY = 10000 # in milliseconds

MAX_MODEL_SIZE = 18 * 1024 * 1024 * 1024 # in bytes
MAX_REPO_SIZE = 80 * 1024 * 1024 * 1024 #  in bytes
SAMPLE_SIZE = 100 # number of samples to evaluate the model from the dataset
BATCH_SIZE = 2 # batch size for evaluation
VOCAB_TRUNCATION = 1000 # truncate the vocab to top 50 tokens
PROB_TOP_K = 10 # the correct token should be in the top 50 tokens, else a score of 0 is given to that token
# TODO: this will truncate the sequence to MAX_SEQ_LEN tokens. This is a temporary fix to make the evaluation faster.
MAX_SEQ_LEN = 4096 # maximum sequence length that should be allowed because eval gets really slow with longer sequences than this
SAVE_REMOTE = True # Save the leaderboard to Supabase 

leaderboard_file = 'leaderboard.csv'

# if the leaderboard file does not exist, create it with proper columns
columns = ['hash', 'repo_namespace', 'repo_name', 'chat_template_type', 'model_size_score', 'qualitative_score', 'latency_score', 'total_score', 'timestamp', 'status', 'notes']
if not os.path.exists(leaderboard_file):
    pd.DataFrame(columns=columns).to_csv(leaderboard_file, index=False)

device = "cuda" if torch.cuda.is_available() else "cpu"

class EvaluateModelRequest(BaseModel):
    repo_namespace: str
    repo_name: str
    chat_template_type: str
    hash: str
    revision: Optional[str] = "main"
    competition_id: Optional[str] = "d1"

evaluation_queue = Queue()

app = FastAPI()

# create dir data if not exists
if not os.path.exists("data"):
    os.makedirs("data")
# download the file pippa_deduped.jsonl from huggingface
if not os.path.exists("data/pippa_deduped.jsonl"):
    huggingface_hub.hf_hub_download(repo_id="PygmalionAI/PIPPA", filename="pippa_deduped.jsonl", repo_type="dataset", local_dir = "data")

dataset = PippaDataset("data/pippa_deduped.jsonl")

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
        'total_score': 'float64',  # Use 'float64' to allow NaNs
        'timestamp': str,
        'status': str,
        'notes': str
    }
    leaderboard = pd.read_csv(leaderboard_file, dtype=dtype_dict, parse_dates=['timestamp'])
    # Replace NaN with None for JSON serialization
    leaderboard = leaderboard.where(pd.notnull(leaderboard), None)
    return leaderboard

def save_leaderboard(leaderboard: pd.DataFrame, hash=None, save_remote = True):
    leaderboard.to_csv(leaderboard_file, index=False)
    if hash is not None:
        leaderboard_row = leaderboard[leaderboard['hash'] == hash].iloc[0]
        if save_remote:
            supabase_url = os.getenv("SUPABASE_URL")
            supabase_key = os.getenv("SUPABASE_KEY")
            supabase_client = create_client(supabase_url, supabase_key)
            try:
                supabase_client.table("leaderboard").upsert({"hash": leaderboard_row['hash'], "repo_namespace": leaderboard_row['repo_namespace'], "repo_name": leaderboard_row['repo_name'], "chat_template_type": leaderboard_row['chat_template_type'], "model_size_score": leaderboard_row['model_size_score'], "qualitative_score": leaderboard_row['qualitative_score'], "latency_score": leaderboard_row['latency_score'], "total_score": leaderboard_row['total_score'], "status": leaderboard_row['status'], "notes": leaderboard_row['notes']}).execute()
            except Exception as e:
                print(f"Error saving leaderboard row to Supabase: {e}")

def model_evaluation_worker():
    while True:
        # Get the next evaluation task from the queue
        request = evaluation_queue.get()
        if request is None:
            # Stop the thread if the sentinel is received
            break
        try:
            # Process the evaluation task
            result = evaluate_model_logic(request)
            print(f"Model evaluation completed: {result}")
        except Exception as e:
            print(f"Error during model evaluation: {e}")
        finally:
            # Mark the task as done
            evaluation_queue.task_done()

evaluation_thread = Thread(target=model_evaluation_worker)

def load_model_no_download(repo_namespace: str, repo_name: str):
    """
    Validate the model by loading it, without downloading it from the Hugging Face Hub
    """
    try:
        config = AutoConfig.from_pretrained('/'.join([repo_namespace, repo_name]), revision='main')
    except Exception as e:
        return None, str(e)
    
    if config is not None:
        print("Model configuration retrieved from Hub")
        try:
            # Check if GPU is available and see if model fits in GPU
            print('loading model in RAM to check if it fits in GPU')
            model = AutoModelForCausalLM.from_config(
                config=config,
            )
            print("Model loaded successfully")
            return model, None
        except Exception as e:
            return None, str(e)
    else:
        return None, "Could not retrieve model configuration from Hub"

def parse_size(line):
    """
    Parse the size string with unit and convert it to bytes.
    
    Args:
    - size_with_unit (str): The size string with unit (e.g., '125 MB')
    
    Returns:
    - int: The size in bytes
    """
    try:
        # get number enclosed in brackets
        size, unit = line[line.find("(")+1:line.rfind(")")].strip().split(' ')
        size = float(size.replace(',', ''))  # Remove commas for thousands
        unit = unit.lower()
        if unit == 'kb':
            return int(size * 1024)
        elif unit == 'mb':
            return int(size * 1024 * 1024)
        elif unit == 'gb':
            return int(size * 1024 * 1024 * 1024)
        else:
            raise ValueError(f"Unknown unit: {unit}")
    except ValueError as e:
        print(f"Error parsing size string '{size}{unit}': {e}")
        return 0

def check_model_repo_size(hash: int, repo_namespace: str, repo_name: str) -> int:
    """
    Check the size of a model hosted on Hugging Face using Git LFS without checking out the files,
    and clean up the cloned repository afterwards, even if an error occurs.
    
    Args:
    - hash (int): The hash of the model
    - repo_namespace (str): The namespace of the model repository
    - repo_name (str): The name of the model repository
    
    Returns:
    - int: The total size of the model files in bytes
    """
    repo_dir = f"data/{str(hash)}/models--{repo_namespace}--{repo_name}"
    original_dir = os.getcwd()
    try:
        subprocess.run(["git", "clone", "--no-checkout", f"https://huggingface.co/{repo_namespace}/{repo_name}", repo_dir], check=True, timeout=10)
        os.chdir(repo_dir)
        lfs_files_output = subprocess.check_output(["git", "lfs", "ls-files", "-s"], text=True, timeout=10)
        total_size = sum(parse_size(line) for line in lfs_files_output.strip().split('\n') if line)
        return total_size
    except subprocess.TimeoutExpired as e:
        print(f"Operation timed out: {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    finally:
        os.chdir(original_dir)
        shutil.rmtree(os.path.join(original_dir, repo_dir), ignore_errors=True)
    
def get_eval_score(
        model: Any, 
        sampled_data: list[tuple], 
        input_tokenizer: AutoTokenizer, 
        output_tokenizer: AutoTokenizer,
        debug: bool = False
    ):
    """
    Evaluate the model on a dummy task
    """
    # maximum length this model can handle.
    max_len = min(model.config.max_position_embeddings, MAX_SEQ_LEN)
    
    if max_len is None:
        raise ValueError("Model does not have a maximum position embedding set")
    
    # unzip the sampled data
    contexts, target_texts = zip(*sampled_data)

    total_prob = 0
    token_count = 0

    # now we want to calculate the average probability of the target tokens that model assigns.
    batch_size = BATCH_SIZE
    model.eval()

    for i in tqdm(range(0, len(contexts), batch_size), desc="Evaluating batches"):
        with torch.no_grad():
            # Tokenize the inputs and labels

            # Pad the inputs and expected outputs to the same length in such a way that the 
            # padding is on the left for inputs and on the right for outputs
            # this will ensure that the model see an intact context and the expected output is not shifted
            # example: [pad, pad, context, context, target, target, pad, pad]
            
            targets = output_tokenizer(
                target_texts[i:i+batch_size], 
                return_tensors='pt', 
                padding=True,
                add_special_tokens=False # we don't want to add special tokens to the target as it continues from the context and already contains eos token.
            ) # this will put padding to the right

            # get the max length of the input by subtracting the length of the targets from the max length
            max_input_len = max_len - targets['input_ids'].shape[1]

            inputs = input_tokenizer(
                contexts[i:i+batch_size], 
                return_tensors='pt', 
                padding=True,
                truncation=True, 
                max_length=max_input_len,
                add_special_tokens=True,
            ) # this will put padding to the left and truncate the input if it is too long

            # concatenate the inputs and targets and their attention masks
            input_ids = torch.cat([inputs['input_ids'], targets['input_ids']], dim=1).to(device)
            attention_mask = torch.cat([inputs['attention_mask'], targets['attention_mask']], dim=1).to(device)
            
            # get the mask that only give us the output ids
            targets_ids_mask = torch.cat(
                [
                    torch.zeros_like(inputs['attention_mask']), 
                    targets['attention_mask']
                ], dim=1
            )

            # shift the output mask to the right by one to get the corresponding predicted logits
            targets_ids_mask = torch.cat(
                [
                    torch.zeros_like(targets_ids_mask[:, :1]), 
                    targets_ids_mask[:, :-1]
                ], dim=1
            ).to(device)

            # Get model predictions (logits)
            try:
                outputs = model(
                    input_ids, 
                    attention_mask=attention_mask,
                    use_cache=False,
                )
            except Exception as e:
                print("Error getting model predictions for sequence length: ", input_ids.shape[1], " batch size: ", input_ids.shape[0])
                raise ValueError("Error getting model predictions: " + str(e))

            # shift the logits to the right by one to get the corresponding predicted logits
            outputs.logits = torch.cat(
                [
                    torch.zeros_like(outputs.logits[:, :1, :]), 
                    outputs.logits[:, :-1, :]
                ], dim=1
            )

            if torch.isnan(outputs.logits).any():
                raise ValueError("NaN values detected llm -> outputs.logits tensor")

            # Only keep the top PROB_TOP_K scores by -inf the rest
            # This will make the model only consider the top 100 tokens and make sure the models with higher vocab sizes are not penalized

            # get the top k logits and mask out the rest
            top_k_logits, top_k_indices = outputs.logits.topk(VOCAB_TRUNCATION, dim=-1)
            outputs.logits = torch.full_like(outputs.logits, float('-inf')).scatter(-1, top_k_indices, top_k_logits)

            if debug:
                # print the input tokens and top 10 predicted tokens
                print(f"Input: {input_tokenizer.decode(input_ids[0])}")
                for i in range(len(input_ids[0])):
                    if targets_ids_mask[0][i].item() == 1:
                        actual_id = input_ids[0][i].item()
                        actual_token = output_tokenizer.decode([actual_id])
                        top_10_predicted_ids = outputs.logits[0][i].topk(10).indices.tolist()
                        top_10_predicted_tokens = [output_tokenizer.decode([id]) for id in top_10_predicted_ids]
                        print(f"Actual token: {actual_token}", f" -> top 10 pred tokens: {top_10_predicted_tokens}")

            
            # normalize the logits to get probabilities
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1).cuda()

            if torch.isnan(probabilities).any():
                raise ValueError("NaN values detected in the probabilities tensor")
            
            # Get the top PROB_TOP_K indices and zero out all other probabilities
            top_prob_indices = torch.topk(probabilities, PROB_TOP_K, dim=-1).indices
            mask = torch.zeros_like(probabilities, dtype=torch.bool).scatter_(-1, top_prob_indices, True)
            probabilities[~mask] = 1e-9
            # Get the probabilities assigned by the model to the target tokens
            token_probabilities = probabilities.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)

            # Mask out non target tokens
            token_probabilities = token_probabilities * targets_ids_mask

            # Sum the probabilities for each batch and accumulate
            total_prob += token_probabilities.sum().cpu().item()
            token_count += targets_ids_mask.sum().cpu().item()

            del input_ids, attention_mask, targets_ids_mask, outputs, probabilities, top_k_logits, top_k_indices, mask, token_probabilities
            gc.collect()
            torch.cuda.empty_cache()
    
    # Calculate the average probability
    average_prob = total_prob / token_count if token_count > 0 else 0

    return average_prob

def cleanup(model, model_downloaded, request):
    """
    Clean up the model data from memory and disk
    """
    # delete the model from memory
    del model
    torch.cuda.empty_cache()
    total, used, free = shutil.disk_usage("/")
    if used / total > 0.9:
        print("Warning: SSD is more than 90% full.") 
    if model_downloaded:
        # Check if the SSD is more than 90% full before deleting the model data
        try:
            shutil.rmtree(f"data/{str(request.hash)}")
        except Exception as e:
            print(f"Warning: Error deleting model data: {e}")

def warmup_model(model):
    """
    Warm up the model by running it on a dummy input
    """
    # run the max sequence length input through the model with batch size BATCH_SIZE
    model.eval()
    latencies = []
    max_model_len = min(model.config.max_position_embeddings, MAX_SEQ_LEN)
    with torch.no_grad():
        for _ in range(10):
            input_ids = torch.randint(0, model.config.vocab_size, (BATCH_SIZE, max_model_len)).to(device)
            attention_mask = torch.ones_like(input_ids).to(device)
            
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)

            start_time.record()
            outputs = model(input_ids, attention_mask=attention_mask)
            end_time.record()
            # Waits for everything to finish running
            torch.cuda.synchronize()

            latency = start_time.elapsed_time(end_time)  # Measure latency in milliseconds
            if torch.isnan(outputs.logits).any():
                raise ValueError("NaN values detected in the logits tensor")

            latencies.append(latency)
            del input_ids, attention_mask, outputs
            gc.collect()
            torch.cuda.empty_cache()

        average_latency = sum(latencies) / len(latencies)
        print(f"Average model inference latency over 10 runs: {average_latency} ms")
        
    return average_latency
        
def evaluate_model_logic(request: EvaluateModelRequest):
    """
    Evaluate a model based on the model size and the quality of the model.
    """
    # ensure that the model is on the leaderboard with status pending
    leaderboard = get_leaderboard()
    if not (leaderboard['hash'] == request.hash).any():
        print(leaderboard)
        print(leaderboard['hash'])
        print(type(leaderboard['hash']))
        print(request.hash)
        print(type(request.hash))
        raise ValueError(f"Model {request.hash} not found in the leaderboard")
    
    # changed status to in progress
    update_leaderboard_status(request.hash, "RUNNING", "Model evaluation in progress")

    # Now download the weights
    print('Downloading model weights')
    model_downloaded = False
    failure_reason = ""
    try:
        # make dir data/hash if not exist
        if not os.path.exists(f"data/{str(request.hash)}"):
            os.makedirs(f"data/{str(request.hash)}")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True, # This does not hurt performance much according to 
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        try:
            model = AutoModelForCausalLM.from_pretrained(
                f"{request.repo_namespace}/{request.repo_name}",
                revision=request.revision,
                device_map='auto',
                quantization_config=quant_config,
                attn_implementation="flash_attention_2",       
                cache_dir = f"data/{str(request.hash)}",
                force_download=True
            )
        except Exception as e:
            print(f"Error loading model in 4 bit quant with flash attention.: {e}. Trying vanilla load.")
            model = AutoModelForCausalLM.from_pretrained(
                f"{request.repo_namespace}/{request.repo_name}",
                revision=request.revision,
                device_map='auto',
                attn_implementation="sdpa",       
                cache_dir = f"data/{str(request.hash)}",
                force_download=True
            )
        print('Model weights downloaded successfully')
        model_size = model.get_memory_footprint()
        print('Model size: ', model_size, ' Bytes')
        print("Model number of parameters: ", model.num_parameters())
        # check if model size is within the limit. If not, return an error
        if model_size > MAX_MODEL_SIZE:
            raise ValueError(f"Model is too large when loaded in 4 bit quant: {model_size} Bytes. Should be less than {MAX_MODEL_SIZE} Bytes")
        model_size_score = 1 - (model_size / MAX_MODEL_SIZE)
        print('Model size score: ', model_size_score)
        model_downloaded = True
    except Exception as e:
        failure_reason = str(e)
        update_leaderboard_status(request.hash, "FAILED", failure_reason)
        cleanup(None, model_downloaded, request)
        raise RuntimeError("Error loading model: " + failure_reason)

    # get the tokenizers
    print('Downloading tokenizer')
    try:
        input_tokenizer = AutoTokenizer.from_pretrained(
            f"{request.repo_namespace}/{request.repo_name}",
            padding_side='left',
        )
        output_tokenizer = AutoTokenizer.from_pretrained(
            f"{request.repo_namespace}/{request.repo_name}",
            padding_side='right',
        )
        if input_tokenizer.pad_token is None:
            input_tokenizer.pad_token = input_tokenizer.eos_token # add a pad token if not present
            input_tokenizer.pad_token_id = input_tokenizer.eos_token_id
            output_tokenizer.pad_token = output_tokenizer.eos_token # add a pad token if not present
            output_tokenizer.pad_token_id = output_tokenizer.eos_token_id
        
        dataset.set_chat_template_params(chat_template_mappings[request.chat_template_type], input_tokenizer)
        print('Tokenizer downloaded successfully')
    except Exception as e:
        failure_reason = str(e)
        cleanup(model, model_downloaded, request)
        update_leaderboard_status(request.hash, "FAILED", failure_reason)
        raise RuntimeError("Error downloading tokenizer: " + failure_reason)

    # warm up the model
    print('Warming up model')
    try:
        avg_latency = warmup_model(model)
        if not avg_latency: # either 0 or None
            raise ValueError("Error warming up model")
            
    except Exception as e:
        failure_reason = str(e)
        update_leaderboard_status(request.hash, "FAILED", failure_reason)
        cleanup(model, model_downloaded, request)
        raise RuntimeError("Error warming up model: " + failure_reason)
    
    # get latency score
    latency_score = 1 - (avg_latency / MAX_AVG_LATENCY)

    print('Sampling dataset')
    try:
        sampled_data = dataset.sample_dataset(SAMPLE_SIZE)
    except Exception as e:
        failure_reason = str(e)
        update_leaderboard_status(request.hash, "FAILED", failure_reason)
        cleanup(model, model_downloaded, request)
        raise RuntimeError("Error sampling dataset: " + failure_reason)
    
    # Part 2: Evaluate the model
    print('Evaluating model')
    try:
        eval_score = get_eval_score(
            model, 
            sampled_data, 
            input_tokenizer=input_tokenizer,
            output_tokenizer=output_tokenizer
        )
        print('Model evaluation score: ', eval_score)
    except Exception as e:
        failure_reason = str(e)
        update_leaderboard_status(request.hash, "FAILED", failure_reason)
        cleanup(model, model_downloaded, request)
        raise RuntimeError("Error evaluating model: " + failure_reason)

    # calculate the total score
    if model_size_score == -1.0 or eval_score == -1.0 or latency_score == -1.0:
        total_score = 0
    else:
        total_score = model_size_score * MODEL_SIZE_SCORE_WEIGHT + eval_score * QUALITATIVE_SCORE_WEIGHT + latency_score * LATENCY_SCORE_WEIGHT

    # update the model on the leaderboard
    try:
        leaderboard = get_leaderboard()
        leaderboard.loc[leaderboard['hash'] == request.hash, 'model_size_score'] = float(model_size_score)
        leaderboard.loc[leaderboard['hash'] == request.hash, 'qualitative_score'] = float(eval_score)
        leaderboard.loc[leaderboard['hash'] == request.hash, 'latency_score'] = float(latency_score)
        leaderboard.loc[leaderboard['hash'] == request.hash, 'total_score'] = float(total_score)
        leaderboard.loc[leaderboard['hash'] == request.hash, 'status'] = "COMPLETED"
        leaderboard.loc[leaderboard['hash'] == request.hash, 'notes'] = ""
        save_leaderboard(leaderboard, request.hash, SAVE_REMOTE)
    except Exception as e:
        failure_reason = str(e)
        update_leaderboard_status(request.hash, "FAILED", failure_reason)
        cleanup(model, model_downloaded, request)
        raise RuntimeError("Error updating leaderboard: " + failure_reason)
    
    # cleanup the model
    cleanup(model, model_downloaded, request)

    return {
        "model_size_score": model_size_score,
        "qualitative_score": eval_score,
        "latency_score": latency_score,
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
        print(f"Error updating leaderboard status for {hash}: {e}")

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
                "total_score": model_entry['total_score']
            },
            "status": model_entry['status']
        }
    else:
        None

def get_model_size(repo_namespace: str, repo_name: str):
    safetensor_index = f"https://huggingface.co/{repo_namespace}/{repo_name}/resolve/main/model.safetensors.index.json"
    response = requests.get(safetensor_index)
    if response.status_code != 200:
        print(f"Error getting safetensors index: {response.text}")
        return None
    
    response_json = response.json()
    if 'metadata' not in response_json:
        print("Error: metadata not found in safetensors index")
        return None
    
    if 'total_size' not in response_json['metadata']:
        print("Error: total_size not found in safetensors index metadata")
        return None
    
    total_size = response_json['metadata']['total_size']
    
    return total_size

def regenerate_hash(namespace, name, chat_template, competition_id):
    s = " ".join([namespace, name, chat_template, competition_id])
    hash_output = hashlib.sha256(s.encode('utf-8')).hexdigest()
    return int(hash_output[:16], 16)  # Returns a 64-bit integer from the first 16 hexadecimal characters

@app.post("/evaluate_model")
def evaluate_model(request: EvaluateModelRequest):
    # verfify hash
    if int(request.hash) != regenerate_hash(request.repo_namespace, request.repo_name, request.chat_template_type, request.competition_id):
        print(f"Hash does not match the expected hash: {request.hash} != {regenerate_hash(request.repo_namespace, request.repo_name, request.chat_template_type, request.competition_id)}")
        raise HTTPException(status_code=400, detail="Hash does not match the expected hash")
    # read the leaderboard file
    # check if the model already exists in the leaderboard
    current_status = get_json_result(request.hash)
    if current_status is not None:
        return current_status

    # validate the request
    if request.chat_template_type not in chat_template_mappings:
        print(f"Chat template type not supported: {request.chat_template_type}")
        raise HTTPException(status_code=400, detail="Chat template type not supported")
    
    # check repo size of the model to see if it is within the limit
    try:
        model_repo_size = check_model_repo_size(request.hash, request.repo_namespace, request.repo_name)
        if model_repo_size is None:
            print("Error checking model repo size")
            raise HTTPException(status_code=400, detail="Error occured while checking model repo size on Hugging Face Hub.")
    except Exception as e:
        print(f"Error checking model repo size: {e}")
        raise HTTPException(status_code=400, detail=f'"{request.repo_namespace}/{request.repo_name}" is probably a gated model, or it does not exist on the Hugging Face Hub.')
    
    if model_repo_size > MAX_REPO_SIZE:
        print(f"Model repo size is too large: {model_repo_size} bytes. Should be less than {MAX_REPO_SIZE} bytes")
        raise HTTPException(status_code=400, detail="Model repo size is too large: " + str(model_repo_size) + " bytes. Should be less than " + str(MAX_REPO_SIZE) + " bytes")
    
    # check model size by checking safetensors index
    model_size = get_model_size(request.repo_namespace, request.repo_name)
    if model_size is None:
        model_size = 0
        # raise HTTPException(status_code=400, detail="Error getting model size. Make sure the model.index.safetensors.json file exists in the model repository. And it has the metadata->total_size field.")

    if model_size > MAX_MODEL_SIZE:
        print(f"Model size is too large: {model_size} bytes. Should be less than {MAX_MODEL_SIZE} bytes")
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
        "total_score": -1.0,
        "timestamp": pd.Timestamp.utcnow(),
        "status": "QUEUED",
        "notes": ""
    }])
    leaderboard = pd.concat([leaderboard, new_entry], ignore_index=True)
    save_leaderboard(leaderboard, request.hash, SAVE_REMOTE)
    
    # Add the evaluation task to the queue
    evaluation_queue.put(request)

    print('returning result')
    return get_json_result(request.hash)

@app.get("/leaderboard")
def display_leaderboard():
    return get_leaderboard().to_dict(orient='records')

if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Run the server")
    parser.add_argument("--no-remote", action="store_false", help="Disable remote saving")
    args = parser.parse_args()

    SAVE_REMOTE = args.no_remote

    try:
        print("Starting evaluation thread")
        evaluation_thread.start()
        print("Starting API server")
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except KeyboardInterrupt:
        print("Keyboard interrupt received, stopping...")
    except Exception as e:
        print(f"An exception occurred: {e}")
    finally:
        print("Stopping evaluation thread")
        # empty the queue
        while not evaluation_queue.empty():
            evaluation_queue.get()
            evaluation_queue.task_done()
        
        # remove any rows with status QUEUED
        leaderboard = get_leaderboard()
        leaderboard = leaderboard[leaderboard['status'] != 'QUEUED']
        save_leaderboard(leaderboard, None, SAVE_REMOTE)
        # add a sentinel to the queue to stop the thread
        evaluation_queue.put(None)
        evaluation_thread.join()

        # remove any RUNNING status
        leaderboard = get_leaderboard()
        leaderboard = leaderboard[leaderboard['status'] != 'RUNNING']
        save_leaderboard(leaderboard, None, SAVE_REMOTE)
        print("API server and evaluation thread have been stopped")
