import gc
import os
from datetime import datetime, timezone, timedelta
from typing import Any
import torch
import math
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
from accelerate.utils import release_memory
from datetime import datetime, timezone, timedelta
from peft import PeftModel
from peft import AutoPeftModelForCausalLM


from scoring.eval_score import cleanup, warmup_model, eval_score_batch
from scoring.dataset import StreamedSyntheticDataset

# Import necessary modules and functions from the main API file
from scoring.common import (
    MAX_AVG_LATENCY,
    MAX_GENERATION_LENGTH,
    MAX_MODEL_SIZE,
    MODEL_CACHE_DIR,
    VOCAB_TRUNCATION,
    MAX_SEQ_LEN,
    DEFAULT_LORA_BASE,
    EVALUATION_DATASET_SAMPLE_SIZE,
    EvaluateModelRequest,
    chat_template_mappings,
)

max_entropy = math.log(VOCAB_TRUNCATION)


def get_eval_score(request: EvaluateModelRequest, use_lora: bool = False):
    start_time = datetime.now(timezone.utc)
    repo_id = f"{request.repo_namespace}/{request.repo_name}"
    print(f"CUDA devices: {torch.cuda.device_count()}")
    print(f"Using CUDA device: {torch.cuda.current_device()}")
    print(f"Active GPU: {torch.cuda.get_device_name(0)}")
    print(f"Repo ID: {repo_id}")

    print(f"dumping env {os.environ}")
    for i in range(torch.cuda.device_count()):
        print(torch.cuda.get_device_properties(i).name)

    # Now download the weights
    print("Downloading model weights")
    model_download_start = datetime.now(timezone.utc)
    model_downloaded = False
    failure_reason = ""
    # make dir data/hash if not exist
    cache_path = f"{request.hash}_{request.repo_namespace}_{request.repo_name}"
    if not os.path.exists(f"{MODEL_CACHE_DIR}/{cache_path}"):
        os.makedirs(f"{MODEL_CACHE_DIR}/{cache_path}")

    if use_lora:
        repo_id = DEFAULT_LORA_BASE
        print(f"Loading base model from {repo_id}")

    model_type = "base"
    model_path = repo_id
    if os.environ.get("USE_MODEL_DIR", "0") == "1":
        model_path = "/app/model_dir"
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            cache_dir=MODEL_CACHE_DIR,
            device_map="auto",
        )
        model_download_time = datetime.now(timezone.utc) - model_download_start
        print(f"Time to download model: {model_download_time}")
        num_params = sum(p.numel() for p in base_model.parameters())
        rounded_params = num_params / 1e9
        print(f"Total number of parameters (from HF model): {rounded_params}B params")
        if rounded_params < 20:
            raise Exception(f"Model below 20B minimum : {rounded_params}B params")

        if use_lora:
            lora_adapter = f"{request.repo_namespace}/{request.repo_name}"
            print(f"Loading LoRA adapter from {request.repo_namespace}/{request.repo_name}")
            model = PeftModel.from_pretrained(base_model, lora_adapter, cache_dir=MODEL_CACHE_DIR)
            model_type = "lora"
        else:
            model = base_model
        model.to("cuda")

    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")

    model.eval()
    print(f"loaded model as {type(model)} with type {model_type}")

    try:
        tokenizer_start = datetime.now(timezone.utc)
        input_tokenizer = AutoTokenizer.from_pretrained(
            repo_id,
            padding_side="left",
            force_download=True,
            cache_dir=MODEL_CACHE_DIR,
        )
        output_tokenizer = AutoTokenizer.from_pretrained(
            repo_id,
            padding_side="right",
            force_download=True,
            cache_dir=MODEL_CACHE_DIR,
        )
        if input_tokenizer.pad_token is None:
            input_tokenizer.pad_token = input_tokenizer.eos_token  # add a pad token if not present
            input_tokenizer.pad_token_id = input_tokenizer.eos_token_id
            output_tokenizer.pad_token = output_tokenizer.eos_token  # add a pad token if not present
            output_tokenizer.pad_token_id = output_tokenizer.eos_token_id

        tokenizer_time = datetime.now(timezone.utc) - tokenizer_start
        print(f"Time to setup tokenizers: {tokenizer_time}")
    except Exception as e:
        failure_reason = str(e)
        cleanup(model, model_downloaded, request)
        raise Exception("Error downloading tokenizer: " + failure_reason)

    # warm up the model
    warmup_start = datetime.now(timezone.utc)
    num_gpus = torch.cuda.device_count()
    model.to("cuda")
    print(f"Warming up model with gpus {num_gpus}")

    try:
        avg_latency = warmup_model(model)
        warmup_time = datetime.now(timezone.utc) - warmup_start
        print(f"Time to warmup model: {warmup_time}")
        if not avg_latency:  # either 0 or None
            raise Exception("Error warming up model")

    except Exception as e:
        print(e)
        failure_reason = str(e)
        cleanup(model, model_downloaded, request)
        raise Exception("Error warming up model: " + failure_reason)

    # get latency score
    latency_score = 1 - (avg_latency / MAX_AVG_LATENCY)
    print("Latency score: ", latency_score)

    # get model size score
    try:
        print("Model weights downloaded successfully")
        model_size = model.get_memory_footprint()
        print("Model size: ", model_size, " Bytes")
        print("Model number of parameters: ", model.num_parameters())
        # check if model size is within the limit. If not, return an error
        if model_size > MAX_MODEL_SIZE:
            del model
            torch.cuda.empty_cache()
            raise Exception(
                f"Model is too large when loaded in 4 bit quant: {model_size} Bytes. Should be less than {MAX_MODEL_SIZE} Bytes",
            )

        model_size_score = 1 - (model_size / MAX_MODEL_SIZE)
        print("Model size score: ", model_size_score)
        model_downloaded = True
    except Exception as e:
        failure_reason = str(e)
        cleanup(None, model_downloaded, request)
        raise Exception("Error loading model: " + failure_reason)

    print("Sampling dataset")
    dataset_start = datetime.now(timezone.utc)
    eval_period = "None"
    try:
        dataset = StreamedSyntheticDataset(
            max_input_len=MAX_SEQ_LEN - MAX_GENERATION_LENGTH - 200,
        )
        eval_period = dataset.eval_period
        # set the chat template params
        dataset.set_chat_template_params(chat_template_mappings[request.chat_template_type], input_tokenizer)
        sampled_data = dataset.sample_dataset(EVALUATION_DATASET_SAMPLE_SIZE)
        dataset_time = datetime.now(timezone.utc) - dataset_start
        print(f"Time to sample dataset: {dataset_time}")
    except Exception as e:
        failure_reason = str(e)
        cleanup(model, model_downloaded, request)
        raise Exception(f"Error loading dataset: {failure_reason}")

    # Part 2: Evaluate the model
    eval_start = datetime.now(timezone.utc)
    print(f"Evaluating model with len(sampled_data) {len(sampled_data)} and eval_period {eval_period}")
    try:
        evaluation_results = eval_score_batch(
            model,
            sampled_data,
            input_tokenizer=input_tokenizer,
            output_tokenizer=output_tokenizer,
            request=request,
        )
        eval_time = datetime.now(timezone.utc) - eval_start
        print(f"Time to evaluate model: {eval_time}")
        evaluation_score = evaluation_results["average_prob"]
        entropy_score = evaluation_results["average_entropy"]
        print("Model evaluation score: ", evaluation_score)
        print("Model entropy (creativity) score: ", entropy_score)
    except Exception as e:
        failure_reason = str(e)
        cleanup(model, model_downloaded, request)
        raise Exception("Error evaluating model: " + failure_reason)

    total_time = datetime.now(timezone.utc) - start_time
    print(f"Total execution time: {total_time}")

    return {
        "eval_score": evaluation_score,
        "latency_score": latency_score,
        "model_size_score": model_size_score,
        "creativity_score": entropy_score,
        "eval_period": eval_period,
    }
