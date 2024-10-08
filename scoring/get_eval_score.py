import gc
import os
from datetime import datetime, timezone, timedelta
from typing import Any
import torch
import math
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
from accelerate.utils import release_memory
from datetime import datetime, timezone, timedelta

from scoring.eval_score import eval_score, cleanup, warmup_model, eval_score_batch
from scoring.dataset import StreamedSyntheticDataset

# Import necessary modules and functions from the main API file
from scoring.common import (
    MAX_AVG_LATENCY,
    MAX_GENERATION_LENGTH,
    MAX_MODEL_SIZE,
    VOCAB_TRUNCATION,
    MAX_SEQ_LEN,
    EVALUATION_DATASET_SAMPLE_SIZE,
    EvaluateModelRequest,
    chat_template_mappings,
)

max_entropy = math.log(VOCAB_TRUNCATION)


def get_eval_score(request: EvaluateModelRequest):
    print(f"dumping env {os.environ}")
    print("debug_cuda_devices")
    for i in range(torch.cuda.device_count()):
        print(torch.cuda.get_device_properties(i).name)

    # Now download the weights
    print("Downloading model weights")
    model_downloaded = False
    failure_reason = ""
    # make dir data/hash if not exist
    cache_path = f"{request.hash}_{request.repo_namespace}_{request.repo_name}"
    if not os.path.exists(f"model_cache_dir/{cache_path}"):
        os.makedirs(f"model_cache_dir/{cache_path}")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,  # This does not hurt performance much according to
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    try:
        model = AutoModelForCausalLM.from_pretrained(
            f"{request.repo_namespace}/{request.repo_name}",
            revision=request.revision,
            quantization_config=quant_config,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            device_map="sequential",
            cache_dir=f"model_cache_dir/{cache_path}",
        )
        print(f"loaded model as {type(model)}")

    except Exception as e:
        try:
            print(
                f"Error loading model in 4 bit quant with flash attention.: {e}. Trying vanilla load. This might cause OOM."
            )
            model = AutoModelForCausalLM.from_pretrained(
                f"{request.repo_namespace}/{request.repo_name}",
                revision=request.revision,
                device_map="auto",
                cache_dir=f"model_cache_dir/{cache_path}",
                # force_download=True
            )
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")

    model.eval()

    print("Downloading tokenizer")
    try:
        input_tokenizer = AutoTokenizer.from_pretrained(
            f"{request.repo_namespace}/{request.repo_name}",
            padding_side="left",
            force_download=True,
        )
        output_tokenizer = AutoTokenizer.from_pretrained(
            f"{request.repo_namespace}/{request.repo_name}",
            padding_side="right",
            force_download=True,
        )
        if input_tokenizer.pad_token is None:
            input_tokenizer.pad_token = input_tokenizer.eos_token  # add a pad token if not present
            input_tokenizer.pad_token_id = input_tokenizer.eos_token_id
            output_tokenizer.pad_token = output_tokenizer.eos_token  # add a pad token if not present
            output_tokenizer.pad_token_id = output_tokenizer.eos_token_id

        print("Tokenizer downloaded successfully")
    except Exception as e:
        failure_reason = str(e)
        cleanup(model, model_downloaded, request)
        raise Exception("Error downloading tokenizer: " + failure_reason)

    # warm up the model
    num_gpus = torch.cuda.device_count()
    print(f"Warming up model with gpus {num_gpus}")
    
    try:
        avg_latency = warmup_model(model)
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
    try:
        dataset = StreamedSyntheticDataset(
        max_input_len=MAX_SEQ_LEN - MAX_GENERATION_LENGTH - 200,
        )
        # set the chat template params
        dataset.set_chat_template_params(chat_template_mappings[request.chat_template_type], input_tokenizer)
        sampled_data = dataset.sample_dataset(EVALUATION_DATASET_SAMPLE_SIZE)
    except Exception as e:
        failure_reason = str(e)
        cleanup(model, model_downloaded, request)
        raise Exception(f"Error loading dataset: {failure_reason}")

    # Part 2: Evaluate the model
    print("Evaluating model")
    try:
        evaluation_results = eval_score_batch(
            model,
            sampled_data,
            input_tokenizer=input_tokenizer,
            output_tokenizer=output_tokenizer,
            request=request,
        )
        evaluation_score = evaluation_results["average_prob"]
        entropy_score = evaluation_results["average_entropy"]
        print("Model evaluation score: ", evaluation_score)
        print("Model entropy (creativity) score: ", entropy_score)
    except Exception as e:
        failure_reason = str(e)
        cleanup(model, model_downloaded, request)
        raise Exception("Error evaluating model: " + failure_reason)

    return {
        "eval_score": evaluation_score,
        "latency_score": latency_score,
        "model_size_score": model_size_score,
        "creativity_score": entropy_score,
    }
