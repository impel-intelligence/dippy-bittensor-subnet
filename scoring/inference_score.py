from vllm import LLM
import gc
import ray
from vllm.distributed.parallel_state import destroy_model_parallel
import torch

from scoring.common import (
    EvaluateModelRequest,
    MAX_SEQ_LEN_COHERENCE_SCORE,
    DEFAULT_LORA_BASE,
    MODEL_CACHE_DIR,
)
import os

MAX_NUM_SEQS = 16
COHERENCE_THRESHOLD = 0.95



def get_inference_score(request: EvaluateModelRequest, use_lora: bool = False):
    from scoring.coherence_score import get_coherence_score
    from scoring.judge_score import get_judge_score

    # For simplicity, will always look for main branch
    repo_id = f"{request.repo_namespace}/{request.repo_name}"
    print(f"Using CUDA device: {torch.cuda.current_device()}")
    print(f"Active GPU: {torch.cuda.get_device_name(0)}")
    print(f"Repo ID: {repo_id}")

    for i in range(torch.cuda.device_count()):
        print(f"debug_cuda_devices_available : {torch.cuda.get_device_properties(i).name}")

    if use_lora:
        repo_id = DEFAULT_LORA_BASE
        print(f"Loading lora given base model {repo_id}")

    model_path = repo_id
    if os.environ.get("USE_MODEL_DIR", "0") == "1":
        model_path = "/app/model_dir"
    if use_lora:
        model = LLM(
            model=model_path,
            tensor_parallel_size=torch.cuda.device_count(),
            max_num_seqs=MAX_NUM_SEQS,
            max_model_len=MAX_SEQ_LEN_COHERENCE_SCORE,
            download_dir=MODEL_CACHE_DIR,
            enable_lora=use_lora,
            max_lora_rank=256,
        )
    else:
        model = LLM(
            model=model_path,
            tensor_parallel_size=torch.cuda.device_count(),
            max_num_seqs=MAX_NUM_SEQS,
            max_model_len=MAX_SEQ_LEN_COHERENCE_SCORE,
            download_dir=MODEL_CACHE_DIR,
        )

    print(f"loaded model {repo_id} with use_lora {use_lora}")
    coherence_result = {}
    judge_result = {}
    coherence_result = {"coherence_score" : 1}    
    judge_result = get_judge_score(request, model, verbose=False, use_lora=use_lora)
    judge_result = {"judge_score": judge_result.get("judge_score", {}).get("win_rate", 0)}

    inference_result = coherence_result | judge_result

    print("=============inference_result===================")
    print(inference_result)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    destroy_model_parallel()

    del model.llm_engine.model_executor.driver_worker
    del model.llm_engine.model_executor

    del model
    gc.collect()
    torch.cuda.empty_cache()
    ray.shutdown()
    try:
        torch.distributed.destroy_process_group()
    except:
        print("No process group to destroy")
    print("cleaned up all cuda resources")

    return inference_result
