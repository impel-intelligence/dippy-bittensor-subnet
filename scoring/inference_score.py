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


def get_inference_score(request: EvaluateModelRequest, use_lora:bool = False):
    from scoring.vibe_score import get_vibe_match_score
    from scoring.coherence_score import get_coherence_score
    from scoring.judge_score import get_judge_score

    # For simplicity, will always look for main branch
    repo_id = f"{request.repo_namespace}/{request.repo_name}"
    print(f"Using CUDA device: {torch.cuda.current_device()}")
    print(f"Active GPU: {torch.cuda.get_device_name(0)}")
    print(f"Repo ID: {repo_id}")

    # Option 2: Using torch.cuda
    torch.cuda.set_device(0)  # Only use first GPU

    print(f"Using CUDA device: {torch.cuda.current_device()}")
    print(f"Active GPU: {torch.cuda.get_device_name(0)}")

    for i in range(torch.cuda.device_count()):
        print(f"debug_cuda_devices_available : {torch.cuda.get_device_properties(i).name}")

    if use_lora:
        repo_id = DEFAULT_LORA_BASE
        print(f"Loading lora given base {repo_id}")
    model = LLM(
        model=repo_id,
        tensor_parallel_size=torch.cuda.device_count(),
        max_num_seqs=MAX_NUM_SEQS,
        max_model_len=MAX_SEQ_LEN_COHERENCE_SCORE,
        download_dir=MODEL_CACHE_DIR,
        enable_lora=use_lora,
        max_lora_rank=256,
        )


    coherence_result = get_coherence_score(request, model, True)

    vibe_result = {}
    coherence_result = {}
    judge_result = get_judge_score(request, model, verbose=True, use_lora=use_lora)

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
