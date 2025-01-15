from vllm import LLM
import gc
import ray
from vllm.distributed.parallel_state import destroy_model_parallel
import torch

from scoring.common import (
    EvaluateModelRequest,
    MAX_SEQ_LEN_COHERENCE_SCORE,
    MAX_GENERATION_LENGTH,
    MODEL_CACHE_DIR,
)
import os

MAX_NUM_SEQS = 16


def get_inference_score(request: EvaluateModelRequest):
    inference_score_result = wrap_inference_score(request)
    return inference_score_result

def wrap_inference_score(request: EvaluateModelRequest):
    from scoring.vibe_score import get_vibe_match_score
    from scoring.coherence_score import get_coherence_score

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
    model = LLM(
        model=repo_id,
        tensor_parallel_size=torch.cuda.device_count(),
        max_num_seqs=MAX_NUM_SEQS,
        max_model_len=MAX_SEQ_LEN_COHERENCE_SCORE,
        download_dir=MODEL_CACHE_DIR,
    )

    vibe_result = get_vibe_match_score(request, model)
    coherence_result = {}
    coherence_result = get_coherence_score(request, model, True)

    inference_result = vibe_result | coherence_result

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
