from vllm import LLM, SamplingParams
import gc
import ray
from vllm.distributed.parallel_state import destroy_model_parallel
import torch

from scoring.common import (
    EvaluateModelRequest,
    MAX_SEQ_LEN_COHERENCE_SCORE,
    MAX_GENERATION_LENGTH,
    chat_template_mappings,
    SAMPLE_SIZE_VIBE_SCORE,
    COHERENCE_BATCH_SIZE,
    COHERENCE_MAX_TOKENS,
    COHERENCE_EVAL_MODEL,
    COHERENCE_NUM_EVALS,
    MODEL_CACHE_DIR,
)
import os

MAX_NUM_SEQS = 16


def get_inference_score(request: EvaluateModelRequest):
    try:
        inference_score_result = wrap_inference_score(request)
        return inference_score_result
    except Exception as e:
        raise e


def wrap_inference_score(request: EvaluateModelRequest):
    from scoring.vibe_score import get_vibe_match_score
    from scoring.coherence_score import get_coherence_score

    model_name = f"{request.repo_namespace}/{request.repo_name}"
    # For simplicity, will always look for main branch
    revision = "main"
    print(f"modelname : {model_name}")
    for i in range(torch.cuda.device_count()):
        print(f"debug_cuda_devices_available : {torch.cuda.get_device_properties(i).name}")
    model = LLM(
        model=model_name,
        revision=revision,
        tensor_parallel_size=torch.cuda.device_count(),
        max_num_seqs=MAX_NUM_SEQS,
        max_model_len=MAX_SEQ_LEN_COHERENCE_SCORE,
        download_dir=MODEL_CACHE_DIR,
    )

    vibe_result = get_vibe_match_score(request, model)
    coherence_result = {}
    coherence_result = get_coherence_score(request, model)
    
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
