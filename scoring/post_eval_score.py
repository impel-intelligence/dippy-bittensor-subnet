import os
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from transformers import AutoTokenizer


# Import necessary modules and functions from the main API file
from scoring.common import (
    EvaluateModelRequest,
    MAX_SEQ_LEN_COHERENCE_SCORE,
    MAX_GENERATION_LENGTH,
    chat_template_mappings,
    COHERENCE_MAX_TOKENS,
    COHERENCE_EVAL_MODEL,
    COHERENCE_NUM_EVALS,
)


def get_post_eval_score(request: EvaluateModelRequest, model: LLM, verbose=False):
    try:
        repo_id = f"{request.repo_namespace}/{request.repo_name}"

        cscore = 1

        return {"post_eval_score": cscore}
    except Exception as e:
        if verbose:
            print(e)
        raise e


def run_benchmark(model: LLM):
    return
