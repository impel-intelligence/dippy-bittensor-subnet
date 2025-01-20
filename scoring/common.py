from typing import Optional
from pydantic import BaseModel

# Constants
MAX_GENERATION_LEEWAY = 0.5  # should be between 0 and 1. This is the percentage of tokens that the model can generate more than the last user message
MAX_GENERATION_LENGTH = 200  # maximum number of tokens that the model can generate
LENGTH_DIFF_PENALTY_STEEPNESS = 2  # the steepness of the exponential decay of the length difference penalty
MAX_AVG_LATENCY = 10000  # in milliseconds
CREATIVITY_SCALE_FACTOR = 5

MAX_MODEL_SIZE = 72 * 1024 * 1024 * 1024  # in bytes
MIN_REPO_SIZE = 10 * 1024 * 1024  # in bytes
SAMPLE_SIZE = 1024  # number of samples to evaluate the model from the dataset
EVALUATION_DATASET_SAMPLE_SIZE = 4096  # number of samples to evaluate the model from the dataset
BATCH_SIZE = 4  # batch size for evaluation
# VOCAB_TRUNCATION = 1000  # truncate the vocab to top n tokens
VOCAB_TRUNCATION = 10  # truncate the vocab to top n tokens
PROB_TOP_K = 10  # the correct token should be in the top n tokens, else a score of 0 is given to that token
# TODO: this will truncate the sequence to MAX_SEQ_LEN tokens. This is a temporary fix to make the evaluation faster.
MAX_SEQ_LEN = (
    4096  # maximum sequence length that should be allowed because eval gets really slow with longer sequences than this
)

MAX_SEQ_LEN_VIBE_SCORE = 2048  # maximum sequence length that should be allowed for vibe score calculation because it is slow with longer sequences than this
MAX_SEQ_LEN_COHERENCE_SCORE = 8192
BATCH_SIZE_VIBE_SCORE = 4  # batch size for vibe score calculation
SAMPLE_SIZE_VIBE_SCORE = 128  # number of samples to evaluate the model from the dataset for vibe score calculation
# number of samples to evaluate the model from the dataset for coherence score calculation
SAMPLE_SIZE_COHERENCE_SCORE = 128


VLLM_GPU_MEMORY = 0.4

SAVE_LEADERBOARD_EVERY = 60  # save the leaderboard every 60 seconds

COHERENCE_BATCH_SIZE = 16
COHERENCE_MAX_TOKENS = 1024
COHERENCE_NUM_EVALS = 256
COHERENCE_EVAL_MODEL = "openai/gpt-4o-2024-11-20"

DATASET_DIR = "evalsets"
MODEL_CACHE_DIR = "./model_cache_dir"

DIPPA_DATASET_MAX_PARTITIONS = 4


class EvaluateModelRequest(BaseModel):
    repo_namespace: str
    repo_name: str
    chat_template_type: str
    hash: str
    revision: Optional[str] = "main"
    competition_id: Optional[str] = "d1"
    admin_key: Optional[str] = "admin_key"
    hotkey: Optional[str] = ""
    block: Optional[int] = 0
    tokenizer: Optional[str] = "llama"

    def to_args(self) -> str:
        return " ".join([self.repo_name, self.repo_namespace, self.chat_template_type, self.hash])


chat_template_mappings = {
    "vicuna": "prompt_templates/vicuna_prompt_template.jinja",
    "chatml": "prompt_templates/chatml_prompt_template.jinja",
    "zephyr": "prompt_templates/zephyr_prompt_template.jinja",
    "mistral": "prompt_templates/mistral_prompt_template.jinja",
    "alpaca": "prompt_templates/alpaca_prompt_template.jinja",
    "llama2": "prompt_templates/llama2_prompt_template.jinja",
    "llama3": "prompt_templates/llama3_prompt_template.jinja",
    "llama3dot1": "prompt_templates/llama3dot1_prompt_template.jinja",
    "gemma2": "prompt_templates/gemma_it_prompt_template.jinja",
    "qwen2dot5": "prompt_templates/qwen2dot5_prompt_template.jinja",
}
