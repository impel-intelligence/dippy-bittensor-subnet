from typing import Optional
from pydantic import BaseModel

# Constants
MAX_GENERATION_LEEWAY = 0.5  # should be between 0 and 1. This is the percentage of tokens that the model can generate more than the last user message
MAX_GENERATION_LENGTH = 200  # maximum number of tokens that the model can generate
LENGTH_DIFF_PENALTY_STEEPNESS = 2  # the steepness of the exponential decay of the length difference penalty
MAX_AVG_LATENCY = 10000  # in milliseconds
CREATIVITY_SCALE_FACTOR = 5

MAX_MODEL_SIZE = 32 * 1024 * 1024 * 1024  # in bytes
MIN_REPO_SIZE = 10 * 1024 * 1024  # in bytes
MAX_REPO_SIZE = 80 * 1024 * 1024 * 1024  #  in bytes
SAMPLE_SIZE = 1024  # number of samples to evaluate the model from the dataset
BATCH_SIZE = 4  # batch size for evaluation
VOCAB_TRUNCATION = 1000  # truncate the vocab to top n tokens
PROB_TOP_K = 10  # the correct token should be in the top n tokens, else a score of 0 is given to that token
# TODO: this will truncate the sequence to MAX_SEQ_LEN tokens. This is a temporary fix to make the evaluation faster.
MAX_SEQ_LEN = (
    4096  # maximum sequence length that should be allowed because eval gets really slow with longer sequences than this
)

MAX_SEQ_LEN_VIBE_SCORE = 2048  # maximum sequence length that should be allowed for vibe score calculation because it is slow with longer sequences than this
BATCH_SIZE_VIBE_SCORE = 4  # batch size for vibe score calculation
SAMPLE_SIZE_VIBE_SCORE = 128  # number of samples to evaluate the model from the dataset for vibe score calculation

VLLM_GPU_MEMORY = 0.4

SAVE_LEADERBOARD_EVERY = 60  # save the leaderboard every 60 seconds

COHERENCE_SAMPLE_SIZE = 16
COHERENCE_MAX_TOKENS = 1024
COHERENCE_NUM_EVALS = 64
COHERENCE_EVAL_MODEL = "gpt-4o"

PIPPA_FILENAME = "pippa_deduped.jsonl"
PROMPTS_1_FILENAME = "opus-writing-prompts-1-sharegpt.jsonl"
PROMPTS_2_FILENAME = "opus-writing-prompts-2-sharegpt.jsonl"

DATASET_DIR = "./datasets"


def full_path(filename: str) -> str:
    return f"{DATASET_DIR}/{filename}"


class EvaluateModelRequest(BaseModel):
    repo_namespace: str
    repo_name: str
    chat_template_type: str
    hash: str
    revision: Optional[str] = "main"
    competition_id: Optional[str] = "d1"
    admin_key: Optional[str] = "admin_key"

    def to_args(self) -> str:
        return " ".join([self.repo_name, self.repo_namespace, self.chat_template_type, self.hash])


chat_template_mappings = {
    "vicuna": "prompt_templates/vicuna_prompt_template.jinja",
    "chatml": "prompt_templates/chatml_prompt_template.jinja",
    "mistral": "prompt_templates/mistral_prompt_template.jinja",
    "zephyr": "prompt_templates/zephyr_prompt_template.jinja",
    "alpaca": "prompt_templates/alpaca_prompt_template.jinja",
    "llama2": "prompt_templates/llama2_prompt_template.jinja",
    "llama3": "prompt_templates/llama3_prompt_template.jinja",
}
