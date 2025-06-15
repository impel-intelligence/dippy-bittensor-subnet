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

