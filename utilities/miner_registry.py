from typing import Any, Dict, Optional

from pydantic import BaseModel, Field
import sys
from model.data import ModelId


class MinerEntry(BaseModel):
    block: int = Field(default=sys.maxsize, description="The block number")
    hotkey: str = Field(default="", description="The hotkey of the miner") 
    invalid: bool = Field(default=False, description="invalidity of determining score")
    miner_model_id: Optional[ModelId] = Field(default=None, description="The model_id of the miner")
    safetensors_model_size: int = Field(default=0, description="The safetensors model size according to huggingface")
    total_score: float = Field(default=0)
