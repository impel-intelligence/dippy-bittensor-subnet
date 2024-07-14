from typing import Any, Dict, Optional

from pydantic import BaseModel, Field
import sys
from model.data import ModelId


class MinerEntry(BaseModel):
    block: int = Field(default=sys.maxsize, description="The block number")
    hotkey: Optional[str] = Field(default_factory=None, description="The hotkey of the miner")
    model_id: Optional[ModelId] = Field(default_factory=None, description="The hotkey of the miner")
    total_score: float = 0
