from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class MinerEntry(BaseModel):
    block: Optional[int] = Field(default_factory=None, description="The block number")
    hotkey: Optional[str] = Field(default_factory=None, description="The hotkey of the miner")
    score_data: Dict[str, Any] = Field(default_factory=dict, description="Score data for the miner")
    total_score: float = 0
