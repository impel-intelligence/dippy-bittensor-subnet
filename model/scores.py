from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

class StrEnum(str, Enum):
    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value

    @classmethod
    def from_string(cls, value: str):
        try:
            return cls(value.upper())
        except ValueError:
            raise ValueError(f"{value} is not a valid {cls.__name__}")

class StatusEnum(StrEnum):
    QUEUED = "QUEUED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    RUNNING = "RUNNING"


class Scores(BaseModel):
    total_score: float  = Field(default=0, description="The total score of the evaluation")
    coherence_score: float = Field(default=0, description="The coherence score of the text")
    vibe_score: float  = Field(default=0, description="The vibe score of the text")
    status: str = Field(default=StatusEnum.QUEUED, description="The current status of the scoring process")


