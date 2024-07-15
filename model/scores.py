from enum import Enum
from typing import Optional, Any, Dict

from pydantic import BaseModel, Field
import math

CREATIVITY_STEEPNESS = 5
CREATIVITY_THRESHOLD = 0.2
QUALITATIVE_SCORE_WEIGHT = 0.82  # weight of the qualitative score in the total score
MODEL_SIZE_SCORE_WEIGHT = 0.06  # weight of the model size score in the total score
LATENCY_SCORE_WEIGHT = 0.06  # weight of the latency score in the total score
VIBE_SCORE_WEIGHT = 0.06  # weight of the vibe score in the total score


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
    total_score: float = Field(default=0, description="The total score of the evaluation")
    coherence_score: float = Field(default=0, description="The coherence score of the text")
    vibe_score: float = Field(default=0, description="The vibe score of the text")
    creativity_score: float = Field(default=0, description="The creativity score")
    qualitative_score: float = Field(default=0, description="The qualitative score of the text")
    model_size_score: float = Field(default=0, description="The model_size score of the text")
    latency_score: float = Field(default=0, description="The latency score of the text")
    status: str = Field(default=StatusEnum.QUEUED, description="The current status of the scoring process")

    @staticmethod
    def adjusted_q_score(
        initial_score: float, creativity_score: float, threshold=CREATIVITY_THRESHOLD, steepness=CREATIVITY_STEEPNESS
    ):
        adjusted_score = initial_score / (1 + math.exp(-steepness * (creativity_score - threshold)))
        return adjusted_score

    def from_response(self, response: Dict[str, Any]):
        if response is None or len(response) < 1:
            self.total_score = 0
            return self
        self.model_size_score = response.get("model_size_score", 0)
        self.creativity_score = response.get("creativity_score", 0)
        self.qualitative_score = response.get("qualitative_score", 0)
        self.vibe_score = response.get("vibe_score", 0)
        self.coherence_score = response.get("coherence_score", 0)
        self.latency_score = response.get("latency_score", 0)
        return self

    def new_total_score(self) -> float:
        q_score = self.adjusted_q_score(self.qualitative_score, self.creativity_score)
        total_score = 0
        total_score += QUALITATIVE_SCORE_WEIGHT * q_score
        total_score += MODEL_SIZE_SCORE_WEIGHT * self.model_size_score
        total_score += LATENCY_SCORE_WEIGHT * self.latency_score
        total_score += VIBE_SCORE_WEIGHT * self.vibe_score
        total_score = total_score * self.coherence_score
        return total_score

    def classic_score(self) -> float:
        total_score = 0
        total_score += QUALITATIVE_SCORE_WEIGHT * self.qualitative_score
        total_score += MODEL_SIZE_SCORE_WEIGHT * self.model_size_score
        total_score += LATENCY_SCORE_WEIGHT * self.latency_score
        total_score += VIBE_SCORE_WEIGHT * self.vibe_score
        return total_score
