from enum import Enum
from typing import Optional, Any, Dict

from pydantic import BaseModel, Field
import math

CREATIVITY_STEEPNESS = 8
CREATIVITY_THRESHOLD = 0.5

QUALITATIVE_SCORE_THRESHOLD = 0.25
LLM_MODEL_SIZE_THRESHOLD = 0.75
LLM_MODEL_SIZE_STEEPNESS = 8
QUALITATIVE_SCORE_WEIGHT = 0.84  # weight of the qualitative score in the total score
LATENCY_SCORE_WEIGHT = 0.08  # weight of the latency score in the total score
VIBE_SCORE_WEIGHT = 0.08  # weight of the vibe score in the total score
COHERENCE_MINIMUM = 0.95


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
    PRECHECK = "PRECHECK"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    RUNNING = "RUNNING"


class Scores(BaseModel):
    total_score: float = Field(default=0, description="The total score of the evaluation")
    coherence_score: float = Field(default=0, description="The coherence score of the model")
    vibe_score: float = Field(default=0, description="The vibe score of the model")
    creativity_score: float = Field(default=0, description="The creativity score")
    qualitative_score: float = Field(default=0, description="The qualitative score of the model")
    llm_size_score: float = Field(default=0, description="The llm model size score")
    latency_score: float = Field(default=0, description="The latency score of the model")
    post_eval_score: float = Field(default=1, description="The post evaluation score (multiplier) of the model")
    status: str = Field(default=StatusEnum.QUEUED, description="The current status of the scoring process")

    @staticmethod
    def adjusted_q_score(
        initial_score: float, creativity_score: float, threshold=CREATIVITY_THRESHOLD, steepness=CREATIVITY_STEEPNESS
    ):
        # Calculate exponential scaling factor based on distance from midpoint
        creativity_impact = math.exp(steepness * (initial_score - QUALITATIVE_SCORE_THRESHOLD))

        # Combine initial score with creativity adjustment
        adjusted_score = initial_score / (1 + creativity_impact * math.exp(-steepness * (creativity_score - threshold)))
        return adjusted_score

    @staticmethod
    def model_size_adjuster(
        model_size_score: float, threshold=LLM_MODEL_SIZE_THRESHOLD, steepness=LLM_MODEL_SIZE_STEEPNESS
    ):
        if model_size_score < threshold:
            # Exponential penalty that increases as score drops below threshold
            penalty_multiplier = pow(model_size_score / threshold, steepness)
            return penalty_multiplier

        return 1

    def from_response(self, response: Dict[str, Any]):
        if response is None or len(response) < 1:
            self.total_score = 0
            return self
        self.llm_size_score = response.get("model_size_score", 0)
        self.creativity_score = response.get("creativity_score", 0)
        self.qualitative_score = response.get("qualitative_score", 0)
        self.vibe_score = response.get("vibe_score", 0)
        self.coherence_score = response.get("coherence_score", 0)
        self.latency_score = response.get("latency_score", 0)
        self.post_eval_score = response.get("post_eval_score", 1)
        return self

    def calculate_total_score(self, adjust_coherence: bool = False) -> float:
        q_score = self.adjusted_q_score(self.qualitative_score, self.creativity_score)
        total_score = 0
        total_score += QUALITATIVE_SCORE_WEIGHT * q_score
        total_score += LATENCY_SCORE_WEIGHT * self.latency_score
        total_score += VIBE_SCORE_WEIGHT * self.vibe_score
        self.coherence_score = 1 if self.coherence_score >= COHERENCE_MINIMUM else 0
        total_score = total_score * self.coherence_score
        # multiplier = self.model_size_adjuster(self.llm_size_score)
        # total_score = total_score * multiplier
        total_score = total_score * self.post_eval_score
        return total_score


def main():
    import random

    # Create a Scores instance with random values
    scores = Scores(
        qualitative_score=random.uniform(0.25, 0.70),
        creativity_score=random.uniform(0, 1),
        vibe_score=random.uniform(0, 1),
        coherence_score=1,
        llm_size_score=random.uniform(0, 1),
        latency_score=random.uniform(0, 1),
        post_eval_score=1.0,
    )

    # Print input scores
    print("\nInput Scores:")
    print(f"Qualitative Score: {scores.qualitative_score:.3f}")
    print(f"Creativity Score: {scores.creativity_score:.3f}")
    print(f"Vibe Score: {scores.vibe_score:.3f}")
    print(f"Coherence Score: {scores.coherence_score:.3f}")
    print(f"LLM Size Score: {scores.llm_size_score:.3f}")
    print(f"Latency Score: {scores.latency_score:.3f}")

    # Calculate and break down total score
    adjusted_q = scores.adjusted_q_score(scores.qualitative_score, scores.creativity_score)
    print(f"\nAdjusted Qualitative Score: {adjusted_q:.3f}")

    coherence_binary = 1 if scores.coherence_score >= COHERENCE_MINIMUM else 0
    print(f"Coherence Binary: {coherence_binary}")

    size_multiplier = scores.model_size_adjuster(scores.llm_size_score)
    print(f"Size Multiplier: {size_multiplier:.3f}")

    # Calculate weighted scores
    weighted_q = QUALITATIVE_SCORE_WEIGHT * adjusted_q
    weighted_latency = LATENCY_SCORE_WEIGHT * scores.latency_score
    weighted_vibe = VIBE_SCORE_WEIGHT * scores.vibe_score

    print(f"\nWeighted Scores:")
    print(f"Weighted Qualitative: {weighted_q:.3f}")
    print(f"Weighted Latency: {weighted_latency:.3f}")
    print(f"Weighted Vibe: {weighted_vibe:.3f}")

    # Calculate final score
    total = scores.calculate_total_score()
    print(f"\nFinal Total Score: {total:.3f}")


if __name__ == "__main__":
    main()
