import json
import tarfile
import io
import os
import time
import docker
from pydantic import BaseModel
from typing import Optional, Union

from scoring.entrypoint import _dl_dataset
from scoring.common import EvaluateModelRequest
from utilities.event_logger import EventLogger
from model.scores import Scores

DEFAULT_IMAGE_NAME = "grader:latest"
DEFAULT_HOME_DIR = "/home/new_prod_user/dippy-bittensor-subnet"


class EvaluationScore(BaseModel):
    eval_score: float
    latency_score: float
    eval_model_size_score: float
    creativity_score: float


class RunError(BaseModel):
    error: str


class VibeScore(BaseModel):
    vibe_score: float


class CoherenceScore(BaseModel):
    coherence_score: int


class Evaluator:
    def __init__(
        self,
        image_name: str = DEFAULT_IMAGE_NAME,
        logger: EventLogger = EventLogger(),
        trace: bool = False,
    ):
        self.client = docker.from_env()
        self.logger = logger
        self.image_name = image_name
        self.volume_configuration = {
            f"{DEFAULT_HOME_DIR}/scoring/prompt_templates": {
                "bind": "/app/prompt_templates",
                "mode": "ro",
            },
            f"{DEFAULT_HOME_DIR}/datasets": {
                "bind": "/app/datasets",
                "mode": "rw",
            },
        }
        if trace:
            self.volume_configuration[f"{DEFAULT_HOME_DIR}/scoring"] = {
                "bind": "/app/scoring",
                "mode": "ro",
            }
            self.volume_configuration[f"{DEFAULT_HOME_DIR}/evalsets"] = {
                "bind": "/app/evalsets",
                "mode": "rw",
            }
        self.device_requests = [docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])]
        self.env = {
            "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
            "HF_TOKEN": os.environ.get("HF_TOKEN"),
            "COHERENCE_DATASET_NAME": os.environ.get("COHERENCE_DATASET_NAME"),
        }
        self.trace = trace

    def run_docker_container(
        self,
        job_type: str,
        request: EvaluateModelRequest,
    ) -> dict:
        # Configure volume mounting
        volumes = self.volume_configuration
        # Configure GPU support
        device_requests = self.device_requests

        command = f"{job_type} {request.to_args()}"
        self.logger.debug("command", command=command)

        # Run the container
        container = self.client.containers.run(
            self.image_name,
            command=command,
            volumes=volumes,
            device_requests=device_requests,
            environment=self.env,
            detach=True,  # Run in background
        )
        filepath = f"/tmp/{job_type}_output.json"
        filename = f"{job_type}_output.json"

        while container.status == "created":
            time.sleep(10)
            container.reload()
        result = container.wait()
        self.logger.debug(f"container_run_complete, {result}")

        try:
            bits, stat = container.get_archive(filepath)
            with io.BytesIO() as file_data:
                for chunk in bits:
                    file_data.write(chunk)
                file_data.seek(0)
                with tarfile.open(fileobj=file_data) as tar:
                    content = tar.extractfile(filename).read().decode("utf-8")
                    container_results = json.loads(content)
                    self.logger.info(
                        "container_run_results",
                        details={
                            "filepath": filepath,
                            "content": content,
                            "result": result,
                            "container_id": container.id,
                        },
                    )
                    if not self.trace:
                        container.remove()
                    return container_results
        except Exception as e:
            self.logger.error("docker_error", error=e)
            if not self.trace:
                container.remove()
            return {"error": e}

    def eval_score(self, request: EvaluateModelRequest) -> Union[EvaluationScore, RunError]:
        try:
            eval_result = self.run_docker_container(
                job_type="eval",
                request=request,
            )
            if "error" in eval_result:
                raise Exception(eval_result["error"])
            if eval_result["completed"] is False:
                raise Exception("completion internal error")
            score = EvaluationScore(
                eval_score=eval_result["eval_score"],
                latency_score=eval_result["latency_score"],
                eval_model_size_score=eval_result["model_size_score"],
                creativity_score=eval_result["creativity_score"],
            )
            return score
        except Exception as e:
            return RunError(error=str(e))

    def coherence_score(self, request: EvaluateModelRequest) -> Union[CoherenceScore, RunError]:
        try:
            coherence_result = self.run_docker_container(
                job_type="coherence",
                request=request,
            )
            if "error" in coherence_result:
                raise Exception(coherence_result["error"])
            if coherence_result["completed"] is False:
                raise Exception("completion internal error")
            score = CoherenceScore(
                coherence_score=coherence_result["coherence_score"],
            )
            return score
        except Exception as e:
            return RunError(error=str(e))

    def vibe_score(self, request: EvaluateModelRequest) -> Union[VibeScore, RunError]:
        try:
            vibe_result = self.run_docker_container(
                job_type="vibe",
                request=request,
            )
            if "error" in vibe_result:
                raise Exception(vibe_result["error"])
            if vibe_result["completed"] is False:
                raise Exception("completion internal error")
            score = VibeScore(
                vibe_score=vibe_result["vibe_score"],
            )
            return score
        except Exception as e:
            return RunError(error=str(e))


import math

STEEPNESS = 5
THRESHOLD = 0.2


def calculate_c_score(initial_score, creativity_score, threshold=0.2, steepness=5):
    final_score = initial_score / (1 + math.exp(-steepness * (creativity_score - threshold)))
    return final_score


# Command to manually run evaluation
def entry():
    _dl_dataset()
    # add command line arguments for the ports of the two apis
    import argparse

    parser = argparse.ArgumentParser(description="Run a single evaluation instance")
    parser.add_argument("--image", type=str, default="grader:latest", help="image to use")
    parser.add_argument("--repo_namespace", type=str, required=True, help="Repository namespace")
    parser.add_argument("--repo_name", type=str, required=True, help="Repository name")
    parser.add_argument("--chat_template_type", type=str, required=True, help="Chat template type")
    parser.add_argument("--hash", type=str, required=True, help="Unique hash value")

    args = parser.parse_args()
    image_name = args.image
    req = EvaluateModelRequest(
        repo_namespace=args.repo_namespace,
        repo_name=args.repo_name,
        chat_template_type=args.chat_template_type,
        hash=args.hash,
    )
    print(f"running {image_name} with {req}")

    try:
        evaler = Evaluator(image_name=image_name, trace=True)
        eval_result = evaler.eval_score(req)
        print(f"eval_result : {eval_result}")
        if isinstance(eval_result, RunError):
            raise Exception(eval_result.error)
        vibe_result = evaler.vibe_score(req)
        if isinstance(vibe_result, RunError):
            raise Exception(vibe_result.error)
        print(f"vibe_result : {vibe_result}")
        print("coherence start")
        coherence_result = evaler.coherence_score(req)
        if isinstance(coherence_result, RunError):
            raise Exception(coherence_result.error)
        print(f"coherence_result : {coherence_result}")

        scores_data = Scores()
        scores_data.qualitative_score = eval_result.eval_score
        scores_data.latency_score = eval_result.latency_score
        scores_data.creativity_score = eval_result.creativity_score
        scores_data.llm_size_score = eval_result.eval_model_size_score
        scores_data.coherence_score = coherence_result.coherence_score
        scores_data.vibe_score = vibe_result.vibe_score

        final_eval_score = (
            scores_data.adjusted_q_score(
                scores_data.qualitative_score,
                scores_data.creativity_score,
            )
            * 0.82
        )
        final_model_size_score = scores_data.llm_size_score * 0.06
        final_latency_score = scores_data.latency_score * 0.06
        final_vibe_score = scores_data.vibe_score * 0.06

        total_score = final_eval_score + final_model_size_score + final_latency_score + final_vibe_score
        print(f"final_model_size_score {final_model_size_score}")
        print(f"final_latency_score {final_latency_score}")
        print(f"final_vibe_score {final_vibe_score}")
        print(f"final_eval_score {final_eval_score}")
        print(f"coherence score: {scores_data.coherence_score}")
        print(f"score pre coherence: {total_score}")
        print(f"total score: {scores_data.calculate_total_score()}")
    except Exception as e:
        print(e)


if __name__ == "__main__":
    entry()
