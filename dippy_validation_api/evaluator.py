import json
import tarfile
import io
import time
import docker
from pydantic import BaseModel
from typing import Optional, Union

from model_evaluation.common import EvaluateModelRequest
from utilities.event_logger import EventLogger

DEFAULT_IMAGE_ID = "grader:latest"

class EvaluationScore(BaseModel):
    eval_score: float
    latency_score: float
    eval_model_size_score: float


class RunError(BaseModel):
    error: str


class VibeScore(BaseModel):
    vibe_score: float


class Evaluator:
    def __init__(
        self,
        image_name: str = DEFAULT_IMAGE_ID,
        logger: EventLogger = EventLogger(),
    ):
        self.client = docker.from_env()
        self.logger = logger
        self.image_name = image_name
        self.volume_configuration = {
            "/home/new_prod_user/dippy-bittensor-subnet/model_evaluation/prompt_templates": {
                "bind": "/app/prompt_templates",
                "mode": "rw",
            },
            "/home/new_prod_user/dippy-bittensor-subnet/dippy_validation_api/data": {
                "bind": "/app/data",
                "mode": "rw",
            },
        }
        self.device_requests = [
            docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])
        ]
        self.env = {
            "SOME_ENV_VAR": "x",
        }

    def run_docker_container(
        self,
        job_type: str,
        request: EvaluateModelRequest,
    ) -> dict:
        # Configure volume mounting
        volumes = self.volume_configuration
        # Configure GPU support
        device_requests = [docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])]

        command = f"{job_type} {request.to_args()}"
        self.logger.debug("command", command=command)

        # Run the container
        container = self.client.containers.run(
            self.image_name,
            command=command,
            volumes=volumes,
            device_requests=device_requests,
            environment={
                "SOME_ENV_VAR": "x",
            },
            detach=True,  # Run in background
        )
        filepath = f"/tmp/{job_type}_output.json"
        filename = f"{job_type}_output.json"
        result = container.wait()
        while container.status == 'created':
            time.sleep(10)
            container.reload()
        while container.status == 'running':
            time.sleep(30)
            container.reload()

        self.logger.debug("container_run_complete")

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
                    container.remove()
                    return container_results
        except Exception as e:
            self.logger.error("docker_error", error=e)
            container.remove()
            return {"error": e}

    def eval_score(
        self, request: EvaluateModelRequest
    ) -> Union[EvaluationScore, RunError]:
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
            )
            return score
        except Exception as e:
            return RunError(error=f"{e}")

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
            return RunError(error=f"{e}")
