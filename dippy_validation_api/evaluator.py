from typing import Optional

import docker
import os

from pydantic import BaseModel

from dippy_validation_api.validation_api import EvaluateModelRequest

MAX_RUNNING_CONTAINERS=2

class EvaluationResult(BaseModel):
    foo: str

class Evaluator():
    def __init__(self):
        self.client = docker.from_env()
    def queue_eval_job(self, request: EvaluateModelRequest) -> Optional[EvaluationResult]:


        return None
    def queue_vibe_score_job(self, request: EvaluateModelRequest) -> Optional[EvaluationResult]:
        return None

# docker build -f evaluator.Dockerfile -t eval-llm .
def run_docker_container(image_name, command=None):

    # Configure volume mounting
    volumes = {
        "/home/new_prod_user/dippy-bittensor-subnet/dippy_validation_api/prompt_templates": {'bind': "/app/prompt_templates", 'mode': 'rw'},
        "/home/new_prod_user/dippy-bittensor-subnet/dippy_validation_api/data": {'bind': "/app/data", 'mode': 'rw'},
        "/home/new_prod_user/dippy-bittensor-subnet/playground": {'bind': "/app/playground", 'mode': 'rw'},
    }

    # Configure GPU support
    device_requests = [docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])]

    # Run the container
    container = client.containers.run(
        image_name,
        # command=command,
        volumes=volumes,
        device_requests=device_requests,
        detach=True  # Run in background
    )

    print(f"Container {container.id} is running.")
    print("Container logs:")
    for log in container.logs(stream=True):
        print(log.strip().decode())

def check_docker_queue():
    containers = client.containers.list()
    for c in containers:
       print(c)
       print(c.status)
       print(c.logs())


# Usage example
if __name__ == "__main__":
    IMAGE_NAME = "eval-llm:latest"

    # COMMAND = "python run_eval.py"  # Optional: command to run in the container

    check_docker_queue()
    # run_docker_container(IMAGE_NAME)