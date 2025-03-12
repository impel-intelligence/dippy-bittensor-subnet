import argparse
import copy
import docker
import io
import json
import math
import os
import tarfile
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Union, Dict, Any

from pydantic import BaseModel

from common.scores import Scores
from scoring.common import EvaluateModelRequest
from utilities.event_logger import EventLogger

# Constants
DEFAULT_IMAGE_NAME = "grader:latest"
DEFAULT_HOME_DIR = os.environ.get("EVALUATOR_HOME_DIR", "/home/new_prod_user/dippy-bittensor-subnet")
DEFAULT_MODEL_CACHE_DIR = os.environ.get("EVALUATOR_MODEL_CACHE_DIR", "/workdir/model_cache_dir")
STEEPNESS = 5
THRESHOLD = 0.2

DEFAULT_HOME_DIR = os.environ.get("EVALUATOR_HOME_DIR", "/home/new_prod_user/dippy-bittensor-subnet")
DEFAULT_MODEL_CACHE_DIR = os.environ.get("EVALUATOR_MODEL_CACHE_DIR", "/workdir/model_cache_dir")


class EvaluationScore(BaseModel):
    eval_score: float
    latency_score: float
    creativity_score: float


class RunError(BaseModel):
    error: str


class CoherenceScore(BaseModel):
    coherence_score: float


class InferenceScore(BaseModel):
    coherence_score: float
    judge_score: float


class Evaluator:
    def __init__(
        self,
        image_name: str = DEFAULT_IMAGE_NAME,
        gpu_ids: str = "0",
        model_dir: str = "",
        logger: EventLogger = EventLogger(),
        trace: bool = False,
    ):
        self.client = docker.from_env(version="auto", timeout=600)
        self.logger = logger

        if trace:
            self.logger = EventLogger(
                filepath="/tmp/valapi_event_logs/trace_{time:UNIX}.log",
                level="DEBUG",
                stderr=True,
            )
        self.image_name = image_name

        prompt_template_path = os.path.join(DEFAULT_HOME_DIR, "scoring/prompt_templates")
        evalsets_template_path = os.path.join(DEFAULT_HOME_DIR, "evalsets")

        prompt_template_path = str(prompt_template_path)

        self.volume_configuration = {
            prompt_template_path: {
                "bind": "/app/prompt_templates",
                "mode": "ro",
            },
            evalsets_template_path: {
                "bind": "/app/evalsets",
                "mode": "rw",
            },
        }
        if trace:
            scoring_path = os.path.join(DEFAULT_HOME_DIR, "scoring")
            self.volume_configuration[str(scoring_path)] = {
                "bind": "/app/scoring",
                "mode": "ro",
            }
            self.volume_configuration[DEFAULT_MODEL_CACHE_DIR] = {
                "bind": "/app/model_cache_dir",
                "mode": "rw",
            }

        self.device_requests = [docker.types.DeviceRequest(device_ids=[gpu_ids], capabilities=[["gpu"]])]

        self.env = {
            "OPENROUTER_API_KEY": os.environ.get("OPENROUTER_API_KEY"),
            "HF_TOKEN": os.environ.get("HF_TOKEN"),
            "AZURE_URL": os.environ.get("AZURE_URL"),
            "AZURE_KEY": os.environ.get("AZURE_KEY"),
            "VLLM_WORKER_MULTIPROC_METHOD": "_",
            "PYTORCH_CUDA_ALLOC_CONF": "_",
            "VLLM_USE_V1": "1",
            "DATASET_API_JWT": os.environ.get("DATASET_API_JWT"),
            "DATASET_API_KEY": os.environ.get("DATASET_API_KEY"),
        }
        self.trace = trace
        if len(model_dir) > 0:
            self.volume_configuration[model_dir] = {
                "bind": "/app/model_dir",
                "mode": "ro",
            }
            self.env["USE_MODEL_DIR"] = 1

    def run_docker_container(
        self,
        job_type: str,
        request: EvaluateModelRequest,
    ) -> dict:
        volumes = self.volume_configuration
        device_requests = self.device_requests

        command = f"{job_type} {request.to_args()}"
        self.logger.info("command", command=command)
        self.logger.info("device_requests", device_requests=device_requests)

        env = copy.copy(self.env)
        if job_type == "eval":
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            env["VLLM_WORKER_MULTIPROC_METHOD"] = "_"
        if job_type == "inference":
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"
            env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
            env["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
            env["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"

        if job_type == "inference_flash":
            env["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"
            env["VLLM_FLASHINFER_FORCE_TENSOR_CORES"] = "1"

        self.logger.debug("env", env=env)

        container = self.client.containers.run(
            self.image_name,
            command=command,
            volumes=volumes,
            device_requests=device_requests,
            environment=env,
            detach=True,  # Run in background
        )
        filepath = f"/tmp/{job_type}_output.json"
        filename = f"{job_type}_output.json"

        print(f"container_run_started {self.image_name} with command {command} to complete")
        result = container.wait()
        self.logger.debug(f"container_run_complete, {result}")
        print(f"container_run_complete for container {self.image_name} {command} {result}")

        try:
            bits, _ = container.get_archive(filepath)
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
                        try:
                            container.remove()
                        except Exception as e:
                            self.logger.error("container_remove_error")
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
                eval_score=eval_result.get("eval_score", -1),
                latency_score=eval_result.get("latency_score", -1),
                creativity_score=eval_result.get("creativity_score", 0),
            )
            return score
        except Exception as e:
            return RunError(error=str(e))

    def inference_score(self, request: EvaluateModelRequest) -> Union[InferenceScore, RunError]:
        try:
            inference_result = self.run_docker_container(
                job_type="inference",
                request=request,
            )
            if "error" in inference_result:
                raise Exception(inference_result["error"])
            if inference_result["completed"] is False:
                raise Exception("completion internal error")
            score = InferenceScore(
                coherence_score=inference_result.get("coherence_score", 0),
                judge_score=inference_result.get("judge_score", 0),
            )
            return score
        except Exception as e:
            return RunError(error=str(e))


def calculate_c_score(initial_score, creativity_score, threshold=THRESHOLD, steepness=STEEPNESS):
    final_score = initial_score / (1 + math.exp(-steepness * (creativity_score - threshold)))
    return final_score

"""
python worker_api/evaluator.py --repo_namespace DippyAI --repo_name gemma-27b-reference --model_dir /optional/model/path --chat_template_type gemma2 --hash x
"""
# Command to manually run evaluation
def cmd():
    parser = argparse.ArgumentParser(description="Run a single evaluation instance")
    parser.add_argument("--image", type=str, default=DEFAULT_IMAGE_NAME, help="image to use")
    parser.add_argument("--model_dir", type=str, help="Local model directory path")
    parser.add_argument("--repo_namespace", type=str, help="Repository namespace")
    parser.add_argument("--repo_name", type=str, help="Repository name")
    parser.add_argument("--chat_template_type", type=str, required=True, help="Chat template type")
    parser.add_argument("--hash", type=str, required=True, help="Unique hash value")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save results")

    args = parser.parse_args()
    image_name = args.image

    if not args.repo_namespace or not args.repo_name:
        raise ValueError("Must specify either model_dir or both repo_namespace and repo_name")
    req = EvaluateModelRequest(
        repo_namespace=args.repo_namespace,
        repo_name=args.repo_name, 
        chat_template_type=args.chat_template_type,
        hash=args.hash,
    )
    print(f"running {image_name} with {req}")

    try:
        
        evaler = Evaluator(image_name=image_name, trace=True, gpu_ids="0")
        if args.model_dir:
            evaler = Evaluator(image_name=image_name, trace=True, gpu_ids="0",model_dir=args.model_dir)

        start_time = time.time()
        infrence_result = evaler.inference_score(req)
        elapsed_time = time.time() - start_time
        
        if isinstance(infrence_result, RunError):
            raise Exception(infrence_result.error)
        print(f"infrence_result : {infrence_result}")
        print(f"Time elapsed: {elapsed_time:.2f} seconds")

        scores_data = Scores()
        scores_data.qualitative_score = 0
        scores_data.latency_score = 0
        scores_data.creativity_score = 0
        scores_data.llm_size_score = 0
        scores_data.coherence_score = 1
        scores_data.judge_score = infrence_result.judge_score


        total_score = scores_data.judge_score
        
        results = {
            "timestamp": int(time.time()),
            "datetime": datetime.fromtimestamp(time.time()).isoformat(),
            "model": {
                "chat_template_type": req.chat_template_type,
                "hash": req.hash
            },
            "raw_scores": {
                "judge_score": scores_data.judge_score
            },
            "elapsed_time": elapsed_time
        }
        
        print(f"final_judge_score {total_score}")
        print(f"coherence score: {scores_data.coherence_score}")
        print(f"score pre coherence: {total_score}")
        print(f"total score: {scores_data.calculate_total_score()}")

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = int(time.time())
        output_file = output_dir / f"{timestamp}_evaluation_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults written to: {output_file}")

    except Exception as e:
        print(e)


if __name__ == "__main__":
    cmd()
