import shutil
from dataclasses import dataclass
from datetime import datetime
import time
from typing import Dict, List, Tuple, Optional
import json
import os
import subprocess
import hashlib
from typing import Dict, List, Tuple
import requests

from utilities.validation_utils import parse_size

SAFETENSORS_FILE = "model.safetensors.index.json"
env = {}
env["GIT_LFS_SKIP_SMUDGE"] = "1"
env["GIT_TERMINAL_PROMPT"] = "0"  # Disable prompts for username/password
env["GIT_ASKPASS"] = "echo"  # This ensures that no password prompt is shown


@dataclass
class ModelRepo:
    repo_size: int = 0
    model_hash: str = ""


def get_model_size(repo_namespace: str, repo_name: str):
    try:
        safetensor_index = (
            f"https://huggingface.co/{repo_namespace}/{repo_name}/resolve/main/model.safetensors.index.json"
        )
        response = requests.get(safetensor_index)
        if response.status_code != 200:
            print(f"Error getting safetensors index: {response.text}")
            return None

        response_json = response.json()
        if "metadata" not in response_json:
            print("Error: metadata not found in safetensors index")
            return None

        if "total_size" not in response_json["metadata"]:
            print("Error: total_size not found in safetensors index metadata")
            return None

        total_size = response_json["metadata"]["total_size"]

        return total_size
    except Exception as e:
        print(e)
        return None


def check_model_repo_details(hash: str, repo_namespace: str, repo_name: str) -> Optional[ModelRepo]:
    """
    Check the size of a model hosted on Hugging Face using Git LFS without checking out the files,
    and clean up the cloned repository afterwards, even if an error occurs.

    Args:
    - hash (int): The hash of the model
    - repo_namespace (str): The namespace of the model repository
    - repo_name (str): The name of the model repository

    Returns:
    - int: The total size of the model files in bytes
    """
    now = datetime.utcnow().isoformat()

    max_retries = 3
    for attempt in range(max_retries):
        repo_dir = f"/tmp/validation_api_models/{hash}/models--{repo_namespace}--{repo_name}/{now}/{attempt}"
        try:
            subprocess.run(
                [
                    "git",
                    "clone",
                    f"https://huggingface.co/{repo_namespace}/{repo_name}",
                    repo_dir,
                ],
                check=True,
                env=env,
                timeout=600,
            )
            lfs_files_output = subprocess.check_output(
                ["git", "lfs", "ls-files", "-s"],
                text=True,
                timeout=10,
                cwd=repo_dir,
            )
            total_size = sum(parse_size(line) for line in lfs_files_output.strip().split("\n") if line)
            m = SafetensorsModel(repo_dir)

            return ModelRepo(repo_size=total_size, model_hash=m.id())
        except subprocess.TimeoutExpired as e:
            print(f"Operation timed out on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                print("Retrying...")
                time.sleep(2**attempt)  # Exponential backoff
            else:
                print("Max retries exceeded.")
                return None
        except Exception as e:
            print(f"An error occurred on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                print("Retrying...")
                time.sleep(2**attempt)  # Exponential backoff
            else:
                print("Max retries exceeded.")
                raise e
        finally:
            shutil.rmtree(repo_dir, ignore_errors=True)


class SafetensorsModel:
    def __init__(self, repo_path: str):
        safetensors_filepath = os.path.join(repo_path, SAFETENSORS_FILE)
        """
        Initialize the SafetensorsModelIdentifier with the path to the index file.

        Args:
        index_path (str): Path to the model.safetensors.index.json file.
        """
        self.safetensors_filepath = safetensors_filepath
        self.repo_path = repo_path
        if not os.path.exists(self.safetensors_filepath):
            raise FileNotFoundError(f"Index file not found: {self.safetensors_filepath}")
        self.model_files = self._get_model_files()
        self.file_hashes = self._collect_file_hashes()
        self.full_hash = self.generate_model_identifier()

    def _get_model_files(self) -> List[str]:
        """Get list of safetensors files from the index."""
        with open(self.safetensors_filepath, "r") as f:
            index_data = json.load(f)
        return list(set(index_data.get("weight_map", {}).values()))

    def _collect_file_hashes(self) -> Dict[str, str]:
        """Collect LFS hashes for model files."""
        file_hashes = {}
        for filename in self.model_files:
            file_hash = self._get_lfs_hash(filename)
            if file_hash:
                file_hashes[filename] = file_hash
        return file_hashes

    def _get_lfs_hash(self, file_path: str) -> Optional[str]:
        full_path = os.path.join(self.repo_path, file_path)
        try:
            result = subprocess.run(
                ["git", "lfs", "pointer", "--file", full_path], capture_output=True, text=True, check=True, timeout=10
            )
            for line in result.stdout.splitlines():
                if line.startswith("oid sha256:"):
                    return line.split(":")[1]
        except Exception as e:
            return None
        return None

    def generate_model_identifier(self) -> str:
        """Generate a unique identifier for the model."""
        hasher = hashlib.sha256()

        # Hash the index file content
        with open(self.safetensors_filepath, "rb") as f:
            hasher.update(f.read())

        # Hash the sorted file names and their LFS hashes
        for _, hash_lfs in sorted(self.file_hashes.items()):
            hasher.update(hash_lfs.encode())

        return hasher.hexdigest()

    def id(self) -> str:
        return self.full_hash
