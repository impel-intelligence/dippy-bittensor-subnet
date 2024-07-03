import hashlib
import os
import shutil
import subprocess
import time
from datetime import datetime

import requests

from transformers import AutoConfig, AutoModelForCausalLM


def parse_size(line):
    """
    Parse the size string with unit and convert it to bytes.

    Args:
    - size_with_unit (str): The size string with unit (e.g., '125 MB')

    Returns:
    - int: The size in bytes
    """
    try:
        # get number enclosed in brackets
        size, unit = line[line.find("(") + 1 : line.rfind(")")].strip().split(" ")
        size = float(size.replace(",", ""))  # Remove commas for thousands
        unit = unit.lower()
        if unit == "kb":
            return int(size * 1024)
        elif unit == "mb":
            return int(size * 1024 * 1024)
        elif unit == "gb":
            return int(size * 1024 * 1024 * 1024)
        elif unit == "b":
            return int(size)  # No conversion needed for bytes
        else:
            raise ValueError(f"Unknown unit: {unit}")
    except ValueError as e:
        print(f"Error parsing size string '{size}{unit}': {e}")
        return 0


def check_model_repo_size(hash: str, repo_namespace: str, repo_name: str) -> int:
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
    repo_dir = (
        f"/tmp/validation_api_models/{hash}/models--{repo_namespace}--{repo_name}/{now}"
    )

    max_retries = 3
    for attempt in range(max_retries):
        try:
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--no-checkout",
                    f"https://huggingface.co/{repo_namespace}/{repo_name}",
                    repo_dir,
                ],
                check=True,
                timeout=15,
            )
            lfs_files_output = subprocess.check_output(
                ["git", "lfs", "ls-files", "-s"],
                text=True,
                timeout=10,
                cwd=repo_dir,
            )
            total_size = sum(
                parse_size(line)
                for line in lfs_files_output.strip().split("\n")
                if line
            )
            return total_size
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


def regenerate_hash(namespace, name, chat_template, competition_id):
    s = " ".join([namespace, name, chat_template, competition_id])
    hash_output = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return int(
        hash_output[:16], 16
    )  # Returns a 64-bit integer from the first 16 hexadecimal characters


def get_model_size(repo_namespace: str, repo_name: str):
    safetensor_index = f"https://huggingface.co/{repo_namespace}/{repo_name}/resolve/main/model.safetensors.index.json"
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
