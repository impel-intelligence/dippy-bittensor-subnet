import requests
from huggingface_hub.utils import build_hf_headers, hf_raise_for_status
import os

ENDPOINT = "https://huggingface.co"

REPO_TYPES = ["model", "dataset", "space"]

hf_token = os.environ["HF_ACCESS_TOKEN"]


def duplicate(repo_namespace: str, repo_name: str):
    destination = f"DippyAI/{repo_namespace}-{repo_name}"
    r = requests.post(
        f"https://huggingface.co/api/models/{repo_namespace}/{repo_name}/duplicate",
        headers=build_hf_headers(token=hf_token),
        json={"repository": destination, "private": True},
    )
    hf_raise_for_status(r)

    repo_url = r.json().get("url")

    return (f"{repo_url}",)
