import typer
import torch
import json
from typing import Optional
import os
import sys
import gc
import traceback
import huggingface_hub

from scoring.common import (
    EvaluateModelRequest,
    PIPPA_FILENAME,
    full_path,
    PROMPTS_1_FILENAME,
    PROMPTS_2_FILENAME,
)

app = typer.Typer()


def _dl_dataset():
    # create dir data if not exists
    if not os.path.exists("./datasets"):
        os.makedirs("./datasets")


def write_to_json(data: dict, filepath: str = "/tmp/output.json"):
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    typer.echo(f"Results written to {filepath}")


def _run(
    request: EvaluateModelRequest,
    run_type: str,
):
    from scoring.get_eval_score import get_eval_score
    from scoring.coherence_score import get_coherence_score
    from scoring.vibe_score import get_vibe_match_score

    typer.echo(f"Evaluating with parameters: {request}")
    result = {"completed": False}
    try:
        if run_type == "eval":
            result = get_eval_score(request)
        if run_type == "vibe":
            result = get_vibe_match_score(request)
        if run_type == "coherence":
            result = get_coherence_score(request)
        result["completed"] = True
        typer.echo(f"Evaluated with parameters: {result}")
    except Exception as e:
        (
            exc_type,
            exc_value,
            exc_traceback,
        ) = sys.exc_info()  # Capture exception information
        traceback_details = traceback.format_exception(exc_type, exc_value, exc_traceback)
        result["error"] = f'{"".join(traceback_details)} {str(e)}'
    write_to_json(result, f"/tmp/{run_type}_output.json")


@app.command("eval")
def evaluate(
    repo_name: str = typer.Argument(help="Repository name"),
    repo_namespace: str = typer.Argument(help="Repository namespace"),
    chat_template_type: str = typer.Argument(help="Chat template type"),
    hash: Optional[str] = typer.Argument(help="hash"),
):
    request = EvaluateModelRequest(
        repo_namespace=repo_namespace,
        repo_name=repo_name,
        chat_template_type=chat_template_type,
        hash=hash,
    )
    _run(request=request, run_type="eval")


@app.command("coherence")
def coherence(
    repo_name: str = typer.Argument(help="Repository name"),
    repo_namespace: str = typer.Argument(help="Repository namespace"),
    chat_template_type: str = typer.Argument(help="Chat template type"),
    hash: Optional[str] = typer.Argument(help="hash"),
):
    request = EvaluateModelRequest(
        repo_namespace=repo_namespace,
        repo_name=repo_name,
        chat_template_type=chat_template_type,
        hash=hash,
    )
    _run(request=request, run_type="coherence")


@app.command("vibe")
def vibe_score(
    repo_name: str = typer.Argument(help="Repository name"),
    repo_namespace: str = typer.Argument(help="Repository namespace"),
    chat_template_type: str = typer.Argument(help="Chat template type"),
    hash: Optional[str] = typer.Argument(help="hash"),
):
    request = EvaluateModelRequest(
        repo_namespace=repo_namespace,
        repo_name=repo_name,
        chat_template_type=chat_template_type,
        hash=hash,
    )
    _run(request=request, run_type="vibe")


@app.command("stub")
def stub():
    print("stub")
    result = {"g": True}
    write_to_json(result, "/tmp/output.json")



# example: python entrypoint.py python entrypoint.py eval repo_name repo_namespace chat_template_type hash
if __name__ == "__main__":
    _dl_dataset()
    gc.collect()
    torch.cuda.empty_cache()
    app()
    gc.collect()
    torch.cuda.empty_cache()
    torch.distributed.destroy_process_group()
