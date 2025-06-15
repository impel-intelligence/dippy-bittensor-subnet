import typer
import torch
import json
from typing import Optional
import os
import sys
import gc
import traceback

from scoring.common import (
    EvaluateModelRequest,
)

app = typer.Typer()


def write_to_json(data: dict, filepath: str = "/tmp/output.json"):
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    typer.echo(f"Results written to {filepath} with content {data}")


def _run(
    request: EvaluateModelRequest,
    run_type: str,
):
    from scoring.inference_score import get_inference_score

    typer.echo(f"Evaluating with parameters: {request}")
    result = {"completed": False}
    try:
        if run_type == "inference":
            result = get_inference_score(request, use_lora=False)
        result["completed"] = True
        typer.echo(f"Evaluation complete. Result: {result}")
    except Exception as e:
        (
            exc_type,
            exc_value,
            exc_traceback,
        ) = sys.exc_info()  # Capture exception information
        traceback_details = traceback.format_exception(exc_type, exc_value, exc_traceback)
        result["error"] = f'{"".join(traceback_details)} {str(e)}'
    write_to_json(result, f"/tmp/{run_type}_output.json")


@app.command("inference")
def inference_score(
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
    _run(request=request, run_type="inference")


@app.command("stub")
def stub():
    write_to_json({"g": True}, "/tmp/output.json")


# example: python entrypoint.py python entrypoint.py eval repo_name repo_namespace chat_template_type hash
if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    app()
    gc.collect()

