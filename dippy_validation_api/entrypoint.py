import typer
import json
from typing import Optional
import os
import huggingface_hub
from model_evaluation.common import EvaluateModelRequest

app = typer.Typer()

def _dl_dataset():
    # create dir data if not exists
    if not os.path.exists("./data"):
        os.makedirs("./data")
    # download the file pippa_deduped.jsonl from huggingface
    if not os.path.exists("data/pippa_deduped.jsonl"):
        huggingface_hub.hf_hub_download(
            repo_id="PygmalionAI/PIPPA",
            filename="pippa_deduped.jsonl",
            repo_type="dataset",
            local_dir="data",
        )


def write_to_json(data: dict, filepath: str = "/tmp/output.json"):
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    typer.echo(f"Results written to {filepath}")


@app.command("eval")
def evaluate(
    repo_name: str = typer.Argument(help="Repository name"),
    repo_namespace: str = typer.Argument(help="Repository namespace"),
    chat_template_type: str = typer.Argument(help="Chat template type"),
    hash: Optional[str] = typer.Argument(help="hash"),
):
    from model_evaluation.eval_score import get_eval_score
    request = EvaluateModelRequest(
        repo_namespace=repo_namespace,
        repo_name=repo_name,
        chat_template_type=chat_template_type,
        hash=hash,
    )
    typer.echo(f"Evaluating with parameters: {request}")
    result = {"completed": False}
    try:
        result = get_eval_score(request)
        result["completed"] = True
        typer.echo(f"Evaluated with parameters: {result}")
    except Exception as e:
        result["error"] = e
    write_to_json(result, "/tmp/eval_output.json")


@app.command("coherence")
def coherence(
    repo_name: str = typer.Argument(help="Repository name"),
    repo_namespace: str = typer.Argument(help="Repository namespace"),
    chat_template_type: str = typer.Argument(help="Chat template type"),
    hash: Optional[str] = typer.Argument(help="hash"),
):
    from model_evaluation.coherence_score import get_coherence_score
    request = EvaluateModelRequest(
        repo_namespace=repo_namespace,
        repo_name=repo_name,
        chat_template_type=chat_template_type,
        hash=hash,
    )
    typer.echo(f"Evaluating coherence with parameters: {request}")
    result = {"completed": False}
    try:
        result = get_coherence_score(request)
        result["completed"] = True
        typer.echo(f"Evaluated coherence with parameters: {result}")
    except Exception as e:
        result["error"] = e
    write_to_json(result, "/tmp/coherence_output.json")


@app.command("vibe")
def vibe_score(
    repo_name: str = typer.Argument(help="Repository name"),
    repo_namespace: str = typer.Argument(help="Repository namespace"),
    chat_template_type: str = typer.Argument(help="Chat template type"),
    hash: Optional[str] = typer.Argument(help="hash"),
):
    from model_evaluation.vibe_score import get_vibe_match_score
    request = EvaluateModelRequest(
        repo_namespace=repo_namespace,
        repo_name=repo_name,
        chat_template_type=chat_template_type,
        hash=hash,
    )
    typer.echo(f"Evaluating with parameters: {request}")
    result = {"completed": False}
    try:
        result = get_vibe_match_score(request)
        result["completed"] = True
        typer.echo(f"Evaluated with parameters: {result}")
    except Exception as e:
        result["error"] = e
    write_to_json(result, "/tmp/vibe_output.json")


@app.command("stub")
def stub():
    print("stub")
    result = {"g": True}
    write_to_json(result, "/tmp/output.json")


if __name__ == "__main__":
    _dl_dataset()
    app()
