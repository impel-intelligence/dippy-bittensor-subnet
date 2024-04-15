from dippy.dataset import PippaDataset
import torch
import torch.nn.functional as F

from fastapi import FastAPI, HTTPException
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from typing import Any


MAX_MODEL_SIZE = 60000 # 60 GB
SAMPLE_SIZE = 1000
PROB_TOP_K = 50 # the correct token should be in the top 50 tokens
device = "cpu" # torch.device("cuda" if torch.cuda.is_available() else "cpu")
app = FastAPI()

dataset = PippaDataset("data/pippa_deduped.json")


def load_model_no_download(model_name: Any):
    """
    Validate the model by loading it, without downloading it from the Hugging Face Hub
    """
    # Blackbox function to validate the model's metadata
    # Replace with your actual criteria
    try:
        config = AutoConfig.from_pretrained(model_name)
    except Exception as e:
        return None, str(e)
    
    if config is not None:
        try:
            # Check if GPUs are available and set the model to use all available GPUs
            if device == "cuda" and torch.cuda.is_available():
                model = AutoModelForCausalLM.from_config(config=config, device_map='auto') # device_map='auto' will use all available GPUs
            else:
                model = AutoModelForCausalLM.from_config(config=config)
                model.to(torch.device("cpu"))
            
            return model, None
        except Exception as e:
            return None, str(e)
    else:
        return None, "Could not retrieve model configuration from Hub"
    
def get_eval_score(model: Any, sampled_data: list[tuple] = None, tokenizer: Any = None):
    """
    Evaluate the model on a dummy task
    """
    # unzip the sampled data
    contexts, responses = zip(*sampled_data)

    total_prob = 0
    token_count = 0

    # now we want the probabilities of the model on the sampled in batches of 16
    batch_size = 16
    for i in range(0, len(contexts), batch_size):
        with torch.no_grad():
            # Tokenize the inputs and labels
            inputs = tokenizer(contexts[i:i+batch_size], return_tensors="pt", padding=True, truncation=True)
            labels = tokenizer(responses[i:i+batch_size], return_tensors="pt", padding=True, truncation=True).input_ids

            # Move tensors to the correct device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)

            # Get model predictions (logits)
            outputs = model(**inputs, labels=labels)
            logits = outputs.logits

            # Shift the logits to align with the labels
            shift_logits = logits[..., :-1, :].contiguous()

            # Apply softmax to convert logits to probabilities
            softmax_probs = F.softmax(shift_logits, dim=-1)

            # 0 out all probabilities that are not in top k. Thus, the correct token should be in the top k, else a score of 0 is given
            top_k = torch.topk(softmax_probs, k=PROB_TOP_K, dim=-1)
            softmax_probs = torch.zeros_like(softmax_probs).scatter(dim=-1, index=top_k.indices, src=top_k.values)

            # Gather the probabilities of the correct tokens
            target_probs = softmax_probs.gather(dim=-1, index=labels[..., 1:].unsqueeze(-1)).squeeze(-1)

            # Sum the probabilities for each batch and accumulate
            total_prob += target_probs.sum().item()
            token_count += labels[..., 1:].numel()

    # Calculate the average probability
    average_prob = total_prob / token_count if token_count > 0 else 0

    return average_prob


def evaluate_model(model_name: str):
    model, reason = load_model_no_download(model_name)
    if not model:
        raise HTTPException(status_code=400, detail="Model is not valid: " + reason)
    
    # Part 1: Check model size and calculate model size score
    # get model size in MB
    model_size = sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024

    if model_size > MAX_MODEL_SIZE:
        raise HTTPException(status_code=400, detail="Model is too large: " + str(model_size) + " MB")

    model_size_score = 1 - (model_size / MAX_MODEL_SIZE)
    
    # Now download the weights
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # get the tokenizers
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset.set_chat_template_params("data/vicuna_prompt_template.jinja", tokenizer)
    sampled_data = dataset.sample_data(SAMPLE_SIZE)
    # Part 2: Evaluate the model
    eval_score = get_eval_score(model, sampled_data)

    return {
        "model_size_score": model_size_score,
        "qualitative_score": eval_score
    }


@app.get("/validate_model")
def validate_model(model_name: str):
    return evaluate_model(model_name)
 





    
    
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)