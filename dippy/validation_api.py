import torch
from fastapi import FastAPI, HTTPException
from transformers import AutoModelForCausalLM, AutoConfig
from typing import Any


MAX_MODEL_SIZE = 60000 # 60 GB
device = "cpu" # torch.device("cuda" if torch.cuda.is_available() else "cpu")
app = FastAPI()

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
    
def get_eval_score(model: Any):
    """
    Evaluate the model on a dummy task
    """
    # Blackbox function to evaluate the model
    # Replace with your actual evaluation code
    return 0.5

def evaluate_model(model_name: str):
    model, reason = load_model_no_download(model_name)
    if not model:
        raise HTTPException(status_code=400, detail="Model is not valid: " + reason)
    
    # get model size in MB
    model_size = sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024

    if model_size > MAX_MODEL_SIZE:
        raise HTTPException(status_code=400, detail="Model is too large: " + str(model_size) + " MB")
    
    # Now download the weights
    model = AutoModelForCausalLM.from_pretrained(model_name)

    eval_score = get_eval_score(model)







    
    
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)