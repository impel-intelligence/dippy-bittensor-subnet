from tqdm import tqdm
from typing import Any

import torch

from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer, BitsAndBytesConfig


from fastapi import FastAPI, HTTPException

from dippy.dataset import PippaDataset



MAX_MODEL_SIZE = 22500 # 60 GB
SAMPLE_SIZE = 1000
PROB_TOP_K = 50 # the correct token should be in the top 50 tokens
device = "cuda" if torch.cuda.is_available() else "cpu"
app = FastAPI()

dataset = PippaDataset("data/pippa_deduped.jsonl")

chat_template_mappings = {
    "vicuna": "prompt_templates/vicuna_prompt_template.jinja",
}


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
        print("Model configuration retrieved from Hub")
        try:
            # Check if GPU is available and see if model fits in GPU
            print('loading model in RAM to check if it fits in GPU')
            model = AutoModelForCausalLM.from_config(
                config=config,
            )
            print("Model loaded successfully")
            return model, None
        except Exception as e:
            return None, str(e)
    else:
        return None, "Could not retrieve model configuration from Hub"
    
def get_eval_score(model: Any, sampled_data: list[tuple], input_tokenizer: AutoTokenizer, output_tokenizer: AutoTokenizer):
    """
    Evaluate the model on a dummy task
    """
    max_len = model.config.max_position_embeddings
    
    if max_len is None:
        raise ValueError("Model does not have a maximum position embedding set")
    # unzip the sampled data
    contexts, target_texts = zip(*sampled_data)

    total_prob = 0
    token_count = 0

    # now we want the probabilities of the model on the sampled in batches of 16
    batch_size = 1 
    model.eval()

    for i in tqdm(range(0, len(contexts), batch_size), desc="Evaluating batches"):
        with torch.no_grad():
            # Tokenize the inputs and labels

            # TODO: Make sure the tokenizer has all the special tokens
            # Pad the inputs and expected outputs to the same length in such a way that the padding is on the left for inputs and on the right for outputs
            # this will ensure that the model see an intact context and the expected output is not shifted
            targets = output_tokenizer(target_texts[i:i+batch_size], return_tensors='pt', padding=True) # this will put padding to the right

            # get the max length of the input by subtracting the length of the targets from the max length
            max_input_len = max_len - targets['input_ids'].shape[1]

            inputs = input_tokenizer(contexts[i:i+batch_size], return_tensors='pt', padding=True, truncation=True, max_length=max_input_len) # this will put padding to the left and truncate the input if it is too long

            # concatenate the inputs and targets and their attention masks
            input_ids = torch.cat([inputs['input_ids'], targets['input_ids']], dim=1).to(device)
            attention_mask = torch.cat([inputs['attention_mask'], targets['attention_mask']], dim=1).to(device)
            
            # get the mask that only give us the output ids
            targets_ids_mask = torch.cat(
                [
                    torch.zeros_like(inputs['attention_mask']), 
                    targets['attention_mask']
                ], dim=1
            )

            # shift the output mask to the right by one to get the corresponding predicted logits
            targets_ids_mask = torch.cat(
                [
                    torch.zeros_like(targets_ids_mask[:, :1]), 
                    targets_ids_mask[:, :-1]
                ], dim=1
            ).to(device)

            # Get model predictions (logits)
            outputs = model(input_ids, attention_mask=attention_mask)
            
            # normalize the logits to get probabilities
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1).cuda()

            # Get the probabilities assigned by the model to the target tokens
            token_probabilities = probabilities.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)

            # Mask out non target tokens
            token_probabilities = token_probabilities * targets_ids_mask

            # Sum the probabilities for each batch and accumulate
            total_prob += token_probabilities.sum().item()
            token_count += targets_ids_mask.sum().item()
    # Calculate the average probability
    average_prob = total_prob / token_count if token_count > 0 else 0

    return average_prob

from pydantic import BaseModel

class EvaluateModelRequest(BaseModel):
    model_name: str
    chat_template_type: str

@app.post("/evaluate_model")
def evaluate_model(request: EvaluateModelRequest):
    """
    Evaluate a model based on the model size and the quality of the model.
    """
    if request.chat_template_type not in chat_template_mappings:
        raise HTTPException(status_code=400, detail="Chat template type not supported")

    model, reason = load_model_no_download(request.model_name)
    
    if not model:
        raise HTTPException(status_code=400, detail="Model is not valid: " + reason)
    
    print('The model is valid and now we can download the weights')
    
    # Part 1: Check model size and calculate model size score
    # get model size in MB, assuming model is loaded with 4 bit quantization
    model_size = model.num_parameters() * 4 / 8 / 1024 / 1024
    
    if model_size > MAX_MODEL_SIZE:
        raise HTTPException(status_code=400, detail="Model is too large: " + str(model_size) + " MB")

    model_size_score = 1 - (model_size / MAX_MODEL_SIZE)
    print('Model size score: ', model_size_score)
    
    # Now download the weights
    print('Downloading model weights')

    quant_config = BitsAndBytesConfig(load_in_4bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        request.model_name,
        device_map='auto' if device == "cuda" else 'cpu', # auto will leave space in the first GPU if multiple GPUs are available for other operations
        quantization_config=quant_config,
    )
    print('Model weights downloaded successfully')

    # get the tokenizers
    print('Downloading tokenizer')
    input_tokenizer = AutoTokenizer.from_pretrained(
        request.model_name,
        padding_side='left',
    )
    output_tokenizer = AutoTokenizer.from_pretrained(
        request.model_name,
        padding_side='right',
    )
    if input_tokenizer.pad_token is None:
        input_tokenizer.pad_token = input_tokenizer.eos_token # add a pad token if not present
        output_tokenizer.pad_token = output_tokenizer.eos_token # add a pad token if not present
    
    dataset.set_chat_template_params(chat_template_mappings[request.chat_template_type], input_tokenizer)
    print('Tokenizer downloaded successfully')
    print('Sampling dataset')
    sampled_data = dataset.sample_dataset(SAMPLE_SIZE)
    
    # Part 2: Evaluate the model
    print('Evaluating model')
    eval_score = get_eval_score(
        model, 
        sampled_data, 
        input_tokenizer=input_tokenizer,
        output_tokenizer=output_tokenizer
    )

    print('Model evaluation score: ', eval_score)

    # delete the model from memory
    del model
    torch.cuda.empty_cache()

    return {
        "model_size_score": model_size_score,
        "qualitative_score": eval_score
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)

