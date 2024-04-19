from tqdm import tqdm
from typing import Any
import os
import torch
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer, BitsAndBytesConfig
import huggingface_hub
from fastapi import FastAPI, HTTPException
import shutil

from dataset import PippaDataset

# TODO: Add model validation queue to prevent multiple models being validated at the same time
# TODO: evict models from huggingface model cache otherwise hard disk will get full very quickly.
# TODO: Add a timeout to the evaluation API to prevent it from running for too long


MAX_MODEL_SIZE = 18*1024 # 18 GB to leave space for other operations
SAMPLE_SIZE = 100 # number of samples to evaluate the model from the dataset
BATCH_SIZE = 2 # batch size for evaluation
PROB_TOP_K = 50 # the correct token should be in the top 50 tokens, else a score of 0 is given to that token
# TODO: this will truncate the sequence to 8096 tokens. This is a temporary fix to make the evaluation faster.
MAX_SEQ_LEN = 4096 # maximum sequence length that should be allowed because eval gets really slow with longer sequences than this

device = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI()

# create dir data if not exists
if not os.path.exists("data"):
    os.makedirs("data")
# download the file pippa_deduped.jsonl from huggingface
if not os.path.exists("data/pippa_deduped.jsonl"):
    huggingface_hub.hf_hub_download(repo_id="PygmalionAI/PIPPA", filename="pippa_deduped.jsonl", repo_type="dataset", local_dir = "data")
dataset = PippaDataset("data/pippa_deduped.jsonl")

chat_template_mappings = {
    "vicuna": "prompt_templates/vicuna_prompt_template.jinja",
    # TODO: Add chatml, Mistral, and other chat templates
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
    
def get_eval_score(
        model: Any, 
        sampled_data: list[tuple], 
        input_tokenizer: AutoTokenizer, 
        output_tokenizer: AutoTokenizer,
        debug: bool = False
    ):
    """
    Evaluate the model on a dummy task
    """
    # maximum length this model can handle.
    max_len = min(model.config.max_position_embeddings, MAX_SEQ_LEN)
    
    if max_len is None:
        raise ValueError("Model does not have a maximum position embedding set")
    
    # unzip the sampled data
    contexts, target_texts = zip(*sampled_data)

    total_prob = 0
    token_count = 0

    # now we want to calculate the average probability of the target tokens that model assigns.
    batch_size = BATCH_SIZE
    model.eval()

    for i in tqdm(range(0, len(contexts), batch_size), desc="Evaluating batches"):
        with torch.no_grad():
            # Tokenize the inputs and labels

            # Pad the inputs and expected outputs to the same length in such a way that the 
            # padding is on the left for inputs and on the right for outputs
            # this will ensure that the model see an intact context and the expected output is not shifted
            # example: [pad, pad, context, context, target, target, pad, pad]
            
            targets = output_tokenizer(
                target_texts[i:i+batch_size], 
                return_tensors='pt', 
                padding=True,
                add_special_tokens=False # we don't want to add special tokens to the target as it continues from the context and already contains eos token.
            ) # this will put padding to the right

            # get the max length of the input by subtracting the length of the targets from the max length
            max_input_len = max_len - targets['input_ids'].shape[1]

            inputs = input_tokenizer(
                contexts[i:i+batch_size], 
                return_tensors='pt', 
                padding=True, 
                truncation=True, 
                max_length=max_input_len,
                add_special_tokens=True,
            ) # this will put padding to the left and truncate the input if it is too long

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
            try:
                 
                outputs = model(input_ids, attention_mask=attention_mask)
            except Exception as e:
                print("Error getting model predictions for sequence length: ", input_ids.shape[1], " batch size: ", input_ids.shape[0])
                raise ValueError("Error getting model predictions: " + str(e))

            # shift the logits to the right by one to get the corresponding predicted logits
            outputs.logits = torch.cat(
                [
                    torch.zeros_like(outputs.logits[:, :1, :]), 
                    outputs.logits[:, :-1, :]
                ], dim=1
            )

            if torch.isnan(outputs.logits).any():
                raise ValueError("NaN values detected llm -> outputs.logits tensor")

            # Only keep the top PROB_TOP_K scores by -inf the rest
            # This will make the model only consider the top 100 tokens and make sure the models with higher vocab sizes are not penalized

            # get the top k logits and mask out the rest
            top_k_logits, top_k_indices = outputs.logits.topk(PROB_TOP_K, dim=-1)
            outputs.logits = torch.full_like(outputs.logits, float('-inf')).scatter(-1, top_k_indices, top_k_logits)

            if debug:
                # print the input tokens and top 10 predicted tokens
                print(f"Input: {input_tokenizer.decode(input_ids[0])}")
                for i in range(len(input_ids[0])):
                    if targets_ids_mask[0][i].item() == 1:
                        actual_id = input_ids[0][i].item()
                        actual_token = output_tokenizer.decode([actual_id])
                        top_10_predicted_ids = outputs.logits[0][i].topk(10).indices.tolist()
                        top_10_predicted_tokens = [output_tokenizer.decode([id]) for id in top_10_predicted_ids]
                        print(f"Actual token: {actual_token}", f" -> top 10 pred tokens: {top_10_predicted_tokens}")

            
            # normalize the logits to get probabilities
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1).cuda()

            if torch.isnan(probabilities).any():
                raise ValueError("NaN values detected in the probabilities tensor")

            # Get the probabilities assigned by the model to the target tokens
            token_probabilities = probabilities.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)

            # Mask out non target tokens
            token_probabilities = token_probabilities * targets_ids_mask

            # Sum the probabilities for each batch and accumulate
            total_prob += token_probabilities.sum().cpu().item()
            token_count += targets_ids_mask.sum().cpu().item()

            torch.cuda.empty_cache()
    
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

    ##### TODO: This part should be made faster somehow ###### 
    # Calculating model size relies on too many assumptions, but is a good heuristic.
    if request.chat_template_type not in chat_template_mappings:
        raise HTTPException(status_code=400, detail="Chat template type not supported")

    model, reason = load_model_no_download(request.model_name)
    
    if not model:
        raise HTTPException(status_code=400, detail="Model is not valid: " + reason)
    
    print('The model is valid and now we can download the weights')

    # Part 1: Check model size and calculate model size score
    # get model size in MB, assuming model is loaded with 4 bit quantization
    model_size = model.num_parameters() * 4 / 8 / 1024 / 1024
    print('Model size: ', model_size, ' MB')
    print("Model number of parameters: ", model.num_parameters())
    # check if model size is within the limit. If not, return an error
    if model_size > MAX_MODEL_SIZE:
        raise HTTPException(status_code=400, detail="Model is too large when loaded in 4 bit quant: " + str(model_size) + " MB. Should be less than " + str(MAX_MODEL_SIZE/1024) + " MB")

    model_size_score = 1 - (model_size / MAX_MODEL_SIZE)
    print('Model size score: ', model_size_score)

    ###########################################################

    del model
    torch.cuda.empty_cache()
    
    # Now download the weights
    print('Downloading model weights')

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True, # This does not hurt performance much according to 
    )

    model_downloaded = False
    model = AutoModelForCausalLM.from_pretrained(
        request.model_name,
        device_map='auto' if device == "cuda" else 'cpu', # auto will leave space in the first GPU if multiple GPUs are available for other operations
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
        cache_dir = "data",
        force_download=True
    )
    print('Model weights downloaded successfully')
    model_downloaded = True

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
        input_tokenizer.pad_token_id = input_tokenizer.eos_token_id
        output_tokenizer.pad_token = output_tokenizer.eos_token # add a pad token if not present
        output_tokenizer.pad_token_id = output_tokenizer.eos_token_id
    
    dataset.set_chat_template_params(chat_template_mappings[request.chat_template_type], input_tokenizer)
    print('Tokenizer downloaded successfully')
    print('Sampling dataset')
    sampled_data = dataset.sample_dataset(SAMPLE_SIZE)
    
    # Part 2: Evaluate the model
    print('Evaluating model')
    try:
        eval_score = get_eval_score(
            model, 
            sampled_data, 
            input_tokenizer=input_tokenizer,
            output_tokenizer=output_tokenizer
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail="Error evaluating model: " + str(e))

    print('Model evaluation score: ', eval_score)

    # delete the model from memory
    del model
    torch.cuda.empty_cache()
    if model_downloaded:
        shutil.rmtree("data/models--" + request.model_name.split("/")[0] + "--"+request.model_name.split("/")[1])

    return {
        "model_size_score": model_size_score,
        "qualitative_score": eval_score
    }


if __name__ == "__main__":
    # import uvicorn
    # uvicorn.run(app, host="localhost", port=8000)
    # evaluate_model(EvaluateModelRequest(model_name="mistralai/Mistral-7B-Instruct-v0.1", chat_template_type="vicuna"))
    evaluate_model(EvaluateModelRequest(model_name="openai-community/gpt2", chat_template_type="vicuna"))