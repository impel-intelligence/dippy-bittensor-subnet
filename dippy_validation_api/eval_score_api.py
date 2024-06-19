import gc
import os
from typing import Any
import tqdm

from fastapi import FastAPI, HTTPException
import torch
import huggingface_hub
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
from accelerate.utils import release_memory
from accelerate import PartialState

# Import necessary modules and functions from the main API file
from dippy_validation_api.validation_api import (
    MAX_AVG_LATENCY,
    MAX_GENERATION_LENGTH,
    MAX_MODEL_SIZE,
    PROB_TOP_K,
    SAMPLE_SIZE,
    VOCAB_TRUNCATION,
    MAX_SEQ_LEN,
    BATCH_SIZE,
    EvaluateModelRequest,
)

from dippy_validation_api.dataset import PippaDataset
from dippy_validation_api.validation_api import chat_template_mappings

app = FastAPI()

# create dir data if not exists
if not os.path.exists("data"):
    os.makedirs("data")
# download the file pippa_deduped.jsonl from huggingface
if not os.path.exists("data/pippa_deduped.jsonl"):
    huggingface_hub.hf_hub_download(repo_id="PygmalionAI/PIPPA", filename="pippa_deduped.jsonl", repo_type="dataset", local_dir = "data")

dataset = PippaDataset("data/pippa_deduped.jsonl", max_input_len=MAX_SEQ_LEN - MAX_GENERATION_LENGTH - 200)


def get_eval_score(
        model: Any, 
        sampled_data: list[tuple], 
        input_tokenizer: AutoTokenizer, 
        output_tokenizer: AutoTokenizer,
        request: EvaluateModelRequest,
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
    contexts, target_texts, _ = zip(*sampled_data)
    total_prob = 0
    count = 0

    # now we want to calculate the average probability of the target tokens that model assigns.
    batch_size = BATCH_SIZE
    model.eval()
    with torch.no_grad():
        for i in tqdm.tqdm(range(0, len(contexts), batch_size), desc="Evaluating batches"):
            # Tokenize the inputs and labels

            # Pad the inputs and expected outputs to the same length in such a way that the 
            # padding is on the left for inputs and on the right for outputs
            # this will ensure that the model see an intact context and the expected output is not shifted
            # example: [pad, pad, context, context, target, target, pad, pad]
            
            targets = output_tokenizer(
                target_texts[i:i+batch_size], 
                return_tensors='pt', 
                padding='max_length',
                truncation=True,
                max_length=MAX_GENERATION_LENGTH,
                add_special_tokens=False # we don't want to add special tokens to the target as it continues from the context and already contains eos token.
            ) # this will put padding to the right and truncate if necessary

            inputs = input_tokenizer(
                contexts[i:i+batch_size], 
                return_tensors='pt', 
                padding='max_length',
                truncation=True, 
                max_length=max_len - MAX_GENERATION_LENGTH,
                add_special_tokens=True,
            ) # this will put padding to the left and truncate the input if it is too long

            # concatenate the inputs and targets and their attention masks using torch.cat
            input_ids = torch.cat((inputs['input_ids'], targets['input_ids']), dim=1).to('cuda')
            attention_mask = torch.cat((inputs['attention_mask'], targets['attention_mask']), dim=1).to('cuda')

            if input_ids.shape[1] > max_len:
                print(f"Input sequence length is greater than the maximum length the model can handle: {input_ids.shape[1]}")
                raise ValueError("Input sequence length is greater than the maximum length the model can handle")

            
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
            ).to('cuda')

            # Get model predictions (logits)
            try:
                print("Getting model predictions for sequence length: ", input_ids.shape[1], " batch size: ", input_ids.shape[0])
                outputs = model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask,
                    use_cache=False, # don't use cache as we are not generating text. To prevent bug for Mistral models
                )
            except Exception as e:
                print("Error getting model predictions for sequence length: ", input_ids.shape[1], " batch size: ", input_ids.shape[0])
                raise ValueError("Error getting model predictions: " + str(e))
            
            if torch.isnan(outputs.logits).any():
                raise ValueError("NaN values detected in the logits tensor")

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
            top_k_logits, top_k_indices = outputs.logits.topk(VOCAB_TRUNCATION, dim=-1)
            outputs.logits = torch.full_like(outputs.logits, float('-inf')).scatter(-1, top_k_indices, top_k_logits)

            if debug:
                # print the input tokens and top 10 predicted tokens
                print(f"Input: {input_tokenizer.decode(input_ids[0])}")
                for j in range(len(input_ids[0])):
                    if targets_ids_mask[0][j].item() == 1:
                        actual_id = input_ids[0][j].item()
                        actual_token = output_tokenizer.decode([actual_id])
                        top_10_predicted_ids = outputs.logits[0][j].topk(10).indices.tolist()
                        top_10_predicted_tokens = [output_tokenizer.decode([id]) for id in top_10_predicted_ids]
                        print(f"Actual token: {actual_token}", f" -> top 10 pred tokens: {top_10_predicted_tokens}")

            # normalize the logits to get probabilities
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1).cuda()

            if torch.isnan(probabilities).any():
                raise ValueError("NaN values detected in the probabilities tensor")
            
            # Get the top PROB_TOP_K indices and zero out all other probabilities
            top_prob_indices = torch.topk(probabilities, PROB_TOP_K, dim=-1).indices
            mask = torch.zeros_like(probabilities, dtype=torch.bool).scatter_(-1, top_prob_indices, True)
            probabilities[~mask] = 1e-9
            # Get the probabilities assigned by the model to the target tokens
            token_probabilities = probabilities.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)

            # Mask out non target tokens
            token_probabilities = token_probabilities * targets_ids_mask

            # get the 1, 2, 3, 4 gram probabilities
            token_count = targets_ids_mask.sum().cpu().item()
            # 1-gram
            one_gram_probabilities = token_probabilities
            n_gram_prob = (one_gram_probabilities.sum().cpu().item() / token_count) * 0.25
            # 2-gram
            two_gram_probabilities = one_gram_probabilities[:, 1:] * one_gram_probabilities[:, :-1]
            n_gram_prob += (two_gram_probabilities.sum().cpu().item() / token_count) * 0.25
            # 3-gram
            three_gram_probabilities = two_gram_probabilities[:, 1:] * one_gram_probabilities[:, :-2]
            n_gram_prob += (three_gram_probabilities.sum().cpu().item() / token_count) * 0.25
            # 4-gram
            four_gram_probabilities = three_gram_probabilities[:, 1:] * one_gram_probabilities[:, :-3]
            n_gram_prob += (four_gram_probabilities.sum().cpu().item() / token_count) * 0.25
            
            total_prob += n_gram_prob
            count += 1
            
            # delete the tensors to free up memory
            del outputs, targets_ids_mask, probabilities, token_probabilities, one_gram_probabilities, two_gram_probabilities, three_gram_probabilities
            del four_gram_probabilities, n_gram_prob, mask, top_prob_indices, top_k_logits, top_k_indices, inputs, targets
            gc.collect()
            torch.cuda.empty_cache()
    
    average_prob = total_prob / count
    print(f"Average probability of target tokens: {average_prob}")
    cleanup(model, True, request)

    return average_prob


def _prepare_dummy_inputs(model, device='cuda'):
    max_model_len = min(model.config.max_position_embeddings, MAX_SEQ_LEN)
    input_ids = torch.randint(0, model.config.vocab_size, (BATCH_SIZE, max_model_len), requires_grad=False, dtype=torch.int64, device=device)
    attention_mask = torch.ones_like(input_ids, requires_grad=False, dtype=torch.int64, device=device)
    return input_ids, attention_mask

def warmup_model(model):
    """
    Warm up the model by running it on a dummy input
    """
    # run the max sequence length input through the model with batch size BATCH_SIZE
    model.eval()
    latencies = []
    with torch.no_grad():
        for _ in range(10):
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            inputs = _prepare_dummy_inputs(model, device='cuda')
            start_time.record()
            outputs = model(*inputs)
            end_time.record()
            # Waits for everything to finish running
            torch.cuda.synchronize()

            latency = start_time.elapsed_time(end_time)  # Measure latency in milliseconds
            if torch.isnan(outputs.logits).any():
                raise ValueError("NaN values detected in the logits tensor")

            latencies.append(latency)
            del outputs, inputs
            gc.collect()
            torch.cuda.empty_cache()

        average_latency = sum(latencies) / len(latencies)
        print(f"Average model inference latency over 10 runs: {average_latency} ms")

    # Test discount latency
    return average_latency * 0.95

def cleanup(model, model_downloaded, request: EvaluateModelRequest):
    """
    Clean up the model data from memory and disk
    """
    # delete the model from memory
    with torch.no_grad():
        if model:
            release_memory(model)
            model = torch.Tensor([0]) # create a tensor to free up memory
            del model
            gc.collect()
            torch.cuda.empty_cache()
            try:
                torch.distributed.destroy_process_group()
            except:
                print("No process group to destroy")

@app.post("/eval_score")
def eval_score(request: EvaluateModelRequest):
    # Now download the weights
    print('Downloading model weights')
    model_downloaded = False
    failure_reason = ""
    # make dir data/hash if not exist
    if not os.path.exists(f"data/{str(request.hash)}"):
        os.makedirs(f"data/{str(request.hash)}")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True, # This does not hurt performance much according to 
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(
            f"{request.repo_namespace}/{request.repo_name}",
            revision=request.revision,
            quantization_config=quant_config,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            device_map='auto',
            # cache_dir=f"data/{str(request.hash)}",
            # force_download=True
        )

    except Exception as e:
        try:
            print(f"Error loading model in 4 bit quant with flash attention.: {e}. Trying vanilla load. This might cause OOM.")
            model = AutoModelForCausalLM.from_pretrained(
                f"{request.repo_namespace}/{request.repo_name}",
                revision=request.revision,
                device_map='auto',
                # cache_dir = f"data/{str(request.hash)}",
                # force_download=True
            )
        except Exception as e:
            raise HTTPException(417, f"Error loading model: {str(e)}")
        
    model.eval()

    # get the tokenizers
    print('Downloading tokenizer')
    try:
        input_tokenizer = AutoTokenizer.from_pretrained(
            f"{request.repo_namespace}/{request.repo_name}",
            padding_side='left',
            force_download=True
        )
        output_tokenizer = AutoTokenizer.from_pretrained(
            f"{request.repo_namespace}/{request.repo_name}",
            padding_side='right',
            force_download=True,
        )
        if input_tokenizer.pad_token is None:
            input_tokenizer.pad_token = input_tokenizer.eos_token # add a pad token if not present
            input_tokenizer.pad_token_id = input_tokenizer.eos_token_id
            output_tokenizer.pad_token = output_tokenizer.eos_token # add a pad token if not present
            output_tokenizer.pad_token_id = output_tokenizer.eos_token_id
        
        print('Tokenizer downloaded successfully')
    except Exception as e:
        failure_reason = str(e)
        cleanup(model, model_downloaded, request)
        raise HTTPException(417, "Error downloading tokenizer: " + failure_reason)

    # warm up the model
    print('Warming up model')
    try:
        avg_latency = warmup_model(model)
        if not avg_latency: # either 0 or None
            raise HTTPException(417, "Error warming up model")
            
    except Exception as e:
        failure_reason = str(e)
        cleanup(model, model_downloaded, request)
        raise HTTPException(417, "Error warming up model: " + failure_reason)
    

    # get latency score
    latency_score = 1 - (avg_latency / MAX_AVG_LATENCY)
    print('Latency score: ', latency_score)

    # get model size score
    try:
        print('Model weights downloaded successfully')
        model_size = model.get_memory_footprint()
        print('Model size: ', model_size, ' Bytes')
        print("Model number of parameters: ", model.num_parameters())
        # check if model size is within the limit. If not, return an error
        if model_size > MAX_MODEL_SIZE:
            del model
            torch.cuda.empty_cache()
            raise HTTPException(412, f"Model is too large when loaded in 4 bit quant: {model_size} Bytes. Should be less than {MAX_MODEL_SIZE} Bytes")
        
        model_size_score = 1 - (model_size / MAX_MODEL_SIZE)
        print('Model size score: ', model_size_score)
        model_downloaded = True
    except Exception as e:
        failure_reason = str(e)
        cleanup(None, model_downloaded, request)
        raise HTTPException(412, "Error loading model: " + failure_reason)

    # set the chat template params
    dataset.set_chat_template_params(chat_template_mappings[request.chat_template_type], input_tokenizer)

    print('Sampling dataset')
    try:
        sampled_data = dataset.sample_dataset(SAMPLE_SIZE)
    except Exception as e:
        failure_reason = str(e)
        cleanup(model, model_downloaded, request)
        raise HTTPException(500, "Error sampling dataset: " + failure_reason)
    
    # Part 2: Evaluate the model
    print('Evaluating model')
    try:
        eval_score = get_eval_score(
            model, 
            sampled_data, 
            input_tokenizer=input_tokenizer,
            output_tokenizer=output_tokenizer,
            request=request,
        )
        print('Model evaluation score: ', eval_score)
    except Exception as e:
        failure_reason = str(e)
        cleanup(model, model_downloaded, request)
        raise HTTPException(500, "Error evaluating model: " + failure_reason)

    return {
        "eval_score": eval_score,
        "latency_score": latency_score,
        "model_size_score": model_size_score,
    }

@app.post("/shutdown")
def shutdown():
    print("Shutting down eval_score_api")
    os._exit(0)


if __name__ == "__main__":
    # get command line argument for port 
    import sys
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    else:
        port = 8001
    import uvicorn
    # launch the api only if main process
    if PartialState().is_main_process:
        uvicorn.run(app, host="0.0.0.0", port=port, timeout_keep_alive=960)
