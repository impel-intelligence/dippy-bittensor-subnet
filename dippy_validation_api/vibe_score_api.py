import os
from fastapi import FastAPI, HTTPException
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import gc
import ray
from vllm.distributed.parallel_state import destroy_model_parallel
from dippy_validation_api.dataset import PippaDataset
import huggingface_hub


# Import necessary modules and functions from the main API file
from dippy_validation_api.validation_api import (
    BATCH_SIZE_VIBE_SCORE,
    LENGTH_DIFF_PENALTY_STEEPNESS,
    MAX_GENERATION_LEEWAY,
    MAX_GENERATION_LENGTH,
    MAX_SEQ_LEN_VIBE_SCORE,
    SAMPLE_SIZE_VIBE_SCORE,
    EvaluateModelRequest,
    chat_template_mappings,
)

app = FastAPI()

if not os.path.exists("data"):
    os.makedirs("data")
# download the file pippa_deduped.jsonl from huggingface
if not os.path.exists("data/pippa_deduped.jsonl"):
    huggingface_hub.hf_hub_download(repo_id="PygmalionAI/PIPPA", filename="pippa_deduped.jsonl", repo_type="dataset", local_dir = "data")

vibe_score_dataset = PippaDataset("data/pippa_deduped.jsonl", max_input_len=MAX_SEQ_LEN_VIBE_SCORE - MAX_GENERATION_LENGTH - 200)

def calculate_vibe_match_score(model_name, revision, contexts, last_user_messages, expected_outputs, verbose=False):
    # instantiate a vllm model as it is faster and more memory efficient for text generation
    model = LLM(
        model_name,
        revision=revision,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.5,
        max_num_seqs=BATCH_SIZE_VIBE_SCORE,
        max_context_len_to_capture=MAX_SEQ_LEN_VIBE_SCORE,
    )
    decoded_messages = []
    # loop through the context in batches
    for i in range(0, len(contexts), BATCH_SIZE_VIBE_SCORE):
        max_user_message_len = max([len(message) for message in last_user_messages[i:i+BATCH_SIZE_VIBE_SCORE]])
        
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=min(int(max_user_message_len * (1 + MAX_GENERATION_LEEWAY)), MAX_GENERATION_LENGTH)
        )

        outputs = model.generate(
            prompts=contexts[i:i+BATCH_SIZE_VIBE_SCORE],
            sampling_params=sampling_params,
        )

        decoded_messages.extend([output.outputs[0].text for output in outputs])
        
    vibe_scores = []
    # calculate the vibe score
    for last_user_message, decoded in zip(last_user_messages, decoded_messages):
        last_user_message_len = len(last_user_message)
        decoded_len = len(decoded)
        length_difference = abs(decoded_len - last_user_message_len)
        decoded_len_score = 0 if last_user_message_len == 0 else torch.exp(-torch.tensor(length_difference) * LENGTH_DIFF_PENALTY_STEEPNESS / last_user_message_len).item()
        vibe_scores.append(decoded_len_score)
        if verbose:
            print("##############################################")
            if contexts:
                print(f"Context: {contexts[i]}")
            print(f"Last user message: {last_user_message}")
            print(f"Generated text: {decoded}")
            if expected_outputs:
                print(f"Expected output: {expected_outputs[i]}")

            print(f"Vibe score: {decoded_len_score}")
            print("##############################################")
        i += 1

    destroy_model_parallel()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    del model.llm_engine.model_executor
    del model
    gc.collect()
    torch.cuda.empty_cache()
    ray.shutdown()
    try:
        torch.distributed.destroy_process_group()
    except:
        print("No process group to destroy")
    
    return sum(vibe_scores) / len(vibe_scores)

@app.post("/vibe_match_score")
def get_vibe_match_score(request: EvaluateModelRequest):
    # The logic here will be similar to the main API's get_vibe_match function,
    # but it will be exposed as an API endpoint.
    # You can extract the relevant parts from the main API file and adjust as needed.
    # Extract the relevant parts from the main API's evaluate_model_logic function
    # and adjust as needed to only calculate the vibe score.
    try:
        input_tokenizer = AutoTokenizer.from_pretrained(
            f"{request.repo_namespace}/{request.repo_name}",
            revision=request.revision
        )
        # Set chat template params
        vibe_score_dataset.set_chat_template_params(chat_template_mappings[request.chat_template_type], input_tokenizer)

        # Unzip the sampled data
        vibe_contexts, vibe_target_texts, vibe_last_user_messages = zip(*vibe_score_dataset.sample_dataset(SAMPLE_SIZE_VIBE_SCORE))

        # Get the vibe score
        model_name = f"{request.repo_namespace}/{request.repo_name}"
        vibe_score = calculate_vibe_match_score(model_name, request.revision, vibe_contexts, vibe_last_user_messages, vibe_target_texts)

        return {"vibe_score": vibe_score}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/shutdown")
def shutdown():
    print("Shutting down eval_score_api")
    os._exit(0)
    

if __name__ == "__main__":
    # The multiprocessing setup and uvicorn server run command will be similar to the main API file.
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003, timeout_keep_alive=960)