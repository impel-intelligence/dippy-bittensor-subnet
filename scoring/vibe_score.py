import os
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import gc
import ray
from vllm.distributed.parallel_state import destroy_model_parallel
from scoring.dataset import PippaDataset, StreamedSyntheticDataset

# Import necessary modules and functions from the main API file
from scoring.common import (
    BATCH_SIZE_VIBE_SCORE,
    DATASET_DIR,
    LENGTH_DIFF_PENALTY_STEEPNESS,
    MAX_GENERATION_LEEWAY,
    MAX_GENERATION_LENGTH,
    MAX_SEQ_LEN_VIBE_SCORE,
    SAMPLE_SIZE_VIBE_SCORE,
    EvaluateModelRequest,
    chat_template_mappings,
    VLLM_GPU_MEMORY,
)
from typing import List, Any


def calculate_vibe_match_score(
    model: LLM, 
    sampled_data: list[tuple],
    verbose: bool = False
):
    
    contexts, last_user_messages, expected_outputs = zip(*sampled_data)

    decoded_messages = []
    BATCH_SIZE_VIBE_SCORE = 8
    # loop through the context in batches
    for i in range(0, len(contexts), BATCH_SIZE_VIBE_SCORE):
        max_user_message_len = max([len(message) for message in last_user_messages[i : i + BATCH_SIZE_VIBE_SCORE]])

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=min(
                int(max_user_message_len * (1 + MAX_GENERATION_LEEWAY)),
                MAX_GENERATION_LENGTH,
            ),
        )

        outputs = model.generate(
            prompts=contexts[i : i + BATCH_SIZE_VIBE_SCORE],
            sampling_params=sampling_params,
        )

        decoded_messages.extend([output.outputs[0].text for output in outputs])

    vibe_scores = []
    # calculate the vibe score
    for last_user_message, decoded in zip(last_user_messages, decoded_messages):
        last_user_message_len = len(last_user_message)
        decoded_len = len(decoded)
        length_difference = abs(decoded_len - last_user_message_len)
        decoded_len_score = (
            0
            if last_user_message_len == 0
            else torch.exp(
                -torch.tensor(length_difference) * LENGTH_DIFF_PENALTY_STEEPNESS / last_user_message_len
            ).item()
        )
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

    return sum(vibe_scores) / len(vibe_scores)


def get_vibe_match_score(
    request: EvaluateModelRequest,
    model: LLM,
):
    try:
        input_tokenizer = AutoTokenizer.from_pretrained(
            f"{request.repo_namespace}/{request.repo_name}", revision=request.revision
        )
        
        vibe_score_dataset =  StreamedSyntheticDataset(
        max_input_len=MAX_SEQ_LEN_VIBE_SCORE - MAX_GENERATION_LENGTH - 200,
        )
        # Set chat template params
        vibe_score_dataset.set_chat_template_params(chat_template_mappings[request.chat_template_type], input_tokenizer)

        # Unzip the sampled data
        sampled_data = vibe_score_dataset.sample_dataset(SAMPLE_SIZE_VIBE_SCORE)
        

        vibe_score = calculate_vibe_match_score(
            model,
            vibe_data
        )

        return {"vibe_score": vibe_score}

    except Exception as e:
        raise e
