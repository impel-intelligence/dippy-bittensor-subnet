import gc
import os
from typing import Any
import tqdm

import torch
import huggingface_hub
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
from accelerate.utils import release_memory
from accelerate import PartialState
from typing import List

# Import necessary modules and functions from the main API file
from model_evaluation.common import (
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
from model_evaluation.dataset import DolphinDataset


# coherence_dataset = DolphinDataset(
#     "data/flan5m-alpaca-uncensored.jsonl",
# )


def get_coherence_prompts():
    return []


def get_coherence_score(request: EvaluateModelRequest):
    try:
        # input_tokenizer = AutoTokenizer.from_pretrained(
        #     f"{request.repo_namespace}/{request.repo_name}", revision=request.revision
        # )
        # # Set chat template params
        # coherence_dataset.set_chat_template_params(
        #     chat_template_mappings[request.chat_template_type], input_tokenizer
        # )
        #
        # # Unzip the sampled data
        # vibe_contexts, vibe_target_texts, vibe_last_user_messages = zip(
        #     *coherence_dataset.sample_dataset(SAMPLE_SIZE_VIBE_SCORE)
        # )
        #
        # # Get the vibe score
        # model_name = f"{request.repo_namespace}/{request.repo_name}"
        # vibe_score = coherence_score(
        #     model_name,
        #     request.revision,
        #     vibe_contexts,
        #     vibe_last_user_messages,
        # )

        return {"coherence_score": -1}

    except Exception as e:
        raise e


def coherence_score(
    model: Any,
    input_tokenizer: AutoTokenizer,
    output_tokenizer: AutoTokenizer,
) -> float:
    # maximum length this model can handle.
    # max_length = min(model.config.max_position_embeddings, MAX_SEQ_LEN)
    #
    # if max_length is None:
    #     raise ValueError("Model does not have a maximum position embedding set")
    #
    # batch_size = 1
    # model.eval()
    # coherence_scores = [1]
    # with torch.no_grad():
    #     for prompt in tqdm.tqdm([], desc="Evaluating coherence"):
    #         # Generate text
    #         inputs = input_tokenizer(
    #             prompt,
    #             return_tensors="pt",
    #             padding=True,
    #             truncation=True,
    #             max_length=max_length,
    #         )
    #         inputs = {k: v.to(model.device) for k, v in inputs.items()}
    #
    #         outputs = model.generate(
    #             **inputs, max_new_tokens=max_length, num_return_sequences=1
    #         )
    #
    #         generated_text = output_tokenizer.decode(
    #             outputs[0], skip_special_tokens=True
    #         )
    #         print(generated_text)
    #         # Split the generated text into segments
    #         words = generated_text.split()
    #         segment_length = min(
    #             30, len(words) // 4
    #         )  # Adjust segment length based on text length
    #         segments = [
    #             " ".join(words[i : i + segment_length])
    #             for i in range(0, len(words), segment_length)
    #         ]
    #
    #         # Calculate coherence score for the generated text
    #         segment_embeddings = []
    #         for i in range(0, len(segments), batch_size):
    #             batch = segments[i : i + batch_size]
    #             inputs = input_tokenizer(
    #                 batch,
    #                 return_tensors="pt",
    #                 padding=True,
    #                 truncation=True,
    #                 max_length=max_length,
    #             )
    #             inputs = {k: v.to(model.device) for k, v in inputs.items()}
    #
    #             outputs = model(**inputs, output_hidden_states=True)
    #
    #             # Use the average of the last hidden state as the embedding
    #             embeddings = outputs.hidden_states[-1].mean(dim=1)
    #             segment_embeddings.extend(embeddings.to(torch.float32).cpu().numpy())
    #
    #         # Calculate cosine similarity between segment embeddings
    #         similarities = []
    #         for i in range(len(segment_embeddings)):
    #             for j in range(i + 1, len(segment_embeddings)):
    #                 similarity = torch.nn.functional.cosine_similarity(
    #                     torch.tensor(
    #                         segment_embeddings[i], dtype=torch.float32
    #                     ).unsqueeze(0),
    #                     torch.tensor(
    #                         segment_embeddings[j], dtype=torch.float32
    #                     ).unsqueeze(0),
    #                 )
    #                 similarities.append(similarity.item())
    #
    #     # Coherence score is the average similarity
    #     coherence_score = sum(similarities) / len(similarities) if similarities else 0
    #     coherence_scores.append(coherence_score)
    #
    #     # Clear GPU memory
    #     del outputs, inputs, embeddings
    #     torch.cuda.empty_cache()
    #     gc.collect()
    #
    # # Overall coherence score is the average across all prompts
    # overall_coherence = sum(coherence_scores) / len(coherence_scores)
    # print(f"Overall coherence score: {overall_coherence:.4f}")

    return -1
