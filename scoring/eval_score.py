
import gc
import os
from typing import Any
import tqdm
import torch
import math
from transformers import AutoTokenizer
from accelerate.utils import release_memory
import random
from scoring.common import (
    MAX_GENERATION_LENGTH,
    PROB_TOP_K,
    VOCAB_TRUNCATION,
    MAX_SEQ_LEN,
    CREATIVITY_SCALE_FACTOR,
    EvaluateModelRequest,
)

BATCH_SIZE = 2
max_entropy = math.log(VOCAB_TRUNCATION)


def cleanup(model, model_downloaded, request: EvaluateModelRequest):
    """
    Clean up the model data from memory and disk
    """
    # delete the model from memory
    with torch.no_grad():
        if model:
            release_memory(model)
            model = torch.Tensor([0])  # create a tensor to free up memory
            del model
            gc.collect()
            torch.cuda.empty_cache()
            try:
                torch.distributed.destroy_process_group()
            except:
                print("No process group to destroy")


def _prepare_dummy_inputs(model):
    max_model_len = min(model.config.max_position_embeddings, MAX_SEQ_LEN)
    input_ids = torch.randint(
        0,
        model.config.vocab_size,
        (BATCH_SIZE, max_model_len),
        requires_grad=False,
        dtype=torch.int64,
    )
    attention_mask = torch.ones_like(input_ids, requires_grad=False, dtype=torch.int64)
    return input_ids, attention_mask


def warmup_model(model):
    """
    Warm up the model by running it on a dummy input
    """
    # run the max sequence length input through the model with batch size BATCH_SIZE
    model.eval()
    latencies = []
    
    num_gpus = torch.cuda.device_count()
    for gpu_id in range(num_gpus):
        with torch.cuda.device(gpu_id):
            with torch.no_grad():
                for _ in range(10):
                    start_time = torch.cuda.Event(enable_timing=True)
                    end_time = torch.cuda.Event(enable_timing=True)
                    inputs = _prepare_dummy_inputs(model)
                    inputs = tuple(input.to(f"cuda:{gpu_id}") for input in inputs)
                    torch.cuda.synchronize(device=f"cuda:{gpu_id}")
                    start_time.record()
                    outputs = model(*inputs)
                    end_time.record()
                    # Waits for everything to finish running
                    torch.cuda.synchronize(device=f"cuda:{gpu_id}")

                    latency = start_time.elapsed_time(end_time)  # Measure latency in milliseconds
                    if torch.isnan(outputs.logits).any():
                        raise ValueError("NaN values detected in the logits tensor")

                    latencies.append(latency)
                    del outputs, inputs
                    gc.collect()
                    torch.cuda.empty_cache()

                average_latency = sum(latencies) / len(latencies)
                print(f"Average model inference latency over 10 runs: {average_latency} ms")
                return average_latency * 0.95

    # Test discount latency
    return average_latency * 0.95


def eval_score(
    model: Any,
    sampled_data: list[tuple],
    input_tokenizer: AutoTokenizer,
    output_tokenizer: AutoTokenizer,
    request: EvaluateModelRequest,
    debug: bool = False,
):
    """
    Evaluate the model on a dummy task
    """
    # maximum length this model can handle.
    max_len = min(model.config.max_position_embeddings, MAX_SEQ_LEN)

    if max_len is None:
        raise ValueError("Model does not have a maximum position embedding set")

    # unzip the sampled data
    sample_contexts, sample_target_texts, _ = zip(*sampled_data)

    total_prob = 0
    total_entropy = 0
    count = 0
    contexts = sample_contexts
    target_texts = sample_target_texts

    # now we want to calculate the average probability of the target tokens that model assigns.
    # batch_size = BATCH_SIZE
    batch_size = 1
    model.eval()
    with torch.no_grad():
        for i in tqdm.tqdm(range(0, len(contexts), batch_size), desc="Evaluating batches"):
            # Tokenize the inputs and labels

            # Pad the inputs and expected outputs to the same length in such a way that the
            # padding is on the left for inputs and on the right for outputs
            # this will ensure that the model see an intact context and the expected output is not shifted
            # example: [pad, pad, context, context, target, target, pad, pad]

            targets = output_tokenizer(
                target_texts[i : i + batch_size],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=MAX_GENERATION_LENGTH,
                add_special_tokens=False,
                # we don't want to add special tokens to the target as it continues from the context and already contains eos token.
            )  # this will put padding to the right and truncate if necessary

            inputs = input_tokenizer(
                contexts[i : i + batch_size],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_len - MAX_GENERATION_LENGTH,
                add_special_tokens=True,
            )  # this will put padding to the left and truncate the input if it is too long

            # concatenate the inputs and targets and their attention masks using torch.cat
            input_ids = torch.cat((inputs["input_ids"], targets["input_ids"]), dim=1)
            attention_mask = torch.cat((inputs["attention_mask"], targets["attention_mask"]), dim=1)

            if input_ids.shape[1] > max_len:
                print(
                    f"Input sequence length is greater than the maximum length the model can handle: {input_ids.shape[1]}"
                )
                raise ValueError("Input sequence length is greater than the maximum length the model can handle")

            # get the mask that only give us the output ids
            targets_ids_mask = torch.cat(
                [torch.zeros_like(inputs["attention_mask"]), targets["attention_mask"]],
                dim=1,
            )

            # shift the output mask to the right by one to get the corresponding predicted logits
            targets_ids_mask = torch.cat(
                [torch.zeros_like(targets_ids_mask[:, :1]), targets_ids_mask[:, :-1]],
                dim=1,
            )

            # Get model predictions (logits)
            try:
                print(
                    "Getting model predictions for sequence length: ",
                    input_ids.shape[1],
                    " batch size: ",
                    input_ids.shape[0],
                )
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,  # don't use cache as we are not generating text. To prevent bug for Mistral models
                )
            except Exception as e:
                print(
                    "Error getting model predictions for sequence length: ",
                    input_ids.shape[1],
                    " batch size: ",
                    input_ids.shape[0],
                )
                raise ValueError("Error getting model predictions: " + str(e))

            if torch.isnan(outputs.logits).any():
                raise ValueError("NaN values detected in the logits tensor")

            # shift the logits to the right by one to get the corresponding predicted logits
            outputs.logits = torch.cat(
                [torch.zeros_like(outputs.logits[:, :1, :]), outputs.logits[:, :-1, :]],
                dim=1,
            )

            if torch.isnan(outputs.logits).any():
                raise ValueError("NaN values detected llm -> outputs.logits tensor")

            # Only keep the top PROB_TOP_K scores by -inf the rest
            # This will make the model only consider the top 100 tokens and make sure the models with higher vocab sizes are not penalized

            # get the top k logits and mask out the rest
            top_k_logits, top_k_indices = outputs.logits.topk(VOCAB_TRUNCATION, dim=-1)
            outputs.logits = torch.full_like(outputs.logits, float("-inf")).scatter(-1, top_k_indices, top_k_logits)

            if debug:
                # print the input tokens and top 10 predicted tokens
                print(f"Input: {input_tokenizer.decode(input_ids[0], skip_special_tokens=True)}")
                for j in range(len(input_ids[0])):
                    if targets_ids_mask[0][j].item() == 1:
                        actual_id = input_ids[0][j].item()
                        actual_token = output_tokenizer.decode([actual_id])
                        top_10_predicted_ids = outputs.logits[0][j].topk(10).indices.tolist()
                        top_10_predicted_tokens = [output_tokenizer.decode([id]) for id in top_10_predicted_ids]
                        print(
                            f"Actual token: {actual_token}",
                            f" -> top 10 pred tokens: {top_10_predicted_tokens}",
                        )
            # entropy = -(probabilities * torch.log(probabilities + 1e-9)).sum(dim=-1)

            # normalize the logits to get probabilities
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-9), dim=-1)

            batch_entropy = (entropy * targets_ids_mask).sum() / targets_ids_mask.sum()
            normalized = batch_entropy.item() / max_entropy
            scaled_entropy = 1 - math.exp(-CREATIVITY_SCALE_FACTOR * normalized)
            total_entropy += scaled_entropy

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
            del (
                outputs,
                targets_ids_mask,
                probabilities,
                token_probabilities,
                one_gram_probabilities,
                two_gram_probabilities,
                three_gram_probabilities,
            )
            del (
                four_gram_probabilities,
                n_gram_prob,
                mask,
                top_prob_indices,
                top_k_logits,
                top_k_indices,
                inputs,
                targets,
                batch_entropy,
            )
            gc.collect()
            torch.cuda.empty_cache()

    average_prob = total_prob / count
    average_entropy = total_entropy / count
    print(f"Average probability of target tokens: {average_prob}")
    cleanup(model, True, request)

    return {
        "average_prob": average_prob,
        "average_entropy": average_entropy,
    }



def eval_score_batch(
    model: Any,
    sampled_data: list[tuple],
    input_tokenizer: AutoTokenizer,
    output_tokenizer: AutoTokenizer,
    request: EvaluateModelRequest,
    debug: bool = False,
):
    max_len = min(model.config.max_position_embeddings, MAX_SEQ_LEN)
    if max_len is None:
        raise ValueError("Model does not have a maximum position embedding set")

    sample_contexts, sample_target_texts, _ = zip(*sampled_data)

    total_prob = 0
    total_entropy = 0
    count = 0
    contexts = sample_contexts
    target_texts = sample_target_texts

    num_gpus = torch.cuda.device_count()
    batch_size = BATCH_SIZE * num_gpus  # Increase batch size proportionally to GPU count

    model.eval()
    with torch.no_grad():
        for i in tqdm.tqdm(range(0, len(contexts), batch_size), desc="Evaluating batches"):
            batch_contexts = contexts[i : i + batch_size]
            batch_targets = target_texts[i : i + batch_size]

            # Process each GPU's portion of the batch
            batch_results = []
            for gpu_id in range(num_gpus):
                start_idx = gpu_id * BATCH_SIZE
                end_idx = (gpu_id + 1) * BATCH_SIZE
                gpu_contexts = batch_contexts[start_idx:end_idx]
                gpu_targets = batch_targets[start_idx:end_idx]

                if not gpu_contexts:
                    continue  # Skip if this GPU has no data to process

                with torch.cuda.device(gpu_id):
                    inputs = input_tokenizer(
                        gpu_contexts,
                        return_tensors="pt",
                        padding="max_length",
                        truncation=True,
                        max_length=max_len - MAX_GENERATION_LENGTH,
                        add_special_tokens=True,
                    ).to(f"cuda:{gpu_id}")

                    targets = output_tokenizer(
                        gpu_targets,
                        return_tensors="pt",
                        padding="max_length",
                        truncation=True,
                        max_length=MAX_GENERATION_LENGTH,
                        add_special_tokens=False,
                    ).to(f"cuda:{gpu_id}")

                    input_ids = torch.cat((inputs["input_ids"], targets["input_ids"]), dim=1)
                    attention_mask = torch.cat((inputs["attention_mask"], targets["attention_mask"]), dim=1)

                    targets_ids_mask = torch.cat(
                        [torch.zeros_like(inputs["attention_mask"]), targets["attention_mask"]],
                        dim=1,
                    )
                    targets_ids_mask = torch.cat(
                        [torch.zeros_like(targets_ids_mask[:, :1]), targets_ids_mask[:, :-1]],
                        dim=1,
                    )

                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                    )

                    outputs.logits = torch.cat(
                        [torch.zeros_like(outputs.logits[:, :1, :]), outputs.logits[:, :-1, :]],
                        dim=1,
                    )

                    top_k_logits, top_k_indices = outputs.logits.topk(VOCAB_TRUNCATION, dim=-1)
                    outputs.logits = torch.full_like(outputs.logits, float("-inf")).scatter(
                        -1, top_k_indices, top_k_logits
                    )

                    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-9), dim=-1)

                    batch_entropy = (entropy * targets_ids_mask).sum() / targets_ids_mask.sum()
                    normalized = batch_entropy.item() / max_entropy
                    scaled_entropy = 1 - math.exp(-CREATIVITY_SCALE_FACTOR * normalized)

                    top_prob_indices = torch.topk(probabilities, PROB_TOP_K, dim=-1).indices
                    mask = torch.zeros_like(probabilities, dtype=torch.bool).scatter_(-1, top_prob_indices, True)
                    probabilities[~mask] = 1e-9
                    token_probabilities = probabilities.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)
                    token_probabilities = token_probabilities * targets_ids_mask

                    token_count = targets_ids_mask.sum().cpu().item()
                    one_gram_probabilities = token_probabilities
                    n_gram_prob = (one_gram_probabilities.sum().cpu().item() / token_count) * 0.25
                    two_gram_probabilities = one_gram_probabilities[:, 1:] * one_gram_probabilities[:, :-1]
                    n_gram_prob += (two_gram_probabilities.sum().cpu().item() / token_count) * 0.25
                    three_gram_probabilities = two_gram_probabilities[:, 1:] * one_gram_probabilities[:, :-2]
                    n_gram_prob += (three_gram_probabilities.sum().cpu().item() / token_count) * 0.25
                    four_gram_probabilities = three_gram_probabilities[:, 1:] * one_gram_probabilities[:, :-3]
                    n_gram_prob += (four_gram_probabilities.sum().cpu().item() / token_count) * 0.25

                    batch_results.append((n_gram_prob, scaled_entropy, token_count))

                    del (
                        outputs,
                        targets_ids_mask,
                        probabilities,
                        token_probabilities,
                        one_gram_probabilities,
                        two_gram_probabilities,
                        three_gram_probabilities,
                        four_gram_probabilities,
                    )
                    torch.cuda.empty_cache()

            # Aggregate results from all GPUs
            for n_gram_prob, scaled_entropy, token_count in batch_results:
                total_prob += n_gram_prob
                total_entropy += scaled_entropy
                count += 1

    average_prob = total_prob / count
    average_entropy = total_entropy / count
    print(f"Average probability of target tokens: {average_prob}")
    cleanup(model, True, request)
    return {
        "average_prob": average_prob,
        "average_entropy": average_entropy,
    }
