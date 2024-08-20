import os
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import gc
import ray
from vllm.distributed.parallel_state import destroy_model_parallel
import torch
import huggingface_hub
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM


# prompt dataset: ~256 prompts

# Import necessary modules and functions from the main API file
from scoring.common import (
    full_path,
    EvaluateModelRequest,
    MAX_SEQ_LEN_COHERENCE_SCORE,
    MAX_GENERATION_LENGTH,
    chat_template_mappings,
    SAMPLE_SIZE_VIBE_SCORE,
    COHERENCE_BATCH_SIZE,
    COHERENCE_MAX_TOKENS,
    COHERENCE_EVAL_MODEL,
    COHERENCE_NUM_EVALS,
    VLLM_GPU_MEMORY,
)
from scoring.dataset import PersonaHubDataset

coherence_dataset = PersonaHubDataset(
    max_input_len=MAX_SEQ_LEN_COHERENCE_SCORE - MAX_GENERATION_LENGTH - 200,
)

# TODO: Replace with corcel
from openai import OpenAI

remote_client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY", "x"),
)


def coherence_evaluator(generated_text: str):
    evaluation_text = f'''
    You are a text coherence analyzer.
    Your task is to assess the coherence of the following conversation.
    Coherent text should have logical flow, clear connections between ideas, and maintain a consistent theme or purpose throughout.
    Conversations should not have any repeating elements or abrupt ends.
    Respond only with:
    1 - if the text is coherent
    0 - if the text is not coherent

    Do not provide any explanation or additional output. Just respond with 1 or 0.

    Text to analyze:
    """
    {generated_text}
    """

    Coherence assessment (1 or 0):
    '''

    chat_completion = remote_client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": evaluation_text,
            }
        ],
        model=COHERENCE_EVAL_MODEL,
    )
    score = int(chat_completion.choices[0].message.content)
    return score


def get_coherence_score(request: EvaluateModelRequest):
    try:
        input_tokenizer = AutoTokenizer.from_pretrained(
            f"{request.repo_namespace}/{request.repo_name}", revision=request.revision
        )
        # Set chat template params
        coherence_dataset.set_chat_template_params(chat_template_mappings[request.chat_template_type], input_tokenizer)

        # Unzip the sampled data
        chat_contexts, messages = zip(*coherence_dataset.sample_dataset(COHERENCE_NUM_EVALS))

        model_name = f"{request.repo_namespace}/{request.repo_name}"

        cscore = calculate_coherence_score(
            model_name=model_name,
            revision=request.revision,
            chat_contexts=chat_contexts,
            dataset_formatter=coherence_dataset,
            messages=messages
        )

        return {"coherence_score": cscore}
    except Exception as e:
        raise e



def pretty_convo(dict_list):
    for i, item in enumerate(dict_list):
        print(f"Entry {i + 1}:")
        print(f"  Role: {item['role']}")
        print(f"  Content: {item['content']}")
        print()

def stringify_convo(dict_list):
    result = []
    for i, item in enumerate(dict_list):
        if item['role'] == "system":
            result.append(f"System Prompt: {item['content']}")
            continue
        result.append(f"Role: {item['role']}")
        result.append(f"Content: {item['content']}")
        result.append("")  # Add a blank line for spacing
    return "\n".join(result)


import random
MIN_CONVERSATIONS = 2
MAX_CONVERSATIONS = 5
def calculate_coherence_score(
        model_name,
        revision,
        chat_contexts,
        dataset_formatter,
        messages,
        verbose=False) -> int:
    # instantiate a vllm model as it is faster and more memory efficient for text generation
    model = LLM(
        model_name,
        revision=revision,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=VLLM_GPU_MEMORY,
        max_num_seqs=10,
        max_model_len=MAX_SEQ_LEN_COHERENCE_SCORE,
    )

    generated_samples = []

    to_evaluate = []
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=COHERENCE_MAX_TOKENS,
    )
    messages = messages[:COHERENCE_NUM_EVALS]
    # Create a conversation of n messages
    for message in messages:
        message_history = message.copy()
        max_messages = random.randint(MIN_CONVERSATIONS, MAX_CONVERSATIONS)
        for i in range(0, max_messages):
            new_input = dataset_formatter.new_input(message_history)
            new_output = model.generate(
            prompts=[new_input],
            sampling_params=sampling_params,
            )
            generated_text = new_output[0].outputs[0].text
            role = "assistant"
            if message_history[-1]['role'] == 'assistant':
                role = "user"
            message_history.append({
                "role": role,
                "content": generated_text
            })
        generated_samples.append(message_history)
    evaluation_conversations = []
    for m in generated_samples:
        # pretty_convo(m)
        evaluation_conversations.append(stringify_convo(m))

    coherence_score = 0
    penalty = 0
    for convo in evaluation_conversations:
        coherence_score = coherence_evaluator(convo)
        if coherence_score < 1:
            penalty += 1

    coherence_score = (COHERENCE_NUM_EVALS - penalty) / COHERENCE_NUM_EVALS

    destroy_model_parallel()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    del model.llm_engine.model_executor
    del model
    gc.collect()
    torch.cuda.empty_cache()
    ray.shutdown()
    try:
        torch.distributed.destroy_process_group()
    except Exception as e:
        print("No process group to destroy")

    return coherence_score