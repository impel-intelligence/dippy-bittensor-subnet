import os
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from transformers import AutoTokenizer


# Import necessary modules and functions from the main API file
from scoring.common import (
    EvaluateModelRequest,
    MAX_SEQ_LEN_COHERENCE_SCORE,
    MAX_GENERATION_LENGTH,
    chat_template_mappings,
    COHERENCE_MAX_TOKENS,
    COHERENCE_EVAL_MODEL,
    COHERENCE_NUM_EVALS,
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

    try:
        chat_completion = remote_client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": evaluation_text,
                }
            ],
            model=COHERENCE_EVAL_MODEL,
            temperature=0,
        )
        score = int(chat_completion.choices[0].message.content)
        return score
    except Exception as e:
        print(e)
        return 0


def get_coherence_score(request: EvaluateModelRequest, model: LLM):
    try:
        input_tokenizer = AutoTokenizer.from_pretrained(
            f"{request.repo_namespace}/{request.repo_name}", revision=request.revision
        )
        # Set chat template params
        coherence_dataset.set_chat_template_params(chat_template_mappings[request.chat_template_type], input_tokenizer)

        # Unzip the sampled data
        _, messages = zip(*coherence_dataset.sample_dataset(COHERENCE_NUM_EVALS))

        cscore = calculate_coherence_score(model, dataset_formatter=coherence_dataset, messages=messages)

        return {"coherence_score": cscore}
    except Exception as e:
        raise e


def pretty_convo(dict_list, score):
    print(f"score: {score}")
    print(f"convos: {len(dict_list)}")
    for i, item in enumerate(dict_list):
        print(f"Entry {i + 1}:")
        print(f"  Role: {item['role']}")
        print(f"  Content: {item['content']}")
        print()


def stringify_convo(dict_list):
    result = []
    for i, item in enumerate(dict_list):
        if item["role"] == "system":
            result.append(f"System Prompt: {item['content']}")
            continue
        result.append(f"Role: {item['role']}")
        result.append(f"Content: {item['content']}")
        result.append("")  # Add a blank line for spacing
    return "\n".join(result)


import random

MIN_CONVERSATIONS = 2
MAX_CONVERSATIONS = 4
MAX_ERROR_RATE = 0.1


def calculate_coherence_score(model: LLM, dataset_formatter, messages, verbose=False) -> int:
    generated_samples = []

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=COHERENCE_MAX_TOKENS,
    )
    # Initialize all conversations
    conversations = [message.copy() for message in messages]
    max_messages = [random.randint(MIN_CONVERSATIONS, MAX_CONVERSATIONS) for _ in messages]

    # Generate conversations in batches
    for turn in range(max(max_messages)):
        # Prepare batch of prompts
        batch_prompts = []
        active_conversations = []

        for i, (conversation, max_turn) in enumerate(zip(conversations, max_messages)):
            if turn < max_turn:
                new_input = dataset_formatter.new_input(conversation)

                batch_prompts.append(new_input)
                active_conversations.append(i)

        if not batch_prompts:
            break  # All conversations are complete

        # Generate responses for the batch
        outputs = model.generate(prompts=batch_prompts, sampling_params=sampling_params)

        # Update conversations with generated responses
        for i, output in zip(active_conversations, outputs):
            generated_text = output.outputs[0].text
            # print(f"generated_text: {generated_text}")
            role = "assistant" if conversations[i][-1]["role"] == "user" else "user"
            # Insert synthetic response
            conversations[i].append({"role": role, "content": generated_text})

    generated_samples = conversations

    evaluation_conversations = [stringify_convo(m) for m in generated_samples]
    penalty = 0
    exceptions = 0
    for i, convo in enumerate(evaluation_conversations):
        try:
            coherence_score = coherence_evaluator(convo)
            # pretty_convo(generated_samples[i], coherence_score)
            if coherence_score < 1:
                penalty += 1
        except Exception as e:
            exceptions += 1
            print(e)

    if exceptions / COHERENCE_NUM_EVALS > MAX_ERROR_RATE:
        raise RuntimeError("coherence failed due to api issues")

    ADJUSTED_EVALS = COHERENCE_NUM_EVALS - exceptions

    final_coherence_score = (ADJUSTED_EVALS - penalty) / ADJUSTED_EVALS

    return final_coherence_score
