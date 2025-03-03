"""
DEPRECATED
"""


import os
import random
import datetime
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from transformers import AutoTokenizer
from typing import List

# Import AzureOpenAI for using Azure OpenAI client similar to judge_score
from openai import OpenAI, AzureOpenAI

# Import necessary modules and functions from the main API file
from scoring.common import (
    stringify_convo_from_messages,
    EvaluateModelRequest,
    MAX_SEQ_LEN_COHERENCE_SCORE,
    MAX_GENERATION_LENGTH,
    chat_template_mappings,
    COHERENCE_MAX_TOKENS,
    COHERENCE_EVAL_MODEL,
    COHERENCE_NUM_EVALS,
)
from scoring.dataset import PersonaHubDataset

# Define Azure configuration (similar to judge_score)
AZURE_KEY = os.environ.get("AZURE_KEY", "x")
AZURE_URL = os.environ.get("AZURE_URL", "x")
backup_client = AzureOpenAI(api_key=AZURE_KEY, azure_endpoint=AZURE_URL, api_version="2024-08-01-preview")

# Default client for OpenRouter usage
remote_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY", "x"),
)

def strip_surrounding_quotes(text: str) -> str:
    """Remove surrounding single or double quotes from text if present."""
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        return text[1:-1]
    return text


coherence_dataset = PersonaHubDataset(
    max_input_len=MAX_SEQ_LEN_COHERENCE_SCORE - MAX_GENERATION_LENGTH - 200,
)

def coherence_evaluator(generated_text: str, providers: List[str] = None, verbose: bool = False):
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
    messages = [
        {
            "role": "user",
            "content": evaluation_text,
        }
    ]
    try:
        # Use the new helper to get the completion using provider cycling
        response = get_coherence_completion(messages, COHERENCE_EVAL_MODEL, providers=providers, verbose=verbose)
        if response is None:
            return 0
        score = int(response)
        return score
    except Exception as e:
        print(e)
        return 0

def generate_user_response(messages) -> str:
    generate_user_prompt = f'''
    You are a human having a conversation. 
    Generate a natural, casual response that a typical user might give in this conversation. 
    The response should be brief (1-3 sentences) and conversational in tone.
    In the case that the current conversation is incoherent or contains repeating phrases, please explicitly mark that this conversation is incoherent.


    Here is the conversation history:
    """
    {messages}
    """

    Generate a user response that naturally continues this conversation (and please explicitly mark that this conversation is incoherent if applicable):
    '''
    try:
        chat_completion = remote_client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": generate_user_prompt,
                }
            ],
            model=COHERENCE_EVAL_MODEL,
            temperature=0.5,
        )
        content = str(chat_completion.choices[0].message.content)
        return content
    except Exception as e:
        return ""

def get_coherence_score(request: EvaluateModelRequest, model: LLM, verbose=False) -> dict:
    try:
        repo_id = f"{request.repo_namespace}/{request.repo_name}"
        input_tokenizer = AutoTokenizer.from_pretrained(repo_id)
        # Set chat template params
        coherence_dataset.set_chat_template_params(chat_template_mappings[request.chat_template_type], input_tokenizer)

        # Unzip the sampled data
        _, messages = zip(*coherence_dataset.sample_dataset(COHERENCE_NUM_EVALS))

        cscore = calculate_coherence_score(model, dataset_formatter=coherence_dataset, messages=messages)

        return {"coherence_score": cscore}
    except Exception as e:
        if verbose:
            print(e)
        raise e


def pretty_convo(dict_list, score, verbose: bool = False) -> str:
    output = []
    output.append(f"score: {score}")
    output.append(f"convos: {len(dict_list)}")
    for i, item in enumerate(dict_list):
        output.append(f"Entry {i + 1}:")
        output.append(f"  Role: {item['role']}")
        output.append(f"  Content: {item['content']}")
        output.append("")

    result = "\n".join(output)
    if verbose:
        print(result)
    return result


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


MIN_CONVERSATIONS = 2
MAX_CONVERSATIONS = 4
MAX_ERROR_RATE = 0.1

# Add a helper function to toggle between Azure and OpenRouter clients
def get_coherence_completion(messages, model_name, providers: List[str] = None, verbose: bool = False):
    if providers is None:
        providers = ["azure", "openrouter"]
    
    for provider in providers:
        try:
            if provider.lower() == "azure":
                if verbose:
                    print(f"Using Azure OpenAI client for coherence evaluation with model {model_name}")
                # Using the Azure client
                chat_completion = backup_client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    temperature=0,
                )
                content = chat_completion.choices[0].message.content
                return content
            elif provider.lower() in ("openrouter", "openai"):
                if verbose:
                    print(f"Using OpenRouter client for coherence evaluation with model {model_name}")
                # Using the OpenRouter client
                chat_completion = remote_client.chat.completions.create(
                    messages=messages,
                    model="openai/gpt-4o-mini-2024-07-18",
                    temperature=0,
                )
                content = chat_completion.choices[0].message.content
                return content
            else:
                raise Exception(f"Unknown provider: {provider}")
        except Exception as e:
            if verbose:
                print(f"Error with provider {provider}: {e}")
            # Try next provider
    
    print("All providers failed for get_coherence_completion.")
    return None

def calculate_coherence_score(model: LLM, dataset_formatter, messages, verbose=False) -> float:
    start_time = datetime.datetime.now()
    print(f"Starting coherence score calculation at {start_time}")

    generated_samples = []

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=COHERENCE_MAX_TOKENS,
    )
    # Initialize all conversations
    conversations = [message.copy() for message in messages]

    max_messages = [random.randint(MIN_CONVERSATIONS, MAX_CONVERSATIONS) for _ in messages]

    # Generate conversations in batches
    total_turns = max(max_messages)
    for turn in range(total_turns):
        if turn == total_turns // 2:
            print(f"Reached halfway mark of conversation generation at {datetime.datetime.now()}")

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
        outputs = model.generate(
            prompts=batch_prompts,
            sampling_params=sampling_params,
            use_tqdm=False,
        )

        # Update conversations with generated responses
        for i, output in zip(active_conversations, outputs):
            generated_text = output.outputs[0].text
            generated_text = strip_surrounding_quotes(generated_text)
            conversations[i].append({"role": "assistant", "content": generated_text})
            user_response = generate_user_response(conversations[i])
            user_response = strip_surrounding_quotes(user_response)
            conversations[i].append({"role": "user", "content": user_response})

    generated_samples = conversations

    print(f"Starting coherence evaluation at {datetime.datetime.now()}")

    evaluation_conversations = [stringify_convo_from_messages(m, truncate=False) for m in generated_samples]
    scored_convos = []
    penalty = 0
    exceptions = 0

    for i, convo in enumerate(evaluation_conversations):
        if i == len(evaluation_conversations) // 2:
            print(f"Reached halfway mark of coherence evaluation at {datetime.datetime.now()}")
        try:
            coherence_score = coherence_evaluator(convo)
            scored_convos.append(pretty_convo(generated_samples[i], coherence_score))
            if coherence_score < 1:
                penalty += 1
        except Exception as e:
            exceptions += 1
            print(e)

    with open("coherence_evaluation_conversations.txt", "w", encoding="utf-8") as f:
        for convo in scored_convos:
            f.write(convo)
            f.write("\n\n")

    if exceptions / COHERENCE_NUM_EVALS > MAX_ERROR_RATE:
        raise RuntimeError(f"coherence failed due to {exceptions} api issues")

    ADJUSTED_EVALS = COHERENCE_NUM_EVALS - exceptions

    final_coherence_score = (ADJUSTED_EVALS - penalty) / ADJUSTED_EVALS

    end_time = datetime.datetime.now()
    print(f"Completed coherence score calculation at {end_time} with score {final_coherence_score}")
    print(f"Total time elapsed: {end_time - start_time}")

    return final_coherence_score
