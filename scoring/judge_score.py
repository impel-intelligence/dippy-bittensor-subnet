import os
import random
import json
import datetime
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from huggingface_hub import snapshot_download

from transformers import AutoTokenizer
from openai import OpenAI, AzureOpenAI
from typing import List, Optional
from scoring.common import (
    EvaluateModelRequest,
    DEFAULT_LORA_BASE,
    MAX_SEQ_LEN_COHERENCE_SCORE,
    MAX_GENERATION_LENGTH,
    chat_template_mappings,
    stringify_convo_from_messages,
    parse_json_safely,
)
from scoring.dataset import StreamedSyntheticPartialDataset
from pydantic import BaseModel
from typing import Literal

# Constants
JUDGE_NUM_EVALS = 512
JUDGE_MAX_TOKENS = 2048
JUDGE_EVAL_MODEL = "openai/gpt-4o-2024-11-20"
MAX_ERROR_RATE = 0.1


class JudgeScore(BaseModel):
    realism_win: Literal["original", "generated", "tie"]
    entertainment_win: Literal["original", "generated", "tie"]
    coherency_win: Literal["original", "generated", "tie"]
    realism_win_reasoning: str
    entertainment_win_reasoning: str
    coherency_win_reasoning: str


default_sample_params = SamplingParams(
    temperature=random.random() * 0.2,
    max_tokens=8192,
)
chat_params = {
    "gemma2": SamplingParams(temperature=random.random() * 0.2, max_tokens=8192, stop_token_ids=[107]),
    "mistral": SamplingParams(temperature=random.random() * 0.2, max_tokens=8192, stop_token_ids=[2]),
}
AZURE_KEY = os.environ.get("AZURE_KEY", "x")
AZURE_URL = os.environ.get("AZURE_URL", "x")
azure_client = AzureOpenAI(api_key=AZURE_KEY, azure_endpoint=AZURE_URL, api_version="2024-08-01-preview")
azure_model = "gpt-4o"
remote_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY", "x"),
)


"""
example conversation:
[
    {
        "role": "system",
        "content": "A chat between a user and Dirk Gently from 'Dirk Gently's Holistic Detective Agency'."
      },
      {
        "role": "user",
        "content": "Hey Dirk, how's the whole \"holistic detective\" thing going? lol"
      },
      {
        "role": "assistant",
        "content": "Ah, the life of a holistic detective is a wild and tangled journey!"
      }
]
"""


def get_llm_response(
    messages: List[dict],
    model_name: str,
    providers: List[str] = None,
    verbose: bool = False
) -> Optional[object]:
    if providers is None:
        providers = ["azure", "openrouter"]
    
    for provider in providers:
        try:
            if provider.lower() == "azure":
                if verbose:
                    print("Using Azure OpenAI client for judge scoring.")
                # Call Azure provider
                completion = azure_client.beta.chat.completions.parse(
                    model=azure_model,
                    messages=messages,
                    response_format=JudgeScore
                )
                if not completion.choices:
                    raise Exception("No response from Azure client.")
                return completion.choices[0].message.parsed
            elif provider.lower() in ("openrouter", "openai"):
                if verbose:
                    print("Using OpenRouter client for judge scoring.")
                # Call OpenRouter provider
                chat_completion = remote_client.chat.completions.create(
                    messages=messages,
                    model=model_name,
                    temperature=0,
                )
                return chat_completion.choices[0].message.content
            else:
                raise Exception(f"Unknown provider: {provider}")
        except Exception as e:
            if verbose:
                print(f"Error with provider {provider}: {e}")
            # Try next provider
    
    print("All providers failed for get_llm_response.")
    return None


def judge_evaluator(generated_conversation: List, original_conversation: List, verbose: bool = False, providers: List[str] = None):
    generated_stringified = stringify_convo_from_messages(generated_conversation, truncate=False)
    original_stringified = stringify_convo_from_messages(original_conversation, truncate=False)
    system_prompt = original_conversation[0]["content"]
    # compare them against each other
    evaluation_system_prompt = f'''
An assistant follows the provided instructions as mentioned:
=====
{system_prompt}
=====

Scrutinize the assistants's contributions, given there will be two points of comparison: the original text and the generated text.

Based on the given conversation and character guidelines, evaluate these criteria:
- realism: The responses match the character's personality perfectly, staying true to who they are meant to be without any inconsistencies
- entertainment: The responses are engaging and interesting to read, without any repetitive or boring language
- coherency: The responses use clear and correct language throughout, with proper grammar, spelling and punctuation. The writing style stays consistent and appropriate.

Ignore user messages.
Evaluate all assistant messages separately, not just the last one.
Explain your reasoning before giving scores.
Quote specific examples from the conversations using single quotes.
Focus on problems and mistakes.
Make sure scores match your explanations.
Do not judge politeness or aggression.
Practice extreme criticality and prejudice in judgement. Only designate the higher ranking scores in rare, exceptional cases.

Return data in JSON format with the following schema:
"""
{{
            "realism_win": (string : "original" or "generated" or "tie"),
            "entertainment_win": (string : "original" or "generated" or "tie"),
            "coherency_win": (string : "original" or "generated" or "tie"),
            "realism_win_reasoning": (string),
            "entertainment_win_reasoning": (string),
            "coherency_win_reasoning": (string)
}}
"""

Ensure JSON syntactical integrity! Escape embedded double quotes within string values as necessary.
'''

    user_message = f"""
Judge the following conversations to determine which conversation is better according to the provided criteria and respond ONLY in the JSON example provided. Respond with only the JSON object.

"original" :
{original_stringified}


( end of original text )

"generated" :
{generated_stringified}

( end of generated text )

Given the earlier instructions, please return data in JSON format with the following schema:

{{
            "realism_win": (string : "original" or "generated" or "tie"),
            "entertainment_win": (string : "original" or "generated" or "tie"),
            "coherency_win": (string : "original" or "generated" or "tie"),
            "realism_win_reasoning": (string),
            "entertainment_win_reasoning": (string),
            "coherency_win_reasoning": (string)
}}


"""

    messages = [
        {
            "role": "system",
            "content": evaluation_system_prompt,
        },
        {"role": "user", "content": user_message},
    ]
    # Replace direct client calls with the new wrapper function:
    if providers is None:
        providers = ["azure", "openrouter"]
    llm_output = get_llm_response(messages, JUDGE_EVAL_MODEL, providers=providers, verbose=verbose)
    try:
        if isinstance(llm_output, str):
            judge_score, original_output = parse_json_safely(llm_output)
        else:
            judge_score = {
                "realism_win": llm_output.realism_win,
                "entertainment_win": llm_output.entertainment_win,
                "coherency_win": llm_output.coherency_win,
                "realism_win_reasoning": llm_output.realism_win_reasoning,
                "entertainment_win_reasoning": llm_output.entertainment_win_reasoning,
                "coherency_win_reasoning": llm_output.coherency_win_reasoning,
            }
        if not judge_score:
            raise Exception(f"Judge score is empty or invalid given {original_output}")
        return judge_score
    except Exception as e:
        print(e)
        return None


def process_conversation(messages):
    """
    Convert a string message into a conversation format and ensure it ends with a user message.
    """
    # Convert string to conversation format
    conversation = [{"role": "user", "content": messages}]
    return conversation


def collect_judge_scores(scores: List):
    try:

        # Initialize tally structure
        tally = {
            "realism": {"original": 0, "generated": 0, "tie": 0},
            "entertainment": {"original": 0, "generated": 0, "tie": 0},
            "coherency": {"original": 0, "generated": 0, "tie": 0},
        }

        valid = 0

        # Process each score entry
        for item in scores:
            # Skip corrupted/incomplete data
            if not all(key in item for key in ["realism_win", "entertainment_win", "coherency_win"]):
                continue

            valid += 1

            # Tally wins for each category
            tally["realism"][item["realism_win"]] += 1
            tally["entertainment"][item["entertainment_win"]] += 1
            tally["coherency"][item["coherency_win"]] += 1
        # Calculate individual totals
        total_original = sum(cat["original"] for cat in tally.values())
        total_generated = sum(cat["generated"] for cat in tally.values())
        total_ties = sum(cat["tie"] for cat in tally.values())

        # Calculate win rate
        win_rate = (total_generated) / (valid * 3) if valid > 0 else 0

        # Combine into totals dict
        totals = {
            "total_original": total_original,
            "total_generated": total_generated,
            "total_ties": total_ties,
            "by_category": tally,
            "valid": valid,
            "win_rate": win_rate,
        }

        return totals
    except Exception as e:
        print(f"Error parsing file: {str(e)}")
        return None


def get_judge_score(request: EvaluateModelRequest, model: LLM, use_lora: bool = False, verbose=False, batch_size: int=8):
    try:
        start_time = datetime.datetime.now()

        repo_id = f"{request.repo_namespace}/{request.repo_name}"
        input_tokenizer = AutoTokenizer.from_pretrained(repo_id)
        print(f"Starting judge score evaluation at {start_time} given repo id {repo_id}")

        # Load synthetic dataset
        judge_dataset = StreamedSyntheticPartialDataset(cut_message_chain_early=1)
        judge_dataset.set_chat_template_params(chat_template_mappings[request.chat_template_type], input_tokenizer)
        print(f"loaded dataset with chat template {request.chat_template_type}")

        # Sample conversations
        conversations = judge_dataset.sample_dataset(JUDGE_NUM_EVALS, messages_limit=6)

        # Prepare batch prompts for vLLM
        formatted_messages = []
        original_messages_list = []
        last_assistant_responses = []

        for formatted_msg, orig_msgs, last_asst_resp in conversations:
            formatted_messages.append(formatted_msg)
            original_messages_list.append(orig_msgs)
            last_assistant_responses.append(last_asst_resp)

        # Generate all responses in batch
        print(f"Starting batch generation at {datetime.datetime.now()}")
        model_sampling_params = chat_params.get(request.chat_template_type, default_sample_params)
        print(f"Model sampling Params: {model_sampling_params}")
        print(f"Formatted messages: {len(formatted_messages)}")

        lora_request = None
        if use_lora:
            lora_id = f"{request.repo_namespace}/{request.repo_name}"
            lora_path = snapshot_download(repo_id=lora_id)
            lora_request = LoRARequest("lora_adapter", 1, lora_path)

        try:
            outputs = model.generate(
                formatted_messages, model_sampling_params, use_tqdm=False, lora_request=lora_request
            )
        except Exception as e:
            raise RuntimeError(f"Failed batch generation: {e}")

        # Process generations into complete conversations
        print(f"Starting conversation processing at {datetime.datetime.now()}")
        generated_conversations = []
        original_conversations = []

        for idx, output in enumerate(outputs):
            generated_text = output.outputs[0].text

            # Create complete conversations
            generated_conversation = original_messages_list[idx].copy()
            generated_conversation.append({"role": "assistant", "content": generated_text})
            generated_conversations.append(generated_conversation)

            original_conversation = original_messages_list[idx].copy()
            original_conversation.append({"role": "assistant", "content": last_assistant_responses[idx]})
            original_conversations.append(original_conversation)

        # Evaluate conversations in batches
        print(f"Completed text generation for {len(generated_conversations)} generated_conversations. Now starting judge evaluation at {datetime.datetime.now()}")
        all_judge_scores = []
        exceptions = 0
        
        # Process conversations in batches
        for batch_start in range(0, len(generated_conversations), batch_size):
            batch_end = min(batch_start + batch_size, len(generated_conversations))
            print(f"Processing batch {batch_start//batch_size + 1} of {(len(generated_conversations) + batch_size - 1)//batch_size} at {datetime.datetime.now()}")
            
            batch_results = []
            batch_exceptions = 0
            
            # Process each conversation in the current batch
            for idx in range(batch_start, batch_end):
                try:
                    judge_score = judge_evaluator(
                        generated_conversations[idx], original_conversations[idx], verbose=verbose
                    )
                    if judge_score is None:
                        raise Exception("could not parse judge score")
                    
                    batch_results.append(judge_score)
                    if verbose:
                        print(f"completed evaluation {idx + 1} of {len(generated_conversations)} with score {judge_score}")
                except Exception as e:
                    if verbose:
                        print(f"Error in judge evaluation for conversation {idx}: {e}")
                    batch_exceptions += 1
            
            # Add batch results to overall results
            all_judge_scores.extend(batch_results)
            exceptions += batch_exceptions
            
            if batch_end == len(generated_conversations) // 2:
                print(f"Reached halfway mark of judging at {datetime.datetime.now()}")

        if exceptions / JUDGE_NUM_EVALS > MAX_ERROR_RATE:
            raise RuntimeError(f"judge score failed with {exceptions} exceptions")

        judge_score = collect_judge_scores(all_judge_scores)
        if judge_score is None:
            raise RuntimeError("could not calculate judge score")

        if verbose:
            dump_conversations({"judge_score": judge_score, "generated_conversations": generated_conversations})

        end_time = datetime.datetime.now()
        print(f"Completed judge score evaluation at {end_time} with score {judge_score}")
        print(f"Total time elapsed: {end_time - start_time}")

        return {
            "judge_score": judge_score,
        }
    except Exception as e:
        if verbose:
            print(e)
        raise e


def dump_conversations(conversations, suffix: str = "x"):
    """
    Store conversations in a JSON file at /tmp/convos.json
    Args:
        conversations: List of conversation messages
    """
    try:
        with open(f"/tmp/judge_score_dump_{suffix}.json", "w") as f:
            json.dump(conversations, f, indent=2)
    except Exception as e:
        print(f"Error storing conversations: {e}")


class JudgeScore(BaseModel):
    realism_win: Literal["original", "generated", "tie"]
    entertainment_win: Literal["original", "generated", "tie"]
    coherency_win: Literal["original", "generated", "tie"]
    realism_win_reasoning: str
    entertainment_win_reasoning: str
    coherency_win_reasoning: str
