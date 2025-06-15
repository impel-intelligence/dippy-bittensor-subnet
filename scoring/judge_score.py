import os
import random
import json
import datetime
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from huggingface_hub import snapshot_download,hf_hub_download
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
from scoring.dataset import HuggingfaceDataset
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

token_id_mappings = {
    "gemma2": [107],
    "mistral": [2],
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
                completion = azure_client.beta.chat.completions.parse(
                    model=azure_model,
                    messages=messages,
                    response_format=JudgeScore
                )
                if not completion.choices:
                    raise Exception("No response from Azure client.")
                return completion.choices[0].message.parsed
            elif provider.lower() in ("openrouter", "openai"):
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


def load_model_configuration(repo_id: str):
    configuration = None
    chat_template = None

    try:
        config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
        with open(config_path, "r") as f:
            configuration = json.load(f)
    except Exception as e:
        print(f"Could not download or parse config.json for {repo_id}: {e}")

    try:
        template_path = hf_hub_download(repo_id=repo_id, filename="chat_template.json")
        with open(template_path, "r") as f:
            chat_template = json.load(f)
    except Exception as e:
        print(f"Could not download or parse chat_template.json for {repo_id}: {e}")

    return configuration, chat_template

def generate_model_responses(request: EvaluateModelRequest, model: LLM, use_lora: bool = False, verbose: bool = False):
    """
    Generate model responses for a set of conversations using the specified model and configuration.
    
    Args:
        request: The evaluation request containing model configuration
        model: The LLM instance to use for generation
        use_lora: Whether to use LoRA adaptation
        verbose: Whether to print detailed logs
        
    Returns:
        tuple: (generated_conversations, original_conversations)
    """
    repo_id = f"{request.repo_namespace}/{request.repo_name}"
    input_tokenizer = AutoTokenizer.from_pretrained(repo_id)
    
    
    judge_dataset = HuggingfaceDataset(max_messages=6)
    configuration, custom_chat_template = load_model_configuration(repo_id)
    chat_template = chat_template_mappings[request.chat_template_type]

    judge_dataset.set_chat_template_params(chat_template_mappings[request.chat_template_type], input_tokenizer)
    if custom_chat_template is not None:
        judge_dataset.set_chat_template_params_from_str(custom_chat_template["chat_template"], input_tokenizer)
        chat_template = custom_chat_template["chat_template"]
    
    stop_token_ids = token_id_mappings.get(request.chat_template_type, [])
    if configuration is not None:
        stop_token_ids = configuration.get("eos_token_id", stop_token_ids)
        if not isinstance(stop_token_ids, list):
            stop_token_ids = [stop_token_ids]
    model_sampling_params = default_sample_params
    if len(stop_token_ids) > 0:
        model_sampling_params = SamplingParams(
            temperature=random.random() * 0.2,
            max_tokens=8192,
            stop_token_ids=stop_token_ids,
        )

    print(f"Starting dataset sampling at {datetime.datetime.now()}")

    judge_dataset = judge_dataset.process()
    conversations = judge_dataset.sample_dataset(JUDGE_NUM_EVALS, 12)

    print(f"Completed dataset sampling at {datetime.datetime.now()}")

    formatted_messages = []
    original_messages_lists = []
    last_assistant_responses = []

    for formatted_msg, orig_msgs, last_asst_resp in conversations:
        formatted_messages.append(formatted_msg)
        original_messages_lists.append(orig_msgs)
        last_assistant_responses.append(last_asst_resp)
    if verbose:
        print(f"Starting response generation in batches with repo id {repo_id} and configuration {configuration}")
        print(f"start_time {datetime.datetime.now()}")
        print(f"Model sampling Params: {model_sampling_params}")
        print(f"Model chat template: {chat_template}")
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
    if verbose:
        print(f"Starting conversation processing at {datetime.datetime.now()}")
    print(f"Starting conversation processing at {datetime.datetime.now()}")
    generated_conversations = []
    original_conversations = []

    for idx, output in enumerate(outputs):
        generated_text = output.outputs[0].text

        # Create complete conversations
        generated_conversation = original_messages_lists[idx].copy()
        generated_conversation.append({"role": "assistant", "content": generated_text})
        generated_conversations.append(generated_conversation)

        original_conversation = original_messages_lists[idx].copy()
        original_conversation.append({"role": "assistant", "content": last_assistant_responses[idx]})
        original_conversations.append(original_conversation)

    return generated_conversations, original_conversations


def evaluate_conversations(generated_conversations: List, original_conversations: List, verbose: bool = False):
    """
    Evaluate pairs of conversations using the judge evaluator.
    
    Args:
        generated_conversations: List of generated conversation sequences
        original_conversations: List of original conversation sequences
        verbose: Whether to print detailed logs
        
    Returns:
        dict: Aggregated judge scores
    """
    if verbose:
        print(f"Starting judge evaluation at {datetime.datetime.now()}")
    
    all_judge_scores = []
    exceptions = 0
    
    # Process conversations in batches
    judge_evaluator_batch_size = 32
    for batch_start in range(0, len(generated_conversations), judge_evaluator_batch_size):
        batch_end = min(batch_start + judge_evaluator_batch_size, len(generated_conversations))
        if verbose:
            print(f"Processing batch {batch_start//judge_evaluator_batch_size + 1} of {(len(generated_conversations) + judge_evaluator_batch_size - 1)//judge_evaluator_batch_size}")
        
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
                if verbose and random.random() < 1/128:
                    print(f"completed evaluation {idx + 1} of {len(generated_conversations)} with score {judge_score}")
            except Exception as e:
                if verbose:
                    print(f"Error in judge evaluation for conversation {idx}: {e}")
                batch_exceptions += 1
        
        # Add batch results to overall results
        all_judge_scores.extend(batch_results)
        exceptions += batch_exceptions
        
        if batch_end == len(generated_conversations) // 2 and verbose:
            print(f"Reached halfway mark of judging at {datetime.datetime.now()}")

    if exceptions / JUDGE_NUM_EVALS > MAX_ERROR_RATE:
        raise RuntimeError(f"judge score failed with {exceptions} exceptions")

    judge_score = collect_judge_scores(all_judge_scores)
    if judge_score is None:
        raise RuntimeError("could not calculate judge score")

    return judge_score


def get_judge_score(request: EvaluateModelRequest, model: LLM, use_lora: bool = False, verbose=False, batch_size: int=8):
    """
    Get judge scores by generating and evaluating model responses.
    """
    try:
        start_time = datetime.datetime.now()
        if verbose:
            print(f"Starting judge score evaluation at {start_time}")

        # Step 1: Generate model responses
        generated_conversations, original_conversations = generate_model_responses(
            request=request,
            model=model,
            use_lora=use_lora,
            verbose=verbose
        )

        # Save conversations immediately after generation for safety
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        conversations_data = {
            "generated_conversations": generated_conversations,
            "original_conversations": original_conversations,
            "metadata": {
                "timestamp": timestamp,
                "hash": request.hash
            }
        }
        dump_conversations(conversations_data, f"raw_conversations_{timestamp}")
        if verbose:
            print(f"Saved raw conversations to /tmp/judge_score_dump_raw_conversations_{timestamp}.json")

        # Step 2: Evaluate the conversations
        judge_score = evaluate_conversations(
            generated_conversations=generated_conversations,
            original_conversations=original_conversations,
            verbose=verbose
        )

        if verbose:
            dump_conversations({"judge_score": judge_score, "generated_conversations": generated_conversations})

        end_time = datetime.datetime.now()
        if verbose:
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
