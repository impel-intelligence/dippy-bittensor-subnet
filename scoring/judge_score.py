import os
import random
import json
import datetime
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from huggingface_hub import snapshot_download

from transformers import AutoTokenizer
from openai import OpenAI
from typing import List
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

# Constants
JUDGE_NUM_EVALS = 128
JUDGE_MAX_TOKENS = 2048
JUDGE_EVAL_MODEL = "anthropic/claude-3.5-sonnet"
MAX_ERROR_RATE = 0.1


default_sample_params = SamplingParams(
    temperature=random.random() * 0.2,
    max_tokens=4096,
)
chat_params = {
    "gemma2": SamplingParams(temperature=random.random() * 0.2, max_tokens=4096, stop_token_ids=[107]),
}


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


def judge_evaluator(generated_conversation: List, original_conversation: List):
    generated_stringified = stringify_convo_from_messages(generated_conversation)
    original_stringified = stringify_convo_from_messages(original_conversation)
    system_prompt = original_conversation[0]["content"]
    # compare them against each other
    evaluation_system_prompt = f'''
An assistant follows the provided instructions as mentioned:
=====
{system_prompt}
=====

Scrutinize the participant's rhetorical contributions, given there will be two points for comparison: the original text and the generated text.

Given the documented discourse and character parameters, evaluate the subsequent criteria:
- verisimilitude: The participant's utterances demonstrate impeccable concordance with the prescribed persona, exhibiting fidelity to the established characterization without discordant elements.
- entertainment_score: The participant's locutions evince exceptional magnetism and engrossment, eschewing redundant phraseology.
- coherency: The participant's command of language exemplifies superlative proficiency, devoid of solecisms. Their articulation manifests consummate fluency, bereft of infelicitous constructions, with impeccable morphological accuracy, grammatical concord, orthographical precision, and punctuation. No lexical items diverge from the prescribed linguistic parameters.

Employ the 7 point Likert scale as exemplified below:
- 1 = Strongly Disagree
- 2 = Disagree
- 3 = Partially Disagree
- 4 = Neutral
- 5 = Partially Agree
- 6 = Agree
- 7 = Strongly Agree

Disregard utterances demarcated as "user".
Scrutinize all assistant messages, not merely the last message. Evaluate each conversational turn discretely.
Elucidate scores prior to their assignation.
Initiate elucidations with verbatim excerpts from participant contributions, employing single quotation marks. 
Emphasize deficiencies and improprieties.
Be sure to employ the Likert scoring method in concluding explication. 
Scores must demonstrate consonance with these elucidations.
Abstain from evaluating participant decorum or bellicosity.
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

"generated" :
{generated_stringified}
"""

    try:
        chat_completion = remote_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": evaluation_system_prompt,
                },
                {"role": "user", "content": user_message},
            ],
            model=JUDGE_EVAL_MODEL,
            temperature=0,
        )
        judge_score = parse_json_safely(chat_completion.choices[0].message.content)
        if not judge_score:
            raise Exception("Judge score returned empty dictionary")
        if not isinstance(judge_score, dict):
            raise Exception(f"Judge score returned invalid type: {type(judge_score)}")
        if not all(key in judge_score for key in ['realism_win', 'entertainment_win', 'coherency_win']):
            raise Exception(f"Judge score missing required keys. Got keys: {list(judge_score.keys())} {chat_completion.choices[0].message.content}")
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
            'realism': {'original': 0, 'generated': 0, 'tie': 0},
            'entertainment': {'original': 0, 'generated': 0, 'tie': 0},
            'coherency': {'original': 0, 'generated': 0, 'tie': 0}
        }
        
        valid = 0
        
        # Process each score entry
        for item in scores:
            # Skip corrupted/incomplete data
            if not all(key in item for key in ['realism_win', 'entertainment_win', 'coherency_win']):
                continue
                
            valid += 1
            
            # Tally wins for each category
            tally['realism'][item['realism_win']] += 1
            tally['entertainment'][item['entertainment_win']] += 1
            tally['coherency'][item['coherency_win']] += 1
        # Calculate individual totals
        total_original = sum(cat['original'] for cat in tally.values())
        total_generated = sum(cat['generated'] for cat in tally.values())
        total_ties = sum(cat['tie'] for cat in tally.values())
        
        # Calculate win rate
        win_rate = (total_generated) / (valid * 3) if valid > 0 else 0
        
        # Combine into totals dict
        totals = {
            'total_original': total_original,
            'total_generated': total_generated, 
            'total_ties': total_ties,
            'by_category': tally,
            'valid': valid,
            'win_rate': win_rate
        }
        
        return totals
    except Exception as e:
        print(f"Error parsing file: {str(e)}")
        return None


def get_judge_score(request: EvaluateModelRequest, model: LLM, use_lora: bool = False, verbose=False):
    try:
        start_time = datetime.datetime.now()
        print(f"Starting judge score evaluation at {start_time}")

        repo_id = f"{request.repo_namespace}/{request.repo_name}"
        input_tokenizer = AutoTokenizer.from_pretrained(repo_id)

        # Load synthetic dataset
        judge_dataset = StreamedSyntheticPartialDataset(
            max_input_len=MAX_SEQ_LEN_COHERENCE_SCORE - MAX_GENERATION_LENGTH - 200,
        )

        # Set chat template params
        judge_dataset.set_chat_template_params(chat_template_mappings[request.chat_template_type], input_tokenizer)
        print(f"loaded dataset with chat template {request.chat_template_type}")
        # Sample conversations
        conversations = judge_dataset.sample_dataset(JUDGE_NUM_EVALS)
        
        all_judge_scores = []
        exceptions = 0
        # Process conversations
        generated_conversations = []
        original_conversations = []  # Store original conversations

        lora_path = None
        if use_lora:
            lora_id = f"{request.repo_namespace}/{request.repo_name}"
            lora_path = snapshot_download(repo_id=lora_id)

        model_sampling_params: SamplingParams = chat_params.get(request.chat_template_type, default_sample_params)

        for idx, (formatted_message, original_messages, last_assistant_response) in enumerate(conversations):
            try:
                if idx == len(conversations) // 2:
                    print(f"Reached halfway mark at {datetime.datetime.now()}")
                
                if use_lora:
                    output = model.generate(
                        formatted_message,
                        model_sampling_params,
                        use_tqdm=False,
                        lora_request=LoRARequest("lora_adapter", 1, lora_path),
                    )
                else:
                    output = model.generate(formatted_message, model_sampling_params, use_tqdm=False)
                generated_text = output[0].outputs[0].text

                # Create complete conversation with generated response
                generated_conversation = original_messages.copy()
                generated_conversation.append({"role": "assistant", "content": generated_text})
                generated_conversations.append(generated_conversation)

                original_messages.append({"role": "assistant", "content": last_assistant_response})
                original_conversations.append(original_messages)
                judge_score = judge_evaluator(generated_conversation, original_messages)
                if judge_score is None:
                    raise Exception(f"could not parse judge score")
                all_judge_scores.append(judge_score)
                if verbose:
                    print(f"completed round of judging with score {judge_score}")
            except Exception as e:
                if verbose:
                    print(f"Error in judge evaluation: {e}")
                exceptions += 1

        if exceptions / JUDGE_NUM_EVALS > MAX_ERROR_RATE:
            raise RuntimeError(f"judge score failed with {exceptions} exceptions")
        judge_score = collect_judge_scores(all_judge_scores)
        if judge_score is None:
            raise RuntimeError(f"could not calculate judge score")
        x = {"judge_score": judge_score, "generated_conversations": generated_conversations}
        if verbose:
            dump_conversations(x)

        end_time = datetime.datetime.now()
        print(f"Completed judge score evaluation at {end_time}")
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
