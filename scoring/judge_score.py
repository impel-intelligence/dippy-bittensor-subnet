import os
import random
import json
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
    parse_json_safely
)
from scoring.dataset import StreamedSyntheticPartialDataset

# Constants
JUDGE_NUM_EVALS = 5
JUDGE_MAX_TOKENS = 1000
JUDGE_EVAL_MODEL = "anthropic/claude-3.5-sonnet"
MAX_ERROR_RATE = 0.1







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
    system_prompt = original_conversation[0]['content']
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
    "scores": [
        {{
            "verisimilitude_win": (string : "original" or "generated"),
            "entertainment_win": (string : "original" or "generated"),
            "coherency_win": (string : "original" or "generated"),
            "verisimilitude_win_reasoning": (string),
            "entertainment_win_reasoning": (string),
            "coherency_win_reasoning": (string)
        }}
    ]
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
                {
                    "role": "user",
                    "content": user_message
                },
            ],
            model=JUDGE_EVAL_MODEL,
            temperature=0,
        )
        win_score = parse_json_safely(chat_completion.choices[0].message.content)

        return {
            "win_score": win_score,
        }



    except Exception as e:
        print(e)
        return None

def process_conversation(messages):
    """
    Convert a string message into a conversation format and ensure it ends with a user message.
    """
    # Convert string to conversation format
    conversation = [
        {"role": "user", "content": messages}
    ]
    return conversation

def get_judge_score(request: EvaluateModelRequest, model: LLM,use_lora:bool=False, verbose=False):
    try:
        repo_id = f"{request.repo_namespace}/{request.repo_name}"
        input_tokenizer = AutoTokenizer.from_pretrained(repo_id)

        # Load synthetic dataset
        judge_dataset = StreamedSyntheticPartialDataset(
            max_input_len=MAX_SEQ_LEN_COHERENCE_SCORE - MAX_GENERATION_LENGTH - 200,
        )
        
        # Set chat template params
        judge_dataset.set_chat_template_params(
            chat_template_mappings[request.chat_template_type], 
            input_tokenizer
        )

        # Sample conversations
        # conversations = judge_dataset.sample_dataset(JUDGE_NUM_EVALS)
        # conversations = judge_dataset.sample_dataset(128)
        conversations = judge_dataset.sample_dataset(1)
        scores = []
        exceptions = 0
        # Process conversations
        generated_conversations = []
        original_conversations = []  # Store original conversations

        lora_path = None
        if use_lora:
            # lora_id = "mindw96/Gemma-2-27b-it-LoRA-dacon-llm2"
            lora_id = "SultanR/sage-27b-it-lora"
            lora_path = snapshot_download(repo_id=lora_id)
        
        model_sampling_params = SamplingParams(
            temperature=random.random(),
            max_tokens=4096,
            stop_token_ids = [107]
            )
        
        for formatted_message,original_messages,last_assistant_response in conversations:
            try:
                if use_lora:
                    output = model.generate(
                        formatted_message, 
                        model_sampling_params, 
                        use_tqdm=False,
                        lora_request=LoRARequest("lora_adapter", 1, lora_path)
                        )
                else:
                    output = model.generate(formatted_message, model_sampling_params, use_tqdm=False)
                generated_text = output[0].outputs[0].text
                
                # Create complete conversation with generated response
                generated_conversation = original_messages.copy()
                generated_conversation.append({
                    "role": "assistant",
                    "content": generated_text
                })
                generated_conversations.append(generated_conversation)

                original_messages.append({
                    "role": "assistant",
                    "content": last_assistant_response
                })
                original_conversations.append(original_messages)
                score = judge_evaluator(generated_conversation, original_messages)
                scores.append(score)
            except Exception as e:
                if verbose:
                    print(f"Error in judge evaluation: {e}")
                exceptions += 1

        if exceptions / JUDGE_NUM_EVALS > MAX_ERROR_RATE:
            raise RuntimeError("judge score failed due to API issues")


        # win rate is at least 2/3 over original 


        # adjusted_evals = JUDGE_NUM_EVALS - exceptions
        # final_score = sum(scores) / adjusted_evals if adjusted_evals > 0 else 0
        # x= {"generated_conversations": generated_conversations, "original_conversations": original_conversations}
        x= {"generated_conversations": generated_conversations, "scores": scores}
        store_conversations(x)

        return {
            # "judge_score": final_score,
            "judge_score": 1,
        }
    except Exception as e:
        if verbose:
            print(e)
        raise e

def store_conversations(conversations, c : str = "x"):
    """
    Store conversations in a JSON file at /tmp/convos.json
    Args:
        conversations: List of conversation messages
    """
    try:
        with open(f'/tmp/judge_score_{c}.json', 'w') as f:
            json.dump(conversations, f, indent=2)
    except Exception as e:
        print(f"Error storing conversations: {e}")
