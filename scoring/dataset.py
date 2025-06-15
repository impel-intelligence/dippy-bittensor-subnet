import json
import random
import jinja2
import os
from typing import Any, Dict, List
from transformers import AutoTokenizer
from torch.utils.data import Dataset
from datetime import datetime, timezone
from scoring.common import SPECIAL_TOKENS
from urllib.parse import urlparse
from datasets import load_dataset, get_dataset_split_names

DATASET_CACHE_DIR = "evalsets"
hf_token = os.environ.get("HF_TOKEN")
DATASET_NAME = "https://huggingface.co/datasets/DippyAI/conversations"
def prepare_from_hf_dataset(repo_id: str):
    dataset_ = load_dataset(repo_id, streaming=False, token=hf_token, cache_dir=DATASET_CACHE_DIR)
    return dataset_["train"]

import requests
DATASET_URL = "localhost"
BACKUP_URLS = [
]
DATASET_API_JWT = os.environ.get("DATASET_API_JWT", "dippy")

DEFAULT_EPOCH_DATE = "20250321"


"""
In the case of requiring multiple fetches to the dataset api:
It would be more efficient to save a single API result to json and load accordingly.

"""

def get_latest_from_file(filter: str = "both", filename: str = "/tmp/dataset.json"):
    try:
        with open(filename, "r") as file:
            data = json.load(file)

        if not isinstance(data, list):
            raise ValueError("The top-level structure in the JSON file is not an array.")

        # Ensure each item in the list is a dictionary
        data = [item for item in data]

        print(f"loaded data with len {len(data)}")
    except FileNotFoundError:
        print("Error: dataset.json file not found.")
        data = []
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in dataset.json.")
        data = []
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        data = []
    return data

class HuggingfaceDataset(Dataset):
    CHAR_TO_TOKEN_RATIO = 4.0  # Rule-of-thumb ratio for characters to tokens for approximation.

    def __init__(self, max_messages: int):
        repo_id = "DippyAI/dataset0"
        try:
            data = prepare_from_hf_dataset(repo_id)
            self.raw_data = data
        except Exception as e:
            print(f"error loading dataset from Hugging Face: {e}")
            raise e
        self._chat_template = None
        self._tokenizer = None
        self._last_sampled_indices = None
        self.max_messages = max_messages
    def process(self):
        data = self.raw_data
        self.dataset = self.process_data(data, self.max_messages)
        return self

    def set_chat_template_params(self, template_path: str, tokenizer: AutoTokenizer):
        self._chat_template = jinja2.Template(open(template_path).read())
        self._tokenizer = tokenizer
        self._last_sampled_indices = None

    def set_chat_template_params_from_str(self, template: str, tokenizer: AutoTokenizer):
        self._chat_template = jinja2.Template(template)
        self._tokenizer = tokenizer

    def _check_token_limit(
        self, current_token_count: float, new_message: Dict[str, str], limit: int = 8192
    ) -> (bool, float):
        # Estimate tokens based on character count. This is a fast approximation.
        tokens_to_add = len(new_message["content"]) / self.CHAR_TO_TOKEN_RATIO

        is_over_limit = (current_token_count + tokens_to_add) > limit

        return is_over_limit, tokens_to_add

    def process_data(self, data, max_messages):
        converted_dataset = []
        for data_point in data:
            if not data_point.get("messages"):
                continue
            system_prompt = data_point["system_prompt"]
            messages = [{"role": "system", "content": system_prompt}]

            # Initialize running token count using the character ratio approximation
            running_token_count = len(system_prompt) / self.CHAR_TO_TOKEN_RATIO

            msg_roles = data_point["messages"]["role"]
            msg_contents = data_point["messages"]["content"]

            for i, (role, content) in enumerate(zip(msg_roles, msg_contents)):
                if i >= max_messages:
                    break

                # Remove special tokens from message content
                for special_token in SPECIAL_TOKENS:
                    content = content.replace(special_token, "")
                entry = {
                    "role": role,
                    "content": content,
                }

                # Use the character-based approximation check
                is_over_limit, tokens_to_add = self._check_token_limit(running_token_count, entry)
                if is_over_limit:
                    break

                messages.append(entry)
                running_token_count += tokens_to_add

            if len(messages) > 1 and messages[-1]["role"] == "user":
                messages.pop()

            if len(messages) <= 1:
                continue

            character_response = messages.pop()["content"]
            last_user_message = next((msg["content"] for msg in reversed(messages) if msg["role"] == "user"), None)
            if last_user_message is None:
                continue
            converted_dataset.append(
                {
                    "messages": messages,
                    "last_user_message": last_user_message,  # get the last user message
                    "character_response": character_response,
                }
            )

        return converted_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self._chat_template is None:
            raise ValueError("Chat template is not set. Please set the chat template before generating chat.")

        if self._tokenizer is None:
            raise ValueError("Tokenizer is not set. Please set the tokenizer before generating chat.")

        messages_to_render = self.dataset[idx]["messages"]
        if len(messages_to_render) < 1:
            raise ValueError("empty messages")
        for m in messages_to_render:
            if len(m["content"]) < 1:
                raise ValueError("empty message content")

        rendered = self._chat_template.render(
            bos_token=self._tokenizer.bos_token,
            eos_token=self._tokenizer.eos_token,
            messages=messages_to_render,
            include_beginning_of_conversation=True,
            add_generation_prompt=True,
        )  # shouldn't end with eos token
        if rendered is None:
            raise ValueError(f"chat_input could not be rendered")

        if self._tokenizer.eos_token is not None and rendered.endswith(self._tokenizer.eos_token):
            rendered = rendered[: -len(self._tokenizer.eos_token)]

        if self._tokenizer.bos_token is not None and not rendered.startswith(self._tokenizer.bos_token):
            rendered = f"{self._tokenizer.bos_token}{rendered}"
        next_assistant = f"{self.dataset[idx]['character_response']}{self._tokenizer.eos_token}"
        return (
            rendered,  # context
            messages_to_render,
            next_assistant
        )

    def sample_dataset(self, n: int, messages_limit: int = None):
        # get indices of the dataset
        indices = list(range(len(self.dataset)))
        if messages_limit is not None:
            indices = [
                i for i in indices if len(self.dataset[i].get("messages", [])) <= messages_limit
            ]
        random.shuffle(indices)
        indices = indices[:n]
        error_count = 0
        sampled_data = []
        valid_indices = []
        for i in indices:
            try:
                sample_data = self[i]
                sampled_data.append(sample_data)
                valid_indices.append(i)
            except Exception as e:
                error_count += 1
                print(f"Skipping index {i} due to error: {str(e)}")
                continue
        if error_count > 0:
            print(f"Skipped {error_count} samples due to errors")
            # Try to get additional samples to replace the errors
            remaining_indices = [i for i in range(len(self.dataset)) if i not in indices]
            random.shuffle(remaining_indices)
            additional_needed = min(error_count, len(remaining_indices))

            for i in remaining_indices[:additional_needed]:
                try:
                    sample_data = self[i]
                    sampled_data.append(sample_data)
                    valid_indices.append(i)
                except Exception as e:
                    print(f"Additional sample at index {i} also failed: {str(e)}")
                    continue

            print(f"Added {len(sampled_data) - (n - error_count)} additional valid samples")

        self._last_sampled_indices = valid_indices
        return sampled_data

    def get_original_messages(self):
        """Returns the original messages for the last sampled dataset."""
        if self._last_sampled_indices is None:
            raise ValueError("No samples have been generated yet. Call sample_dataset() first.")

        original_messages = []
        for idx in self._last_sampled_indices:
            original_messages.append(
                {
                    "messages": self.dataset[idx]["messages"],
                    "last_user_message": self.dataset[idx]["last_user_message"],
                    "character_response": self.dataset[idx]["character_response"],
                }
            )
        return original_messages

