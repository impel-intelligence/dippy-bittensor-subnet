import json
import random
import jinja2
import os
from typing import Any, Dict, List
from transformers import AutoTokenizer
from torch.utils.data import Dataset
import tiktoken


class PippaDataset(Dataset):
    def __init__(self, filename, max_input_len):
        self.filename = filename
        with open(filename, "r") as f:
            data = [json.loads(line) for line in f]

        self.dataset = self.process_data(data, max_input_len)

        self._chat_template = None
        self._tokenizer = None

    def set_chat_template_params(self, template_path: str, tokenizer: AutoTokenizer):
        self._chat_template = jinja2.Template(open(template_path).read())
        self._tokenizer = tokenizer

    def process_data(self, data, max_input_len):
        """
        Convert pippa dataset to a format that can be used downstream.
        """
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")  # to get approx token count
        converted_dataset = []
        for data_point in data:
            # construct the system prompt using the bot_description and bot_greeting
            if not data_point["bot_definitions"]:
                data_point["bot_definitions"] = ""

            if not data_point["categories"]:
                data_point["categories"] = "None"

            system_prompt = f"""A chat between a user and a curious artificial intelligence that is an expert at roleplay. 
The AI is roleplaying as a character named {data_point['bot_name']}. 
The character's description: {data_point['bot_description']}. {data_point['bot_definitions']}.
The themes of the conversation are: {data_point['categories']}."""

            messages = [{"role": "system", "content": system_prompt}]

            messages.append(
                {
                    "role": "assistant",
                    "content": f"{data_point['bot_name']}: {data_point['bot_greeting']}",
                }
            )

            # get index of the last message from the chatbot
            last_message_index = 0
            input_len_so_far = len(encoding.encode(messages[0]["content"] + messages[1]["content"]))

            if input_len_so_far > max_input_len:
                # skip this data point
                continue

            for i, message in enumerate(data_point["conversation"]):
                input_len_so_far += len(encoding.encode(message["message"]))
                if input_len_so_far > max_input_len:
                    break

                if not message["is_human"]:
                    last_message_index = i

            last_user_message_index = 0
            for i, message in enumerate(data_point["conversation"][:last_message_index]):
                if message["is_human"]:
                    messages.append({"role": "user", "content": message["message"]})
                    last_user_message_index = i
                else:
                    messages.append({"role": "assistant", "content": f"{message['message']}"})

            character_response = data_point["conversation"][last_message_index]["message"]
            last_user_message = messages[last_user_message_index]["content"]

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

        chat_input = self._chat_template.render(
            bos_token=self._tokenizer.bos_token,
            eos_token=self._tokenizer.eos_token,
            messages=self.dataset[idx]["messages"],
            include_beginning_of_conversation=True,
            add_generation_prompt=True,
        )  # shouldn't end with eos token

        if chat_input.endswith(self._tokenizer.eos_token):
            chat_input = chat_input[: -len(self._tokenizer.eos_token)]

        if not chat_input.startswith(self._tokenizer.bos_token):
            chat_input = f"{self._tokenizer.bos_token}{chat_input}"

        return (
            chat_input,  # context
            f"{self.dataset[idx]['character_response']}{self._tokenizer.eos_token}",  # target text
            self.dataset[idx]["last_user_message"],  # last user message
        )

    def sample_dataset(self, n: int):
        # get indices of the dataset
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        indices = indices[:n]

        return [self[i] for i in indices]


class BippaDataset(Dataset):
    def __init__(self, filename, max_input_len):
        self.filename = filename
        with open(filename, "r") as f:
            data = [json.loads(line) for line in f]

        self.dataset = self.process_data(data, max_input_len)

        self._chat_template = None
        self._tokenizer = None

    def set_chat_template_params(self, template_path: str, tokenizer: AutoTokenizer):
        self._chat_template = jinja2.Template(open(template_path).read())
        self._tokenizer = tokenizer

    def process_data(self, data, max_input_len):
        """
        Convert pippa dataset to a format that can be used downstream.
        """
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")  # to get approx token count
        converted_dataset = []
        for data_point in data:
            system_prompt = data_point["system_prompt"]
            messages = [{"role": "system", "content": system_prompt}]
            # get index of the last message from the chatbot
            input_len_so_far = len(encoding.encode(messages[0]["content"]))

            for chat_message in data_point["messages"]:
                input_len_so_far += len(encoding.encode(chat_message["content"]))
                if input_len_so_far > max_input_len:
                    break
                entry = {
                    "role": chat_message["role"],
                    "content": chat_message["content"],
                }
                messages.append(entry)
            character_response = messages.pop()["content"]
            last_user_message = messages[-1]["content"]

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

        chat_input = self._chat_template.render(
            bos_token=self._tokenizer.bos_token,
            eos_token=self._tokenizer.eos_token,
            messages=self.dataset[idx]["messages"],
            include_beginning_of_conversation=True,
            add_generation_prompt=True,
        )  # shouldn't end with eos token

        if chat_input.endswith(self._tokenizer.eos_token):
            chat_input = chat_input[: -len(self._tokenizer.eos_token)]

        if not chat_input.startswith(self._tokenizer.bos_token):
            chat_input = f"{self._tokenizer.bos_token}{chat_input}"

        return (
            chat_input,  # context
            f"{self.dataset[idx]['character_response']}{self._tokenizer.eos_token}",  # target text
            self.dataset[idx]["last_user_message"],  # last user message
        )

    def sample_dataset(self, n: int):
        # get indices of the dataset
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        indices = indices[:n]

        return [self[i] for i in indices]


DATASET_CACHE_DIR = "evalsets"
hf_token = os.environ.get("HF_TOKEN")


def prepare_from_hf_dataset(dataset_name: str, partitions: List[str]):
    dataset_ = load_dataset(dataset_name, streaming=True, token=hf_token, cache_dir=DATASET_CACHE_DIR)
    partial_data = []
    for partition in partitions:
        if partition not in dataset_:
            continue
        partition_data = [d["data"] for d in dataset_[partition]]
        partial_data.extend(partition_data)
    return partial_data


class DippaFormattedDataset(Dataset):
    def __init__(self, dataset_name: str, partitions: List[str], max_input_len: int):
        self.dataset_name = dataset_name

        data = prepare_from_hf_dataset(dataset_name, partitions)

        self.dataset = self.process_data(data, max_input_len)

        self._chat_template = None
        self._tokenizer = None

    def set_chat_template_params(self, template_path: str, tokenizer: AutoTokenizer):
        self._chat_template = jinja2.Template(open(template_path).read())
        self._tokenizer = tokenizer

    def process_data(self, data, max_input_len):
        """
        Convert pippa dataset to a format that can be used downstream.
        """
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")  # to get approx token count
        converted_dataset = []
        for data_point in data:
            system_prompt = data_point["system_prompt"]
            messages = [{"role": "system", "content": system_prompt}]
            # get index of the last message from the chatbot
            input_len_so_far = len(encoding.encode(messages[0]["content"]))

            for chat_message in data_point["messages"]:
                input_len_so_far += len(encoding.encode(chat_message["content"]))
                if input_len_so_far > max_input_len:
                    break
                entry = {
                    "role": chat_message["role"],
                    "content": chat_message["content"],
                }
                messages.append(entry)
            character_response = messages.pop()["content"]
            last_user_message = messages[-1]["content"]

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

        chat_input = self._chat_template.render(
            bos_token=self._tokenizer.bos_token,
            eos_token=self._tokenizer.eos_token,
            messages=self.dataset[idx]["messages"],
            include_beginning_of_conversation=True,
            add_generation_prompt=True,
        )  # shouldn't end with eos token

        if chat_input.endswith(self._tokenizer.eos_token):
            chat_input = chat_input[: -len(self._tokenizer.eos_token)]

        if not chat_input.startswith(self._tokenizer.bos_token):
            chat_input = f"{self._tokenizer.bos_token}{chat_input}"

        return (
            chat_input,  # context
            f"{self.dataset[idx]['character_response']}{self._tokenizer.eos_token}",  # target text
            self.dataset[idx]["last_user_message"],  # last user message
        )

    def sample_dataset(self, n: int):
        # get indices of the dataset
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        indices = indices[:n]

        return [self[i] for i in indices]

import requests
DATASET_URL = "http://75.101.234.38:8111/latest"
DATASET_API_KEY = os.environ.get("DATASET_API_KEY", "dippy")
def get_latest_from_set():
    response = requests.get(DATASET_URL, params={"key": DATASET_API_KEY})
    response.raise_for_status()  # Raise an error for bad responses
    data = response.json().get("data", [])
    return data

class StreamedSyntheticDataset(Dataset):
    def __init__(self, max_input_len: int):
        data = get_latest_from_set()

        self.dataset = self.process_data(data, max_input_len)

        self._chat_template = None
        self._tokenizer = None

    def set_chat_template_params(self, template_path: str, tokenizer: AutoTokenizer):
        self._chat_template = jinja2.Template(open(template_path).read())
        self._tokenizer = tokenizer

    def process_data(self, data, max_input_len):
        """
        Convert pippa dataset to a format that can be used downstream.
        """
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")  # to get approx token count
        converted_dataset = []
        for data_point in data:
            system_prompt = data_point["system_prompt"]
            messages = [{"role": "system", "content": system_prompt}]
            # get index of the last message from the chatbot
            input_len_so_far = len(encoding.encode(messages[0]["content"]))

            for chat_message in data_point["messages"]:
                input_len_so_far += len(encoding.encode(chat_message["content"]))
                if input_len_so_far > max_input_len:
                    break
                entry = {
                    "role": chat_message["role"],
                    "content": chat_message["content"],
                }
                messages.append(entry)
            character_response = messages.pop()["content"]
            last_user_message = messages[-1]["content"]

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

        chat_input = self._chat_template.render(
            bos_token=self._tokenizer.bos_token,
            eos_token=self._tokenizer.eos_token,
            messages=self.dataset[idx]["messages"],
            include_beginning_of_conversation=True,
            add_generation_prompt=True,
        )  # shouldn't end with eos token

        if chat_input.endswith(self._tokenizer.eos_token):
            chat_input = chat_input[: -len(self._tokenizer.eos_token)]

        if not chat_input.startswith(self._tokenizer.bos_token):
            chat_input = f"{self._tokenizer.bos_token}{chat_input}"

        return (
            chat_input,  # context
            f"{self.dataset[idx]['character_response']}{self._tokenizer.eos_token}",  # target text
            self.dataset[idx]["last_user_message"],  # last user message
        )

    def sample_dataset(self, n: int):
        # get indices of the dataset
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        indices = indices[:n]

        return [self[i] for i in indices]


class PromptDataset(Dataset):
    def __init__(self, filenames, max_input_len):
        all_data = []
        for filename in filenames:
            with open(filename, "r") as f:
                data = [json.loads(line) for line in f]
                all_data.append(data)

        self.dataset = self.process_data(all_data, max_input_len)
        self._chat_template = None
        self._tokenizer = None

    def set_chat_template_params(self, template_path: str, tokenizer: AutoTokenizer):
        self._chat_template = jinja2.Template(open(template_path).read())
        self._tokenizer = tokenizer

    def process_data(self, data, max_input_len):
        """
        Convert opus dataset to a format that can be used downstream.
        """

        converted_dataset = []

        for data_point in data:
            for entry in data_point:
                # Always only 3 messages
                conversations = entry["conversations"]
                messages = [
                    {"role": "system", "content": conversations[0]["value"]},
                ]
                prompt_content = f'{conversations[1]["value"]} \n Please limit to (200-300) words.'
                messages.append(
                    {"role": "user", "content": prompt_content},
                )
                output = conversations[2]["value"]
                converted_dataset.append(
                    {
                        "messages": messages,
                        "output": output,
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
        chat_input = self._chat_template.render(
            bos_token=self._tokenizer.bos_token,
            eos_token=self._tokenizer.eos_token,
            messages=self.dataset[idx]["messages"],
            include_beginning_of_conversation=True,
            add_generation_prompt=True,
        )  # shouldn't end with eos token

        if chat_input.endswith(self._tokenizer.eos_token):
            chat_input = chat_input[: -len(self._tokenizer.eos_token)]

        if not chat_input.startswith(self._tokenizer.bos_token):
            chat_input = f"{self._tokenizer.bos_token}{chat_input}"
        return (
            chat_input,  # context
            self.dataset[idx]["messages"],  # full message history
            self.dataset[idx]["output"],  # prompt output for comparison
        )

    def sample_dataset(self, n: int):
        # get indices of the dataset
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        indices = indices[:n]

        return [self[i] for i in indices]


from datasets import load_dataset


class SyntheticCoherenceDataset(Dataset):
    def __init__(self, dataset_name="DippyAI/dippa_dataset_test0"):
        datass = load_dataset(dataset_name, token=os.environ.get("HF_TOKEN")).get("train", [])

        self.dataset = self.process_data(datass)
        self.max_size = len(datass)
        self._chat_template = None
        self._tokenizer = None

    def set_chat_template_params(self, template_path: str, tokenizer: AutoTokenizer):
        self._chat_template = jinja2.Template(open(template_path).read())
        self._tokenizer = tokenizer

    def cut_messages(self, entry: Dict[Any, Any]):
        messages = entry["messages"]
        # Truncate the messages list
        truncated_messages = messages[:-1]
        return truncated_messages

    def process_data(self, datass):
        """
        Convert dataset to a format that can be used downstream.
        """

        converted_dataset = []

        for entry in datass:
            system_prompt = entry["system_prompt"]
            messages = [
                {"role": "system", "content": system_prompt},
            ]
            cut_messages = self.cut_messages(entry)
            for m in cut_messages:
                messages.append(
                    {
                        "role": m["role"],
                        "content": m["content"],
                    }
                )

            converted_dataset.append(
                {
                    "messages": messages,
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
        chat_input = self._chat_template.render(
            bos_token=self._tokenizer.bos_token,
            eos_token=self._tokenizer.eos_token,
            messages=self.dataset[idx]["messages"],
            include_beginning_of_conversation=True,
            add_generation_prompt=True,
        )  # shouldn't end with eos token

        if chat_input.endswith(self._tokenizer.eos_token):
            chat_input = chat_input[: -len(self._tokenizer.eos_token)]

        if not chat_input.startswith(self._tokenizer.bos_token):
            chat_input = f"{self._tokenizer.bos_token}{chat_input}"
        return (
            chat_input,  # context
            self.dataset[idx]["messages"],  # full message history
        )

    def sample_dataset(self, n: int):
        # get indices of the dataset
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        indices = indices[:n]

        return [self[i] for i in indices]
