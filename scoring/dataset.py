import json
import random
import jinja2
import os
from typing import Any, Dict, List
from transformers import AutoTokenizer
from torch.utils.data import Dataset
import tiktoken
from datetime import datetime, timezone

DATASET_CACHE_DIR = "evalsets"
hf_token = os.environ.get("HF_TOKEN")
DATASET_USERNAME = os.environ.get("DATASET_USERNAME", "x")
DATASET_PASSWORD = os.environ.get("DATASET_PASSWORD", "x")


def prepare_from_hf_dataset(dataset_name: str, partitions: List[str]):
    dataset_ = load_dataset(dataset_name, streaming=True, token=hf_token, cache_dir=DATASET_CACHE_DIR)
    partial_data = []
    for partition in partitions:
        if partition not in dataset_:
            continue
        partition_data = [d["data"] for d in dataset_[partition]]
        partial_data.extend(partition_data)
    return partial_data


import requests

DATASET_URL = "https://temp-miner-dataset-sn11.dippy-bittensor-subnet.com/dataset"
DATASET_API_JWT = os.environ.get("DATASET_API_JWT", "dippy")

DEFAULT_EPOCH_DATE = "20241201"

def get_latest_from_set():
    current_date = datetime.now(timezone.utc).strftime("%Y%m%d")
    url = f"{DATASET_URL}?start_date={DEFAULT_EPOCH_DATE}&end_date={current_date}"

    response = requests.get(
        url, headers={"Authorization": f"Bearer {DATASET_API_JWT}"}
    )
    response.raise_for_status()  # Raise an error for bad responses
    data = response.json().get("all_convos", [])
    return data


def get_latest_from_file(filter: str = "both", filename: str = "/tmp/dataset.json"):
    import json

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


class StreamedSyntheticDataset(Dataset):
    def __init__(self, max_input_len: int):
        try:
            data = get_latest_from_set()
        except Exception as e:
            print(f"error loading dataset {e}")
            raise e
        self.dataset = self.process_data(data, max_input_len)

        self._chat_template = None
        self._tokenizer = None

    def set_chat_template_params(self, template_path: str, tokenizer: AutoTokenizer):
        self._chat_template = jinja2.Template(open(template_path).read())
        self._tokenizer = tokenizer

    def process_data(self, data, max_input_len):
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")  # to get approx token count
        converted_dataset = []
        for data_point in data:
            system_prompt = data_point["system_prompt"]
            messages = [{"role": "system", "content": system_prompt}]
            # get index of the last message from the chatbot
            input_len_so_far = 0
            limit_reached = False
            for chat_message in data_point["messages"]:
                input_len_so_far += len(encoding.encode(chat_message["content"]))
                if input_len_so_far > max_input_len:
                    limit_reached = True
                    break

                entry = {
                    "role": chat_message["role"],
                    "content": chat_message["content"],
                }
                messages.append(entry)
            if limit_reached:
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

        messages = self.dataset[idx]["messages"]
        if len(messages) < 1:
            raise ValueError("empty messages")
        for m in messages:
            if len(m["content"]) < 1:
                raise ValueError("empty message content")

        chat_input = self._chat_template.render(
            bos_token=self._tokenizer.bos_token,
            eos_token=self._tokenizer.eos_token,
            messages=self.dataset[idx]["messages"],
            include_beginning_of_conversation=True,
            add_generation_prompt=True,
        )  # shouldn't end with eos token
        if chat_input is None:
            raise ValueError(f"chat_input could not be rendered")

        if self._tokenizer.eos_token is not None and chat_input.endswith(self._tokenizer.eos_token):
            chat_input = chat_input[: -len(self._tokenizer.eos_token)]

        if self._tokenizer.bos_token is not None and not chat_input.startswith(self._tokenizer.bos_token):
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
        error_count = 0
        sampled_data = []
        for i in indices:
            try:
                sample_data = self[i]
                sampled_data.append(sample_data)
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
                except Exception as e:
                    print(f"Additional sample at index {i} also failed: {str(e)}")
                    continue

            print(f"Added {len(sampled_data) - (n - error_count)} additional valid samples")
        return sampled_data


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

        if self._tokenizer.eos_token is not None and chat_input.endswith(self._tokenizer.eos_token):
            chat_input = chat_input[: -len(self._tokenizer.eos_token)]

        if not chat_input.startswith(self._tokenizer.bos_token):
            if self._tokenizer.bos_token is not None:
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
    def __init__(self, dataset_name="DippyAI/dippy_synthetic_dataset"):
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

        if self._tokenizer.eos_token is not None and chat_input.endswith(self._tokenizer.eos_token):
            chat_input = chat_input[: -len(self._tokenizer.eos_token)]

        if not chat_input.startswith(self._tokenizer.bos_token):
            if self._tokenizer.bos_token is not None:
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


class JSONLDataset(Dataset):
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
        converted_dataset = []

        for data_point in data:
            for entry in data_point:
                # Always only 3 messages
                system_prompt = entry["system_prompt"]
                augmented_system_prompt = f"""
                You are an assistant playing the following character:
                {system_prompt}
                """
                messages = [
                    {"role": "system", "content": augmented_system_prompt},
                ]

                first_message = entry["messages"][0]
                messages.append(
                    {"role": "user", "content": first_message["content"]},
                )
                converted_dataset.append(
                    {
                        "messages": messages,
                    }
                )

        return converted_dataset

    def __len__(self):
        return len(self.dataset)

    def new_input(self, messages):
        chat_input = self._chat_template.render(
            bos_token=self._tokenizer.bos_token,
            eos_token=self._tokenizer.eos_token,
            messages=messages,
            include_beginning_of_conversation=True,
            add_generation_prompt=True,
        )  # shouldn't end with eos token

        if self._tokenizer.eos_token is not None and chat_input.endswith(self._tokenizer.eos_token):
            chat_input = chat_input[: -len(self._tokenizer.eos_token)]

        if not chat_input.startswith(self._tokenizer.bos_token):
            if self._tokenizer.bos_token is not None:
                chat_input = f"{self._tokenizer.bos_token}{chat_input}"

        return chat_input

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

        if self._tokenizer.eos_token is not None and chat_input.endswith(self._tokenizer.eos_token):
            chat_input = chat_input[: -len(self._tokenizer.eos_token)]

        if not chat_input.startswith(self._tokenizer.bos_token):
            if self._tokenizer.bos_token is not None:
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


class PersonaHubDataset(Dataset):
    def __init__(self, max_input_len):
        all_data = load_dataset("DippyAI/personahub_augmented_v0", cache_dir=DATASET_CACHE_DIR)
        partitions = []
        for partition in all_data:
            partition_data = all_data.get(partition)
            for chunk in partition_data:
                prompt = chunk.get("data")
                partitions.append(prompt)
        self.dataset = self.process_data(partitions, max_input_len)
        self._chat_template = None
        self._tokenizer = None

    def set_chat_template_params(self, template_path: str, tokenizer: AutoTokenizer):
        self._chat_template = jinja2.Template(open(template_path).read())
        self._tokenizer = tokenizer

    def process_data(self, data, max_input_len):
        converted_dataset = []

        for entry in data:
            system_prompt = entry["system_prompt"]
            augmented_system_prompt = f"""
            You are an assistant playing the following character:
            {system_prompt}
            """
            messages = [
                {"role": "system", "content": augmented_system_prompt},
            ]

            first_message = entry["messages"][0]
            messages.append(
                {"role": "user", "content": first_message["content"]},
            )
            converted_dataset.append(
                {
                    "messages": messages,
                }
            )

        return converted_dataset

    def __len__(self):
        return len(self.dataset)

    def new_input(self, messages):
        chat_input = self._chat_template.render(
            bos_token=self._tokenizer.bos_token,
            eos_token=self._tokenizer.eos_token,
            messages=messages,
            include_beginning_of_conversation=True,
            add_generation_prompt=True,
        )  # shouldn't end with eos token

        if self._tokenizer.eos_token is not None and chat_input.endswith(self._tokenizer.eos_token):
            chat_input = chat_input[: -len(self._tokenizer.eos_token)]

        if self._tokenizer.bos_token is not None and not chat_input.startswith(self._tokenizer.bos_token):
            chat_input = f"{self._tokenizer.bos_token}{chat_input}"

        return chat_input

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

        if self._tokenizer.eos_token is not None and chat_input.endswith(self._tokenizer.eos_token):
            chat_input = chat_input[: -len(self._tokenizer.eos_token)]

        if self._tokenizer.bos_token is not None and not chat_input.startswith(self._tokenizer.bos_token):
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
