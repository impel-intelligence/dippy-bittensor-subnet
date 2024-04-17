import json
import random
import jinja2

from transformers import AutoTokenizer
from torch.utils.data import Dataset


class PippaDataset(Dataset):
    def __init__(self, filename):
        self.filename = filename
        with open(filename, 'r') as f:
            data = [json.loads(line) for line in f]

        self.dataset = self.process_data(data)

        self._chat_template = None
        self._tokenizer = None

    def set_chat_template_params(self, template_path: str, tokenizer: AutoTokenizer):
        self._chat_template = jinja2.Template(open(template_path).read())
        self._tokenizer = tokenizer

    def process_data(self, data):
        """
        Convert pippa dataset to a format that can be used downstream.
        """
        converted_dataset = []
        for data_point in data:
            # construct the system prompt using the bot_description and bot_greeting
            if not data_point['bot_definitions']:
                data_point['bot_definitions'] = ''

            if not data_point['categories']:
                data_point['categories'] = 'None'

            system_prompt = f"""A chat between a user and a curious artificial intelligence that is an expert at roleplay. 
The AI is roleplaying as a character named {data_point['bot_name']}. 
The character's description: {data_point['bot_description']}. {data_point['bot_definitions']}.
The themes of the conversation are: {data_point['categories']}."""

            messages = [{
                'role': 'system',
                'content': system_prompt
            }]

            messages.append(
                {
                    'role': 'assistant',
                    'content': f"{data_point['bot_name']}: {data_point['bot_greeting']}"
                }
            )

            # get index of the last message from the chatbot
            last_message_index = 0
            for i, message in enumerate(data_point['conversation']):
                if not message['is_human']:
                    last_message_index = i
            
            for message in data_point['conversation'][:last_message_index]:
                if message['is_human']:
                    messages.append(
                        {
                            'role': 'user',
                            'content': message['message']
                        }
                    )
                else:
                    messages.append(
                        {
                            'role': 'assistant',
                            'content': f"{message['message']}"
                        }
                    )
            
            character_response = data_point['conversation'][last_message_index]['message']

            converted_dataset.append(
                {
                    'messages': messages,
                    'character_response': character_response
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
            bos_token='',
            eos_token='',
            messages=self.dataset[idx]['messages'],
            include_beginning_of_conversation=True,
            add_generation_prompt=True
        )


        return chat_input, f"{self.dataset[idx]['character_response']}{self._tokenizer.eos_token}"
    
    def sample_dataset(self, n: int):
        # get indices of the dataset
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        indices = indices[:n]
                
        return [self[i] for i in indices]
        
