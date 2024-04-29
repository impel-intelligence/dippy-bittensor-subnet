import torch
from validation_api import get_eval_score
from dataset import PippaDataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the dataset
dataset_path = "data/pippa_deduped.jsonl"
dataset = PippaDataset(dataset_path, max_input_len=256)

model_name = 'Manavshah/llama-test' # Replace with your actual model name

# model_name = 'openai-community/gpt2'
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
)
# load the model without downloading
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map='auto' if device == "cuda" else 'cpu',
    quantization_config=quant_config,
)

# Load the tokenizers
input_tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    padding_side='left',
)
output_tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    padding_side='right',
)
if input_tokenizer.pad_token is None:
    input_tokenizer.pad_token = input_tokenizer.eos_token  # Ensure pad token is set
    output_tokenizer.pad_token = output_tokenizer.eos_token  # Ensure pad token is set

dataset.set_chat_template_params('prompt_templates/vicuna_prompt_template.jinja', input_tokenizer)

# Prepare sample data
sample_size = 2  # Adjust as needed
sampled_data = dataset.sample_dataset(sample_size)

# sampled_data = [
#     ("What is the capital of France?", "\n\nParis is the capital of France.")
# ]

# Evaluate the model
eval_score = get_eval_score(
    model, 
    sampled_data, 
    input_tokenizer=input_tokenizer,
    output_tokenizer=output_tokenizer,
    debug=True
)

# Output the evaluation score
print(f"Model evaluation score: {eval_score}")