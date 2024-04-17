from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

from dippy.dataset import PippaDataset



model_name = 'mistralai/Mistral-7B-v0.1' # Replace with your actual model name

dataset_path = "data/pippa_deduped.jsonl"
dataset = PippaDataset(dataset_path)

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
)
# load the model 
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,
    device_map='auto'
)

# Load the tokenizer
input_tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    padding_side='left',
)

if input_tokenizer.pad_token is None:
    input_tokenizer.pad_token = input_tokenizer.eos_token  # Ensure pad token is set
    input_tokenizer.pad_token_id = input_tokenizer.eos_token_id  # Ensure pad token is set

# Prepare sample data
sample_size = 2  # Adjust as needed
dataset.set_chat_template_params('prompt_templates/vicuna_prompt_template.jinja', input_tokenizer)
sampled_data = dataset.sample_dataset(2)

prompts = [sample[0] for sample in sampled_data]

# Prepare the inputs with attention mask
inputs = input_tokenizer(prompts, return_tensors='pt', padding=True)
output_sequences = inputs['input_ids']
attention_mask = inputs['attention_mask']

for _ in range(10):
    with torch.no_grad():
        # Generate the next token
        outputs = model(output_sequences, attention_mask=attention_mask).logits
        
        # Get the last token logits for each sequence
        next_token_logits = outputs[:, -1, :]
        
        # Sample the next token from the probability distribution
        next_token = torch.multinomial(torch.nn.functional.softmax(next_token_logits, dim=-1), num_samples=1)
        
        # Append the last token to the sequence
        output_sequences = torch.cat((output_sequences, next_token), dim=-1)
        
        # Update the attention mask to consider the new token
        attention_mask = torch.cat((attention_mask, torch.ones((attention_mask.shape[0], 1), dtype=torch.long)), dim=-1)
        
        # Update the inputs for the next iteration
        inputs = {'input_ids': output_sequences, 'attention_mask': attention_mask}

# Decode the generated sequences to get the text output
generated_texts = input_tokenizer.batch_decode(output_sequences, skip_special_tokens=True)

# Print the generated texts
for prompt, generated_text in zip(prompts, generated_texts):
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}\n")
    



