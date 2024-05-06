"""
Example script to download any arbitrary model and format the repo correctly.
"""
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from utilities.utils import save_model
from huggingface_hub import login
import dotenv

dotenv.load_dotenv()

login(
    token=os.environ["HF_ACCESS_TOKEN"],
)


model_name = 'lmsys/vicuna-13b-v1.5'
save_path = 'bittensor_models'
model_dir_name = model_name.split('/')[1]

print(f"Loading model {model_name}")
# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Save the model and tokenizer
print(f"Saving model {model_name}")
save_model(model, tokenizer, save_path, model_dir_name)
print(f"Model {model_name} saved to {save_path}")
