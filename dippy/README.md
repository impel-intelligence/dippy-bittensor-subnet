Some pointers for the final doc:

Requirements:
- The model should load with AutoModelForCausalLM with the repo ID, score will be 0 otherwise.
- model.config.max_position_embeddings should not be None
- naturally, models with higher max sequence length will be better as the long sequences will be truncated otherwise
- ideally the miners should provide the quantization config for accurate results.


HOW TO USE:

To install the required packages:
pip install -r requirements.txt


To test if scoring function works on a toy example, run:
python3 test_get_eval_score.py


To run API:
python3 validation_api.py and monitor the terminal for output. Final scores are returned by the API too.