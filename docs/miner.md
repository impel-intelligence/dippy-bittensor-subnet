# Miner Guide
This document will outline advanced steps that miners can utilize for competing in the subnet.
## Running local evaluation
The same evaluation used by validators can be run locally. 
Note that some score results, such as latency, may not be 100% exact given the nature of the scoring.

In the root of this repository, first build the docker image:
```shell
docker build -f evaluator.Dockerfile -t grader:latest .
```
This will build the docker image used for determining model score.

We recommend [`uv`](https://pypi.org/project/uv/) for managing your python environment. 
Feel free to modify the below with the appropriate commands if you use anything otherwise.

```shell
# Setup virtual env
uv venv
source .venv/bin/activate
uv pip install -r requirements.eval.txt
# You will need access to openai to grade coherence. Will transition to Corcel in a future update
export OPENAI_API_KEY=x
# You will also need access to the dataset api to set a token. Be sure to follow the instructions set out in `token_check.py` to set the correct value here
export DATASET_API_KEY="Bearer YOUR_JWT_HERE"

# In the `dippy_validation_api/evaluator.py` script, there is the following line:
evaler = Evaluator(image_name=image_name, trace=True, gpu_ids="0")
# Edit the gpu_ids as needed to map to your actual GPU ids

# Run evaluator
python dippy_validation_api/evaluator.py --image grader:latest \
--repo_namespace <your-repo-namespace> --repo_name <your-repo-name> \
--chat_template_type <your-chat-template-type> --hash <your-hash>
```

## Comparing validation results

It is possible to run the same steps of code as a validator could to get a better scope of weights and scores.
To do so, ensure that you have installed all the dependencies for a validator and run the following from the root repository:
```shell
python neurons/validator.py --wallet.name <your-wallet> --wallet.hotkey <your-hotkey> --offline --immediate
```
This command will run the validation run step immediately. Note that this will run indefinitely and in rapid succession
## Datasets Used
https://huggingface.co/datasets/DippyAI/dippy_synthetic_dataset
This dataset is an archive of previously generated synthetic data. The current evaluation samples from both this and a more recently generated stream of synthetic data.

https://huggingface.co/datasets/DippyAI/personahub_augmented_v0
This dataset is not meant to be trained on, but rather a reference for how the current implementation of coherence is calculated. 
