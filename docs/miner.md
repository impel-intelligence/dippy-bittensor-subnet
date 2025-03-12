# Miner Guide
This document will outline advanced steps that miners can utilize for competing in the subnet.

## Environment Setup
We recommend [`uv`](https://pypi.org/project/uv/) for managing your python environment. The following instructions will assume that you are running a uv virtual environment.

```shell
uv pip install -r requirements.miner.txt
uv pip install -e .
```


## Submitting a model

Register on the subnet
```shell
btcli s register --netuid 11
```

```shell
python neurons/miner.py --wallet.name coldkey  --wallet.hotkey hotkey --repo_namespace <your_huggingface_username> --repo_name <your_huggingface_repo> --chat_template <your_chat_template> --online True
```

## Running local evaluation

The evaluation code used by validators can also be run locally by miners. 
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
# Install requirements specifically for running evaluator script
uv pip install -r requirements.miner.txt
# Install local package
uv pip install -e .

# You will need access to openrouter to grade coherence.
export OPENROUTER_API_KEY=x
# Build the docker image used to score locally
docker build -f evaluator.Dockerfile -t grader:latest .
# In the `dippy_validation_api/evaluator.py` script, there is the following line:
evaler = Evaluator(image_name=image_name, trace=True, gpu_ids="0")
# Edit the gpu_ids as needed to map to your actual GPU ids

# Run evaluator
python dippy_validation_api/evaluator.py --image grader:latest \
--repo_namespace <your-repo-namespace> --repo_name <your-repo-name> \
--chat_template_type <your-chat-template-type> --hash <your-hash>
```


## How to be a competitive miner
As a miner, you're responsible for leveraging all methods available at your disposal, including but not limited to training new models, merging existing models (we recommend [MergeKit](https://github.com/arcee-ai/mergekit)), finetuning existing models, and so on to push roleplay LLMs forward.

Given that model training can be computationally expensive, we highly recommend exhausting all requisite testing methods before making an official submission. Given a failed model, the subnet will not re-queue or otherwise reassess models for any reason outside of specific reasonable scenarios. Miners can always create new submissions via whichever method of their choice. 

### Using cached dataset
Sometimes, when running multiple instances of evaluation, it can help to utilize a cached version of the dataset to prevent issues with rate limits.
To do so, simply save the results of the dataset endpoint. Afterwards, you can modify the dataset URL in the scoring file to use a locally hosted API. Note that this API is _not_ the same as the `dippy_validation_api` in this project.

## Comparing validation results

It is possible to run the same steps of code as a validator could to get a better scope of weights and scores.
To do so, ensure that you have installed all the dependencies for a validator and run the following from the root repository:
```shell
python neurons/validator.py --wallet.name <your-wallet> --wallet.hotkey <your-hotkey> --offline --immediate
```
This command will run the validation run step immediately. Note that this will run indefinitely and in rapid succession
## Datasets Used (deprecated)
https://huggingface.co/datasets/DippyAI/dippy_synthetic_dataset_v1
This dataset is an archive of previously generated synthetic data. The current evaluation samples from both this and a more recently generated stream of synthetic data.

https://huggingface.co/datasets/DippyAI/personahub_augmented_v0
This dataset is not meant to be trained on, but rather a reference for how the current implementation of coherence score is calculated. 
