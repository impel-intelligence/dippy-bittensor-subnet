# Miner Guide
This document will outline advanced steps that miners can utilize for competing in the subnet.
## Running local evaluation
The same evaluation used by validators can be run locally. 
Note that some score results, such as latency, may not be 100% exact.

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
# Run evaluator
python dippy_validation_api/evaluator.py --image grader:latest \
--repo_namespace <your-repo-namespace> --repo_name <your-repo-name> \
--chat_template_type <your-chat-template-type> --hash <your-hash>
```

## Datasets Used
https://huggingface.co/datasets/Gryphe/Opus-WritingPrompts
https://huggingface.co/datasets/PygmalionAI/PIPPA