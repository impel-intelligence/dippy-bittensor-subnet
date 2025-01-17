<div align="center">

# Dippy SN11: Creating The World's Best Open-Source Roleplay LLM <!-- omit in toc -->

*Check out the beta version of our [Front-End](https://bittensor.dippy.ai/play)! Also, please check our [Launch Tweet](https://twitter.com/angad_ai/status/1788993280002175415) for our vision of creating the world's best open-source roleplay LLM.*

[![DIPPY](/assests/banner.png)](https://dippy.ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

---

</div>

- [Introduction](#introduction)
- [Roadmap](#roadmap)
- [Overview of Miner and Validator Functionality](#overview-of-miner-and-validator-functionality)
  - [Miner](#miner)
  - [Validator](#validator)
- [Running Miners and Validators](#running-miners-and-validators)
  - [Running a Miner](#running-a-miner)
  - [Running a Validator](#running-a-validator)
- [Contributing](#contributing)
- [License](#license)

---

## Introduction

> **Note:** The following documentation assumes you are familiar with basic Bittensor concepts: Miners, Validators, and incentives. If you need a primer, please check out https://docs.bittensor.com/learn/bittensor-building-blocks.

Dippy is one of the world's leading AI companion apps with **1M+ users**. The app has ranked [**#3 on the App Store**](https://x.com/angad_ai/status/1850924240742031526) in countries like Germany, been covered by publications like [**Wired magazine**](https://www.wired.com/story/dippy-ai-girlfriend-boyfriend-reasoning/) and the average Dippy user **spends 1+ hour on the app.** 

The Dippy Roleplay subnet on Bittensor aims to create the world's best open-source roleplay LLM by leveraging the collective efforts of the open-source community. This subnet addresses the critical issue of loneliness, which affects a significant portion of the population and is linked to various mental and physical health problems. 

Current SOTA LLMs (Claude, OpenAI etc.) are designed for the assistant use case and lack the empathetic qualities necessary for companionship. While some companies (like Character AI and Inflection) have developed closed-source roleplay LLMs, the open-source alternatives lag significantly behind in performance. 

![DIPPY](/assests/comp.png)

## Roadmap

Given the complexity of creating a state of the art roleplay LLM, we plan to divide the process into 3 distinct phases.

**Phase 1:** 
- [x] Subnet launch with robust pipeline for roleplay LLM evaluation on public datasets and response length 
- [x] Public model leaderboard based on evaluation criteria
- [x] Introduce Coherence and Creativity as a criteria for live model evaluation

**Phase 2:** 
- [x] Publicly release front-end powered by top miner submitted model of the week
- [ ] Segment model submission into different "expert" categories (funny, romantic, therapeutic etc)
- [ ] Models with the highest score in each personality type are chosen as "expert" models and made publicly available on the front-end

**Phase 3:** 
- [ ] New Mixture of Experts model made as a baseline based on the "expert" models chosen from Phase 2
- [ ] Robust pipeline to evaluate new MOE model submissions against live evaluation criteria
- [ ] Expand the state of the art in roleplay LLMs through continuous iteration and data collection

## Overview of Miner and Validator Functionality

![overview](/assests/architecturenew.png)

**Miners** would use existing frameworks, fine tuning techniques, or MergeKit, to train, fine tune, or merge models to create a unique roleplay LLM. These models would be submitted to a shared Hugging Face pool. 

**Validators** would evaluate the and assess model performance via our protocol and rank the submissions based on various metrics (empathy, conciseness etc). We will provide a suite of 
testing and benchmarking protocols with state-of-the-art datasets.

## Running Miners and Validators
### Running a Miner
> **Important:** Please carefully read through the [FAQ](docs/FAQ.md) and [Detailed Miner Documentation](docs/miner.md). These contain critical information about model requirements, evaluation criteria, and best practices that will help ensure your submissions are valid and competitive.



### Running a Validator

#### Requirements
- Python 3.9+

#### Setup
To start, clone the repository and `cd` to it:
```
git clone https://github.com/impel-intelligence/dippy-bittensor-subnet.git
cd dippy-bittensor-subnet
pip install -e .
```
To run the evaluation, simply use the following command:

``` 
python neurons/validator.py --wallet.name WALLET_NAME --wallet.hotkey WALLET_HOT_NAME
```

To run auto-updating validator with PM2 (recommended):
```bash
pm2 start --name sn11-vali-updater --interpreter python scripts/start_validator.py -- --pm2_name sn11-vali --wallet.name WALLET_NAME --wallet.hotkey WALLET_HOT_NAME [other vali flags]
```

If you wish to use a local subtensor node, the additional flags required are `--local` in additional to the typical arguments. 
Example:
```bash
python neurons/validator.py \
--wallet.name coldkey \
--wallet.hotkey hotkey \
--local \
--subtensor.network local --subtensor.chain_endpoint ws://chain_endpoint
```

Please note that this validator will call the model validation service hosted by the dippy subnet owners. If you wish to run the model validation service locally, please follow the instructions below.

### Running the model evaluation API (Optional, not recommended)

**Note**: Currently (Jan 17 2025) there are some issues with the local evaluation api. We recommend using the remote validation api temporarily.

Starting a validator using your local validator API requires starting validator with `--use-local-validation-api` flag. 
Additionally, a model queue is required to push models to the validation api.

**Note**: Validator API needs to be installed in a different venv than validator due to `pydantic` version conflict. 


### Requirements
- Python 3.9+
- Linux

#### Setup

Install Git Lfs if not installed.
```bash
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
```

To start, clone the repository and `cd` into it:
```bash
git clone https://github.com/impel-intelligence/dippy-bittensor-subnet.git
cd dippy-bittensor-subnet
python3 -m venv model_validation_venv
source model_validation_venv/bin/activate
model_validation_venv/bin/pip install -e . --no-deps
model_validation_venv/bin/pip install -r requirements.api.txt
```

#### Run model validation API service (optional, not recommended)
(Note: there are currently breaking changes that pose challenges to running a local validation API service. Any tasks that require the env vars `ADMIN_KEY` or `DIPPY_KEY` applies here)
```bash
cd dippy_validation_api
chmod +x start_validation_service.sh
./start_validation_service.sh
```

#### Stop model validation API service
```bash
chmod +x kill_validation_api.sh
./kill_validation_api.sh
```

#### Running the validator with your own validation API service running locally (optional, not recommended)
```bash
# Make a separate venv for the validator because of pydantic version conflict
python -m venv validator_venv
validator_venv/bin/pip install -e .
validator_venv/bin/python neurons/validator.py --wallet.name WALLET_NAME --wallet.hotkey WALLET_HOT_NAME --use-local-validation-api
# Run model queue to push models to validation api to be evaluated
validator_venv/bin/python neurons/model_queue.py --use-local-validation-api
```

## Subnet Incentive Mechanism

The general structure of the incentive mechanism is as follows:
1. Every miner has a model registered per UID
2. Each miner's model submission is scored, with details outlined below
   - The scoring mechanism is constantly evolving according to SOTA model bechmark data

3. The validator compares each miner's score against all the other miners, and calculates a win rate
    - Note that there are some modifiers for a miner's score such as their submission age in relation to other miner submissions (aka time penalty) to combat blatant model copying
4. Given each miner's win rate, weights are assigned sorted by highest win rate


### Model Evaluation Criteria
### Model Size
A smaller model will score higher than a big model. Model size is the disk space occupied by the model repo from HF. The max model size is limited to 72GB.

<!-- $S_{size} = 1 - ModelSize/ MaxModelSize$ -->
### Latency
A faster model will score higher than a slow model.

### Output Similarity
Evaluated against datasets, a model that generates similiar resposne to groundtruth will score higher. There is an additional creativity component that utilizes the 

### Vibe Matching
A model that can generate outputs with similiar length to its inputs will score higher.

### Post Evaluation 
After initial evaluation, a model will be selected for post evaluation after some time. The current process for this is a proprietary solution that is based on judging criteria from SOTA model benchmarking approaches. In the future, the details for this will be available on https://research.dippy.ai


## Acknowledgement

Our codebase is built upon [Nous Research's](https://github.com/NousResearch/finetuning-subnet) and [MyShell's](https://github.com/myshell-ai/MyShell-TTS-Subnet?tab=readme-ov-file) Subnets.

## License

The Dippy Bittensor subnet is released under the [MIT License](./LICENSE).


# Project Structure Overview

## Core Components

### 1. Main Application
- `neurons/` - Core neural network components
  - `miner.py` - Mining node implementation
  - `validator.py` - Validation node implementation
  - `model_queue.py` - Queue management for model processing

### 2. Model Management
- `model/` - Model-related functionality
  - `data.py` - Data structures and model definitions
  - `scores.py` - Scoring system implementation

### 3. Validation API
- `dippy_validation_api/` - API for model validation. Only validators and subnet operators require usage of this API. Miners do not need to set this up in 99% of cases

### 4. Utilities
- `utilities/` - Common utility functions
  - `repo_details.py` - Repository management utilities
  - `validation_utils.py` - Validation helper functions

### 5. Documentation
- `docs/` - Project documentation
  - `miner.md` - Miner setup and usage guide
  - `validator.md` - Validator setup and usage guide
  - `FAQ.md` - Frequently asked questions

## Configuration Files
- `pyproject.toml` - Project metadata and dependencies
- `requirements.txt` - Main project dependencies
- `requirements.api.txt` - Validation API dependencies
- `requirements.miner.txt` - Miner-specific dependencies
- `requirements.eval.txt` - Evaluation-specific dependencies used for docker based evaluation
- `min_compute.yml` - Minimum compute requirements specification

## Docker Configuration
- `evaluator.Dockerfile` - Docker configuration for evaluator
- `dippy_validation_api/vapi.Dockerfile` - Docker configuration for validation API
