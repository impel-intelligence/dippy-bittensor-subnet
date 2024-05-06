<div align="center">

# Dippy Subnet: Creating The World's Best Open-Source Roleplay LLM <!-- omit in toc -->
[![DIPPY](/assests/Dippy.png)](https://dippy.ai)
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

The Dippy Roleplay subnet on Bittensor aims to create the world's best open-source roleplay LLM by leveraging the collective efforts of the open-source community. This subnet addresses the critical issue of loneliness, which affects a significant portion of the population and is linked to various mental and physical health problems. 

Current SOTA LLMs are designed for the assistant use case and lack the empathetic qualities necessary for companionship. While some companies (c.ai, chai, inflection etc) have developed closed-source roleplay LLMs, the open-source alternatives lag significantly behind in performance. 

Our team at Impel Intelligence Inc. knows this issue intimately through building Dippy, a proactive AI companion app for iOS.  In this subnet, we will bring together the entire open-source eco-system to build the world’s best roleplay LLM.

## Roadmap

Given the complexity of creating a state of the art roleplay LLM, we plan to divide the process into 3 distinct phases.

**Phase 1:** 
- [ ] Subnet launch with robust pipeline for roleplay LLM evaluation on public datasets and response length 
- [ ] New, evolving evaluation datasets curated by community as well as contributed by Dippy's mobile app users
- [ ] Public model leaderboard based on evaluation criteria

**Phase 2:** 
- [ ] Multiple TAO themed characters introduced in the app with different personalities (funny, romantic, therapeutic etc)
- [ ] Top models rotated among characters and evaluated in the app based on user metrics (engagement time, conversation length, retention etc)
- [ ] Models with the highest score in each personality type are chosen as "expert" models and made publicly available

**Phase 3:** 
- [ ] New Mixture of Experts model made as a baseline based on the "expert" models chosen from Phase 2
- [ ] Robust pipeline to evaluate new MOE model submissions against all the characters in the app
- [ ] Expand the state of the art in roleplay LLMs through continuous iteration and data collection

## Overview of Miner and Validator Functionality

![overview](/assests/model_architecture.png)

**Miners** would use existing frameworks, fine tuning techniques, or MergeKit, to train, fine tune, or merge models to create a unique roleplay LLM. These models would be submitted to a shared Hugging Face pool. 

**Validators** would evaluate the and assess model performance via our protocol and rank the submissions based on various metrics (empathy, conciseness etc). We will provide a suite of 
testing and benchmarking protocols with state-of-the-art datasets.



## Running Miners and Validators
### Running a Miner

#### Requirements
- Python 3.8+
- GPU with at least 24 GB of VRAM

#### Setup
To start, clone the repository and `cd` to it:
```
git clone https://github.com/impel-intelligence/dippy-bittensor-subnet.git
cd dippy-bittensor-subnet
pip install -e .
```
#### Submitting a model
As a miner, you're responsible for leveraging all methods available at your disposal, including but not limited to training new models, merging existing models (we recommend [MergeKit](https://github.com/arcee-ai/mergekit)), finetuning existing models, and so on to push roleplay LLMs forward.

We outline the following criteria for Phase 1:

- Models should be 7B-13B in size
- Models shouldn't be quantized, just submit safetensors

Once you're happy with the performance of the model for the usecase of Roleplay, you can simply submit it to Hugging Face 🤗 and then use the following command:

```
python3 dippy_subnet/upload_model.py --hf_repo_id HF_REPO --wallet.name WALLET  --wallet.hotkey HOTKEY --subtensor.network test --chat_template MODEL_CHAT_TEMPLATE --model_dir PATH_TO_MODEL   
```


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
python neurons/validator.py --subtensor.network test --wallet.name WALLET_NAME --wallet.hotkey WALLET_HOT_NAME
```

### Running the model evaluation api

### Requirements
- Python 3.9+
- Linux

#### Setup
To start, clone the repository and `cd` into it:
```
git clone https://github.com/impel-intelligence/dippy-bittensor-subnet.git
cd dippy-bittensor-subnet
python3 -m venv venv
source venv/bin/activate
pip install -e .
pip install -r requirements_val_api.txt
```

#### Run
```
cd dippy_validation_api
chmod +x start_validation_service.sh
./start_validation_service.sh
```

#### Stop
```
chmod +x kill_validation_api.sh
./kill_validation_api.sh
```

## Model Evaluation Criteria
### Model Size
A smaller model will score higher than a big model. Model size is the disk space occupied by the model repo from HF. The max model size is limited to 18GB.

<!-- $S_{size} = 1 - ModelSize/ MaxModelSize$ -->
### Latency
A faster model will score higher than a slow model.

### Output Similarity
Evaluted against datasets, a model that generates similiar resposne to groundtruth will score higher.

### Vibe Matching
A model that can generate outputs with similiar length to its inputs will score higher.

## Acknowledgement

Our codebase is built upon [Nous Research's](https://github.com/NousResearch/finetuning-subnet) and [MyShell's](https://github.com/myshell-ai/MyShell-TTS-Subnet?tab=readme-ov-file) Subnets.

## License

The Dippy Bittensor subnet is released under the [MIT License](./LICENSE).
