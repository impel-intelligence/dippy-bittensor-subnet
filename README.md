<div align="center">

# Dippy Subnet: Creating The World's Best Open-Source Roleplay LLM <!-- omit in toc -->
[![DIPPY](/Dippy.png)](https://dippy.ai)
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

Given the complexity of creating a SOTA Roleplay LLM, we plan to divide the process into 3 distinct phases.

**Phase 1:** 
We’ll launch our subnet with a robust pipeline for roleplay LLM evaluation on both public and proprietary dataset. 
We rank models on a public dashboard on based on our evaluation protocol

**Phase 2:** 
We will create Bittensor themed characters within the Dippy app and randomly serve the top miner submitted models to measure model quality based on real user engagement and retention. 
The app will be a playground to evaluate and rank miner model submissions

**Phase 3:** 
The best models will be integrated into an open-sourced MOE LLM, with a unique recommendation mechanism that selects the best “expert” suited for the character based on personality, characteristics etc.

## Overview of Miner and Validator Functionality

![overview](/drawio.png)

**Miners** would use existing frameworks, fine tuning techniques, or MergeKit, to train, fine tune, or merge models to create a unique roleplay LLM. These models would be submitted to a shared Hugging Face pool. 

**Validators** would evaluate the and assess model performance via our protocol and rank the submissions based on various metrics (empathy, conciseness etc). We will provide a suite of 
testing and benchmarking protocols with state-of-the-art datasets.



## Running Miners and Validators
### Running a Miner
python dippy_subnet/check_model.py --hotkey HOTKEY --subtensor.network test 
#### Requirements

#### Setup

### Running a Validator
python neurons/validator.py --subtensor.network test --wallet.name WALLET_NAME --wallet.hotkey WALLET_HOT_NAME
#### Requirements

#### Setup


## License

The Dippy Bittensor subnet is released under the [MIT License](./LICENSE).
