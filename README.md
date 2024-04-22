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

Given the complexity of creating a State of the art roleplay LLM, we plan to divide the process into 3 distinct phases.

**Phase 1:** 
- [X] Subnet launch with robust pipeline for roleplay LLM evaluation on public datasets and response length 
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
