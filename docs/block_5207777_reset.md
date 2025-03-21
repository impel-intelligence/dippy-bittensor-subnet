# Block Era 5207777 Reset Information for Miners

⚠️ **CRITICAL WARNING** ⚠️

As repeatedly mentioned before, volatility in scoring is expected. As a miner, you should ideally only submit when you have verified for yourself the potential score and behavior of your model. 

## 1. What's happening?
The subnet is making drastic changes in order to encourage qualitatively better model submissions. Previous methods of scoring (evaluation score, coherency score, vibe score, etc) have been eliminated.
The main criteria for this epoch is to determine the efficacy of LLMs-as-judges and the associated judge score.
For more details on the judge score, see the dedicated [scoring](/docs/llm_scoring.md) page for details.

## 2. What are the changes?

The changes are the following:
1. There is now only one score - the judge score
2. Additional _temporary_ workers will be utilized to manage the incoming flux of models
3. Any model submissions submitted before block 5207777 will automatically receive a score of 0
