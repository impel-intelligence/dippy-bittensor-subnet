# Miner FAQ

## I'm new to bittensor. Can you please explain how this subnet works for miners?
This subnet's goal is to facilitate the creation of an open source LLM that is specialized for the use case of roleplay.
For miners specifically, the subnet's intended purpose is to reward miners that submit public LLMs that match the subnet's scoring criteria.

## What will I need to effectively mine on this subnet?
Objective requirements are:
1. Access to a capable GPU for model training an evaluation (recommended A100 80GB)
2. Enough tao for the subnet registration fee (baseline is 0.15 tao)  
3. Reading comprehension  
4. Intimate knowledge of bittensor network operations
Subjective requirements:
1. Ability to understand the code in this subnet
2. Python experience
3. LLM training experience

## How exactly is my miner evaluated?
There are three general scoring phases:
1. Evaluation
   1. The model responses are graded against a streaming dataset. At certain intervals, the previous day data is dumped into https://huggingface.co/datasets/DippyAI/dippy_synthetic_dataset for reference. Note that the dataset if there to provide reference, but the actual evaluation is done on a sample that is generated in realtime. 
   2. Creativity is a score that modifies the above evaluation score. The less creative your model is (aka overfit), the less total score is achievable for evaluation.
   3. A small amount of the total score takes into account model size, latency, etc. The details can be seen in the `scoring` folder
4. Vibe Score
   5. A small amount of the total score is based on how well your model's output length matches that of the user
6. Coherence Score
   7. Your model will generate a conversation based on augmented data from https://huggingface.co/datasets/proj-persona/PersonaHub. This output is then compared with gpt-4o to evaluate if your model is coherent enough. This is a binary factor, meaning that if your model is not coherent enough, you will automatically be scored 0

Once your model is scored, it is compared against other submitted models to create a win rate. Note that to discourage model copying, there is a time penalty for newer models that can reduce your overall win rate.

## I submitted my model. When will I get my rewards?

Rewards are based upon a combination of your model being scored and your model win rate. 
The subnet's epoch time is 360 blocks, or roughly every 30 minutes. This means that once your model is fully scored, it will take at least one epoch if not another to fully allocate incentive. The bittensor network itself may also vary in its payout of miner emissions. 

## My model has a high score but I'm not receiving enough emissions. Why?

Note that your score by itself is not the metric that is used to provide incentive, it is your win rate that determines your emissions. To discourage model copying, there is a time penalty applied for newer models. 
This means that ideally, your model should score significantly better than the current top scoring model. It is possible to calculate your score before you submit for evaluation, so be sure to check in case of deregistration.

## Why did I get deregistered?

Deregistration is inherently caused when a new participant registers, this is behavior that happens across all subnets.
When a new participant registers, the worst performing uid (aka with lowest incentive) will become deregistered. 
If there exists more than 1 uid with 0 incentive, then there is a specific order of deregistration, usually based on the age of the registration.
If your win rate is only average at best, then it is likely you will not receive any significant incentive and will therefore be deregistered according to the protocol.
As a miner, you can also run a pseudo validator to verify if validators are setting weights properly.

## How can I contribute to this repository?
1. Create an issue
2. Submit a PR referencing the issue

Note that to prevent favoring miners over others, we will prioritize public discourse over DMs / private messages.

