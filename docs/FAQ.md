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
   - A small amount of the total score is based on how well your model's output length matches that of the user
6. Coherence Score  
   - Your model will generate a conversation based on augmented data from https://huggingface.co/datasets/proj-persona/PersonaHub. This output is then compared with gpt-4o to evaluate if your model is coherent enough. This is a binary factor, meaning that if your model is not coherent enough, you will automatically be scored 0
8. Post Evaluation Score  
   - As part of efforts to more closely align with industry standard benchmarks, we are introducing a score multiplier that will be run post evaluation. The scoring for this can be considered arbitrary for now, until a more formal score attribute is added
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

## I submitted my model some time ago and it still hasn’t shown up in the queue.

There are some troubleshooting steps a miner can take to ensure that their submission was managed successfully:
1. Ensure that the commit data to the bittensor network is correct. 
1a. A service such as taostats, or fetching raw block data from a subtensor node, can provide specific evidence. 
1b. Example of a successful `commit` call to the bittensor network for this subnet can be seen here: https://taostats.io/extrinsic/3997404-0010
2. Run the sample validator step mentioned in the `miner.md` documentation to check the specific status of your entry as reported by validators
3. If all else fails, create a detailed github issue with the specific details of the miner entry (hotkey, uid, registration time) and the team will investigate if necessary

## My entry failed. Can I get another chance?

Short answer: No .
Long answer:

There are (generally) 3 types of failed entries:
1. Repository check failure: the model repository must always be public for evaluation - if not, the miner can always resubmit a public one (note that sometimes huggingface can cache repos that converted from private to public, so be aware of this when submitting)

2. Miner skill issue: the model submitted is not a model that can actually be run (miner may have uploaded corrupted data, a model not compatible with scoring, etc)

3. Error in scoring worker: the model submitted is valid, but some issue in the scoring worker runtime has resulted in a failure (ie cuda out of memory, etc)

For 1 and 2 these will never be re-evaluated as it is the miner's responsibility to ensure that the model will be public and loaded correctly. Given that it is possible for a miner to run the same scoring code, any deterministic failures in this case will stay failed.

For 3, this is a rare but possible occurence. For these cases, there is a separate queue that runs _only_ after _all other_ models have been evaluated. 


## I have a specific question about my miner. Where can I get answers?

Assuming you are encountering a technical issue that cannot be answered from the existing documentation, the fastest way to receive a response is through Github issues for this project. Any other attempts (DMs, ping, etc) will not be prioritized. If the answer to your question is easily found in the existing documentation or code, your question will be ignored or closed.


## I heard there's a scoring reset. What do I have to do?
Scoring resets are a frequent and expected occurrence in this subnet. Some general rules which apply unless explicitly overruled:

1. Miners must re-submit their models after a scoring reset. This is trivial to accomplish
2. Emissions will shift dramatically following a scoring reset. For the first 24 to 48 hours, extreme volatility is expected.
3. Given the increased amount of miner activity after a scoring reset, models may take longer to be evaluated. Again, as mentioned, extremem volatility is expected and there is no downside for a miner to wait until later to register if they feel that it is too risky to submit a current entry.

