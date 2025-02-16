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
3. LLM training /fine tuning experience

## How exactly is my miner evaluated?
There are three general scoring phases:
1. Evaluation
   1. The model responses are graded against a streaming dataset. At certain intervals, the previous day data is dumped into https://huggingface.co/datasets/DippyAI/dippy_synthetic_dataset_v1 for reference. Note that the dataset is there to provide reference, but the actual evaluation is done on a sample that is generated in realtime. 
   2. Creativity is a score that modifies the above evaluation score. The less creative your model is (aka overfit), the less total score is achievable for evaluation.
   3. A small amount of the total score takes into account model size, latency, etc. The details can be seen in the `scoring` folder
5. Judge Score
   - This is an experimental score that acts as a multiplier. If the generated model, according to the LLM judge, wins or ties more than a certain threshold (at the time of this writing this is 30% but subject to change in the future), a significant score boost will be added.
6. Coherence Score  
   - Your model will generate a conversation based on augmented data from https://huggingface.co/datasets/proj-persona/PersonaHub. This output is then compared with gpt-4o to evaluate if your model is coherent enough. This is a binary factor, meaning that if your model is not coherent enough, you will automatically be scored 0
8. Post Evaluation Score  
   - As part of efforts to more closely align with industry standard benchmarks, we are introducing an experimental score multiplier that will be run post evaluation. The scoring for this can be considered arbitrary for now, until a more formal score attribute is added
Once your model is scored, it is compared against other submitted models to create a win rate. Note that to discourage model copying, there is a time penalty for newer models that can reduce your overall win rate.

## How does the chat template integrate with my model submission?
Different classes of LLMs use different systems for tokens, and for instruct models this is much more important. In particular, LLMs can use different stop tokens or asssistant syntax to provide the syntax around a "chat" session. For popular foundational models and other common templates, we have provided them in jinja format in scoring/prompt_templates.


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

In the case that there seems to be an issue with a miner submitted model that may be a deeper technical issue, please provide the following:
1. A github issue with the appropriate title
2. Steps to reproduce the failure

## How can I check the detailed status of my model?
Assuming that you know your model submission parameters, you can use the following curl request for reference:
```
curl -X GET "https://dippy-bittensor-subnet.com/model_submission_details?repo_namespace=my-org&repo_name=my-model&chat_template_type=chatml&hash=12345678&competition_id=d1"
```

## I have a specific question about my miner. Where can I get answers?

Assuming you are encountering a technical issue that cannot be answered from the existing documentation, the fastest way to receive a response is through Github issues for this project. Any other attempts (DMs, ping, etc) will _not_ be prioritized. If the answer to your question is easily found in the existing documentation or code, your question will be ignored or closed.

If you are asking questions on the discord channel, you _must_ end your response with the character 🫘 (subject to change over time). This will provide proof that you have taken the time to read the documentation carefully.


## I heard there's a scoring reset. What do I have to do?
Scoring resets are a frequent and expected occurrence in this subnet. Some general rules which apply unless explicitly overruled:

1. Miners must re-submit their models after a scoring reset. This is trivial to accomplish
2. Emissions will shift dramatically following a scoring reset. For the first 24 to 48 hours, extreme volatility is expected.
3. Given the increased amount of miner activity after a scoring reset, models may take longer to be evaluated. Again, as mentioned, extremem volatility is expected and there is no downside for a miner to wait until later to register if they feel that it is too risky to submit a current entry.

## What's up with the leaderboard?

The leaderboard is a secondary view of the miner state. Note that due to its implementation, it is not guaranteed that the leaderboard state will reflect the same state as on the metagraph. For miners, the source of truth should always be the ranking determined from running the validator script (it is possible to run the script without having to be a validator). 

During score resets, it is common for the state of the leaderboard to be in flux until all stale entries are removed.

## I see a 502 error when calling the dataset API. 
Note that there are two versions of the dataset API at this time:
1. Dataset API for validators
2. Dataset API for miners
As of 2025 Feb 16, there is a tempoary dataset API for miners available at https://temp-miner-dataset-sn11.dippy-bittensor-subnet.com/dataset .
Note that this API may experience intermittent downtime as it is a publicly accessible resource 

Regarding the dataset API for validators: note that there may occassionally be network issues that can interfere with scoring. The current system will automatically purge entries related to these errors and requeue accordingly. Note that this is a temporary solution and is subject to change and improvements over time.
