
# LLM Scoring Criteria

The general goal of the subnet is to incentivize the development of a roleplay model that can reliably serve millions of users worldwide. 
To accomplish this goal, the subnet aims to adjust and tune the scoring criteria as needed.

## Background
The methods of evaluating model quality have evolved quickly and drastically over time.
Initially, the subnet prioritized dataset based evaluation approaches, where model generated text was scored via accuracy to an existing roleplay dataset. 
While this proved effective for smaller models, the process of evaluating a roleplay model has quickly scaled in terms of computational requirements. 
To better manage this scale, the usage of LLMs as a judge has become more popular as a method of benchmarking and comparing model outputs.

## Dataset
The dippy team has developed a continuously generated dataset based on billions of conversations from the Dippy app. 
While other benchmarks rely on a static dataset that can be overfitted, the dippy dataset will always increase in size to combat this.
This dataset is always being improved upon in terms of both quality and conversation length.
With over 1 million conversations, the dippy dataset serves as the basis for model scoring.
The dippy dataset is developed at full cost to the dippy team. 

## Current Scoring Mechanism

### Judge Score
For scoring, `gpt-4o` is used as a judge to compare conversations based on the original dippy dataset versus conversations generated by a miner submitted model.
The current criteria compares the following:

```
- realism: The responses match the character's personality perfectly, staying true to who they are meant to be without any inconsistencies
- entertainment: The responses are engaging and interesting to read, without any repetitive or boring language
- coherency: The responses use clear and correct language throughout, with proper grammar, spelling and punctuation. The writing style stays consistent and appropriate.
```

If according to the judge, the model generated text is considered qualitatively better than the original dataset text, that counts as a "win" against the dataset.
The number of wins against the total number of evaluations comprises the judge score.
#### Judge Score Reasoning
First, let's compare against traditional methods of scoring fine tunes. Usually, there is a given dataset and a model's outputs are graded according to how accurately it can map its responses to that given dataset. 
In an adversarial environment, exploits against this type of scoring method can be easily executed since a model can be overfit. Furthermore, there is a score impact against models that might actually produce responses that are better than the dataset. Without getting into details of what specifically "better" entails, it is reasonable to say that the distribution of "better" text generation would be larger than a specific response in a dataset.
This distribution of "better" data is now limited by the dataset in the fine tuning scenario, and the effectiveness of the scoring is now compromised given that it does not adequately follow all of the potential desired responses / behavior that could be generated from a model.
To address this specific aspect, one common method that arises is the usage of a judge. Another augmentation to the use of a judge is pseudo human feedback by way of introducing real life conversations in the dataset, modified to preserve privacy.
In addition, the judge score by itself must be grounded in some reality. In the case of this subnet and Dippy, this means introducing data and behavior provided from interactions on the Dippy app.
From internal testing, models tested with the current implementation of the judge score resulted in notable improvements across the stated metrics, in addition to retention / engagement with the model. Essentially, the judge score proved effective as a proxy for a human when comparing responses, to the extent of favored model outputs resulting in an improved experience to Dippy users. However, this is an isolated environment with carefully managed outputs, and so the judge score must also be operatable in an unsupervised environment.
Given the nature of bittensor and subnet management, this approach to managing post training involves doing so not only unsupervised, but also in an adversarial environment. Thus, the current prompting is designed to be as critical against the generated dataset as much as possible. Furthermore, the prompting of the judge score must be modeled in a way so as to reduce prompt injection vulnerabilities and other potentional prompt exploits. Given that prompting in general is not an exact system, compromises to consistency in one dimension may be required in order to provide consistency in others. 
In a scenario where one model is known to produce quality outputs (regardless of its _overall_ quality) against another model that may contain adversarial elements, strategically it is acceptable to bias against the known outputs, even in the case where the inverse (exchanging generated for original text) may not result in a direct reversed score. 
This is not to say that the judge score is perfectly designed / intended. In fact, there are improvements to better model human interactions with models outside of synthetic data. For example: a human may intentionally try to antagonize a character, or attempt to steer a character outside of their intended conversational boundaries. The challenge here is to find a method of modeling these interactions given their inherent subjectivity. In fact, this property underpins most of role play model training as a whole, as there is yet a clear RL system compared to models trained for coding / mathematical purposes.
In conclusion, the judge score approach offers a significant improvement over traditional evaluation methods in adversarial environments by comparing responses rather than checking adherence to fixed datasets. Our implementation has shown practical value through improved user metrics on the Dippy platform, effectively serving as a proxy for human judgment.
While effective, this system requires ongoing refinement to better handle the subjective nature of conversational AI, especially in role-play contexts. Future development should focus on more sophisticated evaluation frameworks that can operate reliably in unsupervised environments while capturing the nuances of human-AI interaction.
The judge score methodology provides a valuable foundation for evaluating AI outputs in open systems, representing an important step toward more robust evaluation frameworks for generative AI.


### Post Evaluation 
After initial evaluation, a model will be selected for post evaluation after some time. 
The current process for this is a proprietary solution that is based on judging criteria from SOTA model benchmarking approaches. 
In the future, the details for this will be available on https://research.dippy.ai
