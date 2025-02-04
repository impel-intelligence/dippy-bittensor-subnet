# Block Era 4850000 Reset Information for Miners


1. What's happening?
The subnet is making drastic changes in order to encourage qualitatively better model submissions.

2. What are the changes?

The changes are the following:
1. Models must now be at least 49GB (approx. 20b+ parameters)
2. There are now two experimental multipliers: judge score and post eval score
2a. Judge score uses LLM as a judge to benchmark generated text versus 
2b. Post eval score runs an internal LLM similarity check
3. The vibe score has been removed


3. What can I expect?
This is a critical _experimental_ update that changes the dynamics of the subnet. Communication for this change has been mentioned repeatedly in the past month. 

1. Volatility in scoring is expected. This will stabilize over time
2. Queue times are expected to be long given the larger size of models. Default immunity period is more than 24 hours which is an order of magnitude longer than most subnets
3. Additional scoring changes are expected in the near future, and if you as a miner are not willing to risk, then it may be better to wait for a newer competition period

4. What is the judge score?

The judge score is the first step in creating a more comprehensive scoring method for roleplay LLMs. 
The difficulty here is that unlike other domains such as math or science, it is difficult to create a baseline objective scoring method for roleplay. In addition, human scoring is obviously extremely limited, hence utilizing larger LLMs as judges provides a way to scale for larger datasets.

In the first iteration, we utilize a reference dataset that is compared against a model's generated text. The current version uses the existing dippy synthetic dataset, but with further subsamples of the conversation. Although optimization for this can theoretically conflict with the existing evaluation score, we leave this as intentional so as to prevent overfitting on the current implementation of the judge score.
(The specific details of this code can be seen right here in this repository)
The LLM as a judge compares both texts according to the following criteria / scoring matrix:

```

{
            "realism_win": (string : "original" or "generated" or "tie"),
            "entertainment_win": (string : "original" or "generated" or "tie"),
            "coherency_win": (string : "original" or "generated" or "tie"),
            "realism_win_reasoning": (string),
            "entertainment_win_reasoning": (string),
            "coherency_win_reasoning": (string)
}

```

Note that in this current version, we will consider "generated" entries to contribute to a model's judge score. Any "tie" entries will be considered neutral. 
The final judge score is a simple sum of the model's wins against the dataset, represented as a score between 0 and 1.

As mentioned, there are known pitfalls and potential weaknesses for this specific implementation. Thus, we consider this scoring period to be experimental and expect drastic changes in the future.


⚠️ **CRITICAL WARNING** ⚠️

As repeatedely mentioned above, volatility in scoring is expected. As a miner, you should ideally only submit when you have verified for yourself the potential score and behavior of your model. 
