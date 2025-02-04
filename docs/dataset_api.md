# Dataset API Access (WIP)

The evaluation step of miner model scoring involves fetching a dataset from a managed API located at `https://datasets.dippy-bittensor-subnet.com`

The dataset API is not meant for frequent use, and is ideally only used for official validation by validators. 
Every day, the subnet operators generate a new dataset of conversations that is used for evaluation. Currently, there are 4096 samples where 2048 are from the most recent stream of data (aka within the last 48 hours) and 2048 are sampled from older generated data. 
Exposure to miners is designed to provide a "pre score check" experience before a miner officially submits a model for scoring, but it NOT meant for data collection, scraping, or any other purpose.


## Fetching a token
There is a script in this folder `token_check.py` with the following instructions:

```
To run the script, use the following command:

python token.py --wallet-name <wallet_name> --wallet-hotkey <wallet_hotkey> --signer <signer_type>

Arguments:
--wallet-name: Name of the Bittensor wallet (default: "default")
--wallet-hotkey: Hotkey of the Bittensor wallet (default: "default")
--signer: Type of signer to use, either "coldkey" or "hotkey" (required)

Example:
python token_check.py --wallet-name my_wallet --wallet-hotkey my_hotkey --signer hotkey

This will run the authentication flow using the specified wallet and signer information.
```
As part of the execution of the script, there is an additional step to fetch the dataset API as a check that the token request is functional.

## Questions

### 1. How many requests can I do with one token?
The rate limit implementation is that of [token buckets](https://en.wikipedia.org/wiki/Token_bucket). 
The token bucket configuration is set to the following:
1. Unauthorized requests: Maximum burst up to 1 request , refills at a rate of 1 token per hour
2. Authorized requests: Maximum burst up to 3 requests, refills at a rate of 3 tokens per hour
3. Validators: Maximum burst up to 100 requests, refills at a rate of 50 tokens per hour


### 2. What is the difference between signers (coldkey/hotkey) and what should I choose?
The intended purpose is the following:
1. If you are a registered miner in the subnet, you can use the hotkey to validate your spot.
2. If you are an unregistered miner in the subnet, you can use the coldkey since you would not have a registered hotkey available.

At some point in the future, coldkeys will be limited to accounts with a minimum balance in order to prevent spam token creation.


### Can you explain the behavior of the dataset API?
For reference, the 
The dataset API has two official endpoints:
1. `/dataset` 
There are two required query parameters here:
- `epoch_date` : This is the beginning date for which the older generated data starts from

- `current_date` : This is the current date to start fetching recent data from

It is possible to get a better approximation of a model's score at a specific point in time via setting `current_date` to a point in the past to better gauge previous performance against current performance.

2. `authenticate`
This endpoint is used to generate a token to call the dataset API with.


### 4. How often the StreamedSyntheticDataset changes and at what time?
The dataset API provide a sample of `4096` conversations, 50% which is fetched from historical dates (aka what is available in HuggingFace) and 50% from the past 48 hours. 
The purpose of the dataset API is to provide a protected stream of recently generated data, as HuggingFace currently does not support streaming updates at this time.



