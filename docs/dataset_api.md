# Dataset API Access

The evaluation step of miner model scoring involves fetching a dataset from a managed API. 
The dataset API is not meant for frequent use, and is ideally only used for official validation by validators. 
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

### 4. How often the StreamedSyntheticDataset changes and at what time?
The dataset API provide a sample of `4096` conversations, 50% which is fetched from historical dates (aka what is available in HuggingFace) and 50% from the past 48 hours. 
The purpose of the dataset API is to provide a protected stream of recently generated data, as HuggingFace currently does not support streaming updates at this time.




