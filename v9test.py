import bittensor as bt
import asyncio
from bittensor.core.chain_data import (
    decode_account_id,
)
from typing import cast


def extract_raw_data(data):
    try:
        fields = data.get('info', {}).get('fields', ())
        
        if fields and isinstance(fields[0], tuple) and isinstance(fields[0][0], dict):
            raw_dict = fields[0][0]
            raw_key = next((k for k in raw_dict.keys() if k.startswith('Raw')), None)
            
            if raw_key and raw_dict[raw_key]:
                numbers = raw_dict[raw_key][0]
                result = ''.join(chr(x) for x in numbers)
                return result
                
    except (IndexError, AttributeError):
        pass
    
    return None



async def main():
    # Initialize subtensor
    subtensor = bt.subtensor()
            
    raw_commmitments = subtensor.query_map(
                        module="Commitments",
                        name="CommitmentOf",
                        params=[11])
    for key, value in raw_commmitments:
        try:
            hotkey = decode_account_id(key[0])
            body = cast(dict, value.value)
            chain_str = extract_raw_data(body)
            print(f"hotkey {hotkey} chain_str {chain_str} block {body['block']}")
        except Exception as e:
            print(e)

# Run the async function
if __name__ == "__main__":
    asyncio.run(main())


