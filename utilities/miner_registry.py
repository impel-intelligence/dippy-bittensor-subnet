from typing import Any, Dict, Optional

from pydantic import BaseModel, Field
import sys
from common.data import ModelId


class MinerEntry(BaseModel):
    block: int = Field(default=sys.maxsize, description="The block number")
    hotkey: str = Field(default="", description="The hotkey of the miner")
    invalid: bool = Field(default=False, description="invalidity of determining score")
    miner_model_id: Optional[ModelId] = Field(default=None, description="The model_id of the miner")
    safetensors_model_size: int = Field(default=0, description="The safetensors model size according to huggingface")
    total_score: float = Field(default=0)


def fetch_all_commits():
    import bittensor as bt
    client = bt.subtensor()
    sub = client.substrate
    result = sub.query_map(
            module="Commitments",
            storage_function="CommitmentOf",
            params=[11],
            block_hash=None ,
        )
    commitments = {}
    for key, value in result:
            hotkey = key.value
            commitment_info = value.value.get("info", {})
            fields = commitment_info.get("fields", [])
            if not fields or not isinstance(fields[0], dict):
                continue

            field_value = next(iter(fields[0].values()))
            if field_value.startswith("0x"):
                field_value = field_value[2:]

            try:
                concatenated = bytes.fromhex(field_value).decode("utf-8").strip()
                print(f"concated {hotkey} : {concatenated} ")

                commitments[hotkey] = concatenated
            except Exception as e:
                print(f"Failed to decode commitment for hotkey {hotkey}: {e}")
                continue
    return commitments
