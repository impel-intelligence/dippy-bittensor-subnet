import bittensor as bt
from bittensor.extrinsics.serving import get_metadata
import asyncio
from model.data import ModelId
from model.storage.chain.chain_model_metadata_store import ChainModelMetadataStore
import constants
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--hotkey",
    type=str,
    help="The hotkey of the model to check",
)
bt.subtensor.add_args(parser)
args = parser.parse_args()
config = bt.config(parser)

subtensor = bt.subtensor(config=config)
subnet_uid = constants.SUBNET_UID
metagraph = subtensor.metagraph(subnet_uid)

wallet = None
model_metadata_store = ChainModelMetadataStore(subtensor, subnet_uid, wallet)

model_name = asyncio.run(model_metadata_store.retrieve_model_metadata(args.hotkey))

print(model_name)