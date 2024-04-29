"""A script that pushes a model from disk to the subnet for evaluation.

Usage:
    python scripts/upload_model.py --load_model_dir <path to model> --hf_repo_id my-username/my-project --wallet.name coldkey --wallet.hotkey hotkey

Prerequisites:
   1. HF_ACCESS_TOKEN is set in the environment or .env file.
   2. load_model_dir points to a directory containing a previously trained model, with relevant ckpt file named "checkpoint.pth".
   3. Your miner is registered
"""

import asyncio
import json
import os
import argparse
import torch
import constants
from model.storage.hugging_face.hugging_face_model_store import HuggingFaceModelStore
from model.model_updater import ModelUpdater
import bittensor as bt
from utilities import utils
from model.data import Model, ModelId
from model.storage.chain.chain_model_metadata_store import ChainModelMetadataStore
from huggingface_hub import update_repo_visibility
import time
import hashlib

from dotenv import load_dotenv

load_dotenv()


def get_config():
    # Initialize an argument parser
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--hf_repo_id",
        type=str,
        help="The hugging face repo id, which should include the org or user and repo name. E.g. jdoe/finetuned",
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        help="The director of the model to load.",
    )

    parser.add_argument(
        "--chat_template",
        type=str,
        default="vicuna",
        help="The chat template for the model.",
    )

    parser.add_argument(
        "--netuid",
        type=str,
        default=constants.SUBNET_UID,
        help="The subnet UID.",
    )
    parser.add_argument(
        "--competition_id",
        type=str,
        default=constants.ORIGINAL_COMPETITION_ID,
        help="competition to mine for (use --list-competitions to get all competitions)",
    )
    parser.add_argument(
        "--list_competitions", action="store_true", help="Print out all competitions"
    )

    parser.add_argument(
        "--model_commit_id", type=str, help="The commit id of the model to upload"
    )

    parser.add_argument(
        "--skip_model_upload", action="store_true", help="Skip model upload"
    )

    # Include wallet and logging arguments from bittensor
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)

    # Parse the arguments and create a configuration namespace
    config = bt.config(parser)
    return config

def regenerate_hash(namespace, name, chat_template, competition_id):
    s = " ".join([namespace, name, chat_template, competition_id])
    hash_output = hashlib.sha256(s.encode('utf-8')).hexdigest()
    return int(hash_output[:16], 16)  # Returns a 64-bit integer from the first 16 hexadecimal characters


def check_model_dir(model_dir):
    """Check if model dir has all the required files."""
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory {model_dir} not found.")

    ls_dir = os.listdir(model_dir)
    # check if at least 1 *.safetensors file exists
    if not any(file.endswith(".safetensors") for file in ls_dir):
        raise FileNotFoundError(
            f"No *.safetensors file found in model directory {model_dir}."
        )
    
    # check if tokenizer.json exists
    if not any(file.endswith("tokenizer.json") for file in ls_dir):
        raise FileNotFoundError(
            f"No tokenizer.json file found in model directory {model_dir}."
        )
    
    # check if config.json exists
    if not any(file.endswith("config.json") for file in ls_dir):
        raise FileNotFoundError(
            f"No config.json file found in model directory {model_dir}."
        )
    
    # # check if generation_config.json exists
    # if not any(file.endswith("generation_config.json") for file in ls_dir):
    #     raise FileNotFoundError(
    #         f"No generation_config.json file found in model directory {model_dir}."
    #     )
    
    # check if special_tokens_map.json exists
    if not any(file.endswith("special_tokens_map.json") for file in ls_dir):
        raise FileNotFoundError(
            f"No special_tokens_map.json file found in model directory {model_dir}."
        )
    
    # check if model.safetensors.index.json exists
    if not any(file.endswith("model.safetensors.index.json") for file in ls_dir):
        raise FileNotFoundError(
            f"No model.safetensors.index.json file found in model directory {model_dir}."
        )
    
    # check if this file contains metadata.total_size
    # with open(os.path.join(model_dir, "model.safetensors.index.json"), "r") as f:
    #     index = json.load(f)
    #     if "metadata" not in index or "total_size" not in index["metadata"]:
    #         raise FileNotFoundError(
    #             f"model.safetensors.index.json file in model directory {model_dir} does not contain metadata.total_size."
    #         )


async def main(config: bt.config):
    # Create bittensor objects.
    bt.logging(config=config)

    wallet = bt.wallet(config=config)
    subtensor = bt.subtensor(config=config)
    bt.logging.debug("Subtensor network: ", subtensor.network)
    metagraph: bt.metagraph = subtensor.metagraph(config.netuid)

    # Make sure we're registered and have a HuggingFace token.
    utils.assert_registered(wallet, metagraph)

    # Get current model parameters
    parameters = ModelUpdater.get_competition_parameters(config.competition_id)
    if parameters is None:
        raise RuntimeError(
            f"Could not get competition parameters for block {config.competition_id}"
        )

    repo_namespace, repo_name = utils.validate_hf_repo_id(config.hf_repo_id)
    model_id = ModelId(
        namespace=repo_namespace,
        name=repo_name,
        chat_template=config.chat_template,
        competition_id=config.competition_id,
    )

    if not config.skip_model_upload:
        model = Model(id=model_id, local_repo_dir=config.model_dir)

        check_model_dir(config.model_dir)

        remote_model_store = HuggingFaceModelStore()

        bt.logging.info(f"Uploading model to Hugging Face with id {model_id}")


        model_id_with_commit = await remote_model_store.upload_model(
            model=model,
            competition_parameters=parameters,
        )
    else:
        # get the latest commit id of hf repo
        model_id_with_commit = model_id
        if config.model_commit_id is None or config.model_commit_id == "":
            raise ValueError("model_commit_id should not be set when skip_model_upload is set to True")
        
        model_id_with_commit.commit = config.model_commit_id

    model_id_with_hash = ModelId(
        namespace=repo_namespace,
        name=repo_name,
        chat_template=config.chat_template, 
        hash=regenerate_hash(repo_namespace, repo_name, config.chat_template, config.competition_id), 
        commit=model_id_with_commit.commit,
        competition_id=config.competition_id,
    )
    
    bt.logging.info(
        f"Model uploaded to Hugging Face with commit {model_id_with_hash.commit} and hash {model_id_with_hash.hash}"
    )

    model_metadata_store = ChainModelMetadataStore(
        subtensor=subtensor, wallet=wallet, subnet_uid=config.netuid
    )

    # We can only commit to the chain every n minutes, so run this in a loop, until successful.
    while True:
        try:
            update_repo_visibility(
                model_id.namespace + "/" + model_id.name,
                private=False,
                token=os.getenv("HF_ACCESS_TOKEN"),
            )
            await model_metadata_store.store_model_metadata(
                wallet.hotkey.ss58_address, model_id_with_hash
            )
            bt.logging.success("Committed model to the chain.")
            break
        except Exception as e:
            bt.logging.error(f"Failed to advertise model on the chain: {e}")
            bt.logging.error("Retrying in 120 seconds...")
            time.sleep(120)


if __name__ == "__main__":
    # Parse and print configuration
    config = get_config()
    if config.list_competitions:
        bt.logging.info(constants.COMPETITION_SCHEDULE)
    else:
        bt.logging.info(config)
        asyncio.run(main(config))
