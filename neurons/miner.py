import argparse
import hashlib
from typing import Optional, Type

from pydantic import BaseModel, Field, PositiveInt
import bittensor as bt

from model.data import ModelId

from utilities.validation_utils import regenerate_hash

DEFAULT_NETUID = 11


def get_config():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repo_namespace",
        default="DippyAI",
        type=str,
        help="The hugging face repo id, which should include the org or user and repo name. E.g. jdoe/finetuned",
    )

    parser.add_argument(
        "--repo_name",
        default="your-model-here",
        type=str,
        help="The hugging face repo id, which should include the org or user and repo name. E.g. jdoe/finetuned",
    )

    parser.add_argument(
        "--chat_template",
        type=str,
        default="chatml",
        help="The chat template for the model.",
    )

    parser.add_argument(
        "--netuid",
        type=str,
        default=f"{DEFAULT_NETUID}",
        help="The subnet UID.",
    )
    parser.add_argument(
        "--online",
        type=bool,
        default=False,
        help="Toggle to make the commit call to the bittensor network",
    )
    parser.add_argument(
        "--model_hash",
        type=str,
        default="d1",
        help="Model hash of the submission",
    )
    # Include wallet and logging arguments from bittensor
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)

    # Parse the arguments and create a configuration namespace
    config = bt.config(parser)
    return config


def register():
    config = get_config()
    bt.logging(config=config)

    wallet = bt.wallet(config=config)
    subtensor = bt.subtensor(config=config)

    hotkey = wallet.hotkey.ss58_address
    namespace = config.repo_namespace
    repo_name = config.repo_name
    chat_template = config.chat_template
    entry_hash = str(regenerate_hash(namespace, repo_name, chat_template, hotkey))
    model_id = ModelId(
        namespace=namespace,
        name=repo_name,
        chat_template=chat_template,
        competition_id=config.competition_id,
        hotkey=hotkey,
        hash=entry_hash,
    )
    model_commit_str = model_id.to_compressed_str()

    bt.logging.info(f"Registering with the following data")
    bt.logging.info(f"Coldkey: {wallet.coldkey.ss58_address}")
    bt.logging.info(f"Hotkey: {hotkey}")
    bt.logging.info(f"repo_namespace: {namespace}")
    bt.logging.info(f"repo_name: {repo_name}")
    bt.logging.info(f"chat_template: {chat_template}")
    bt.logging.info(f"entry_hash: {entry_hash}")
    bt.logging.info(f"Full Model Details: {model_id}")
    bt.logging.info(f"Subtensor Network: {subtensor.network}")
    bt.logging.info(f"model_hash: {config.model_hash}")
    bt.logging.info(f"String to be committed: {model_commit_str}")

    try:
        netuid = int(config.netuid)
    except ValueError:
        netuid = DEFAULT_NETUID
    netuid = netuid or DEFAULT_NETUID
    if config.online:
        try:
            subtensor.commit(wallet, netuid, model_commit_str)
            bt.logging.info(f"Succesfully commited {model_commit_str} under {hotkey} on netuid {netuid}")
        except Exception as e:
            print(e)


if __name__ == "__main__":
    register()
