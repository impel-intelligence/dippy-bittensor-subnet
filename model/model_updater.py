import bittensor as bt
from typing import Optional
from constants import CompetitionParameters, COMPETITION_SCHEDULE
import constants
from model.data import ModelMetadata, Model
from model.model_tracker import ModelTracker
from model.storage.local_model_store import LocalModelStore
from model.storage.model_metadata_store import ModelMetadataStore
from model.storage.remote_model_store import RemoteModelStore


class ModelUpdater:
    """Checks if the currently tracked model for a hotkey matches what the miner committed to the chain."""

    def __init__(
        self,
        metadata_store: ModelMetadataStore,
        remote_store: RemoteModelStore,
        local_store: LocalModelStore,
        model_tracker: ModelTracker,
    ):
        self.metadata_store = metadata_store
        self.remote_store = remote_store
        self.local_store = local_store
        self.model_tracker = model_tracker
        self.min_block: Optional[int] = None

    def set_min_block(self, val: Optional[int]):
        self.min_block = val

    @classmethod
    def get_competition_parameters(cls, id: str) -> Optional[CompetitionParameters]:
        for x in COMPETITION_SCHEDULE:
            if x.competition_id == id:
                return x
        return None

    async def _get_metadata(self, hotkey: str) -> Optional[ModelMetadata]:
        """Get metadata about a model by hotkey"""
        return await self.metadata_store.retrieve_model_metadata(hotkey)

    async def sync_model(self, hotkey: str) -> bool:
        """Updates local model for a hotkey if out of sync and returns if it was updated."""
        # Get the metadata for the miner.
        metadata = await self._get_metadata(hotkey)

        if not metadata:
            bt.logging.trace(
                f"No valid metadata found on the chain for hotkey {hotkey}"
            )
            return False

        if self.min_block and metadata.block < self.min_block:
            bt.logging.trace(
                f"Skipping model for {hotkey} since it was submitted at block {metadata.block} which is less than the minimum block {self.min_block}"
            )
            return False

        # Backwards compatability for models submitted before competition id added
        if metadata.id.competition_id is None:
            metadata.id.competition_id = constants.ORIGINAL_COMPETITION_ID

        parameters = ModelUpdater.get_competition_parameters(metadata.id.competition_id)
        if not parameters:
            bt.logging.trace(
                f"No competition parameters found for {metadata.id.competition_id}"
            )
            return False

        # Check what model id the model tracker currently has for this hotkey.
        tracker_model_metadata = self.model_tracker.get_model_metadata_for_miner_hotkey(
            hotkey
        )
        if metadata == tracker_model_metadata:
            return False
        bt.logging.warning(f"Syncing model for hotkey {hotkey}")
        # Get the local path based on the local store to download to (top level hotkey path)
        # Update the tracker
        self.model_tracker.on_miner_model_updated(hotkey, metadata)
        bt.logging.warning(f"Model for hotkey {hotkey} updated to {metadata}")
        return True
