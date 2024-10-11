from typing import Any, ClassVar, Dict, Optional, Type
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from pydantic import BaseModel, Field, PositiveInt

# The maximum bytes for metadata on the chain.
MAX_METADATA_BYTES = 128
# The length, in bytes, of a git commit hash.
HOTKEY_LENGTH = 40
# The length, in bytes, of a base64 encoded sha256 hash.
SHA256_BASE_64_LENGTH = 44
# The max length, in characters, of the competition id
MAX_COMPETITION_ID_LENGTH = 2


class ModelId(BaseModel):
    """Uniquely identifies a trained model"""

    MAX_REPO_ID_LENGTH: ClassVar[int] = (
        MAX_METADATA_BYTES - HOTKEY_LENGTH - SHA256_BASE_64_LENGTH - MAX_COMPETITION_ID_LENGTH - 4  # separators
    )

    namespace: str = Field(description="Namespace where the model can be found. ex. Hugging Face username/org.")
    name: str = Field(description="Name of the model.")

    chat_template: str = Field(description="Chat template for the model.")

    # Include hotkey in hash to uniqely identify model
    hotkey: str = Field(description="Hotkey of the submitting miner.")
    # Hash is filled automatically when uploading to or downloading from a remote store.
    hash: Optional[str] = Field(description="Hash of the trained model.")
    # Identifier for competition
    competition_id: Optional[str] = Field(description="The competition id")

    def to_compressed_str(self) -> str:
        """Returns a compressed string representation."""
        return f"{self.namespace}:{self.name}:{self.chat_template}:{self.hotkey}:{self.hash}:{self.competition_id}"

    @classmethod
    def from_compressed_str(cls, cs: str) -> Type["ModelId"]:
        """Returns an instance of this class from a compressed string representation"""
        tokens = cs.split(":")
        return cls(
            namespace=tokens[0],
            name=tokens[1],
            chat_template=tokens[2] if tokens[2] != "None" else None,
            hotkey=tokens[3] if tokens[3] != "None" else "",
            hash=tokens[4] if tokens[4] != "None" else None,
            competition_id=(tokens[5] if len(tokens) >= 6 and tokens[5] != "None" else None),
        )


class Model(BaseModel):
    """Represents a pre trained foundation model."""

    class Config:
        arbitrary_types_allowed = True

    id: ModelId = Field(description="Identifier for this model.")
    local_repo_dir: str = Field(description="Local repository with the required files.")


class ModelMetadata(BaseModel):
    id: ModelId = Field(description="Identifier for this trained model.")
    block: PositiveInt = Field(description="Block on which this model was claimed on the chain.")
