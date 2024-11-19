from dataclasses import dataclass


@dataclass
class LocalMetadata:
    """Metadata associated with the local validator instance"""

    commit: str
    btversion: str
    uid: int = 0
    coldkey: str = ""
    hotkey: str = ""
