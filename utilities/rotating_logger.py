import os
import json
from loguru import logger
from datetime import datetime
from typing import Dict, Any


class RotatingLogSink:
    """
    Custom sink implementation that handles log rotation based on file size.
    """

    def __init__(self, base_path: str, max_size: int = 100 * 1024 * 1024):
        """
        Initialize the rotating sink.

        Args:
            base_path: Base path template for log files
            max_size: Maximum size in bytes before rotation (default 100MB)
        """
        self.base_path = base_path
        self.max_size = max_size
        self.current_file = None
        self.current_size = 0

    def _get_new_logfile(self) -> str:
        """Generate a new logfile path using the template"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.base_path.format(time=datetime.now(), timestamp=timestamp)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def write(self, message: str):
        """Write message to log file, handling rotation if needed."""
        if self.current_file is None or self.current_size >= self.max_size:
            if self.current_file is not None:
                self.current_file.close()
            new_path = self._get_new_logfile()
            self.current_file = open(new_path, "a", encoding="utf-8")
            self.current_size = 0

        message_bytes = message.encode("utf-8")
        message_size = len(message_bytes)

        if message_size > self.max_size:
            message = message[: self.max_size - 1000] + "... (truncated)\n"
            message_bytes = message.encode("utf-8")
            message_size = len(message_bytes)

        self.current_file.write(message)
        self.current_file.flush()
        self.current_size += message_size

    def get_sink_func(self):
        return self.__call__

    def __call__(self, message):
        """
        Callable interface for loguru sink.
        Serializes the record and writes to file.
        """
        record = message.record
        try:
            serialized = {
                "timestamp": record["time"].timestamp(),
                "datetime": record["time"].strftime("%Y-%m-%d %H:%M:%S.%f"),
                "message": record["message"],
                "extra": record["extra"],
            }

            if record["exception"] is not None:
                serialized["exception"] = str(record["exception"])
            s = json.dumps(serialized)
            self.write(s + "\n")

        except Exception as e:
            self.write(
                json.dumps(
                    {"error": "Serialization failed", "message": str(record["message"]), "error_details": str(e)}
                )
                + "\n"
            )

    def __del__(self):
        """Cleanup when object is destroyed"""
        if self.current_file is not None:
            self.current_file.close()


def example_usage():
    """Example of how to use the RotatingLogSink with loguru"""
    # Remove default logger
    logger.remove()

    # Create and add rotating sink
    sink = RotatingLogSink(base_path="/tmp/logs/myapp/{time:%Y-%m-%d}/app_{timestamp}.log")

    # Add the sink to loguru
    logger.add(sink, enqueue=True, level="INFO")

    # Test logging
    for i in range(5):
        logger.info(f"Test message {i}", extra={"iteration": i})
        logger.error("Test error", extra={"error_code": 500})

    try:
        raise ValueError("Test exception")
    except Exception as e:
        logger.error("Caught exception", extra={"error": str(e)})


if __name__ == "__main__":
    example_usage()
