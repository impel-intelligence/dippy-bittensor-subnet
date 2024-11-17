from loguru import logger
import os
import sys
import json
from datetime import datetime

from utilities.rotating_logger import RotatingLogSink

class EventLogger:
    def __init__(
        self,
        filepath="/tmp/valapi_event_logs/validator_api_{time:UNIX}.log",
        level="INFO",
        stderr=False,
    ):
        self.logger = logger
        self.filepath = filepath
        # Determine the directory part from the filepath
        log_directory = os.path.dirname(filepath)

        # Check if the directory exists, if not, create it
        if not os.path.exists(log_directory):
            try:
                os.makedirs(log_directory)
            except PermissionError:
                raise PermissionError(f"Cannot create log directory at {log_directory}")

        # Check if the directory is writable
        if not os.access(log_directory, os.W_OK):
            raise PermissionError(f"The directory {log_directory} is not writable")

        # Configure loguru logger for file output
        console_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{message}</cyan> | {extra} "
        )

        self.logger.remove()  # Remove default configuration
        
        if stderr:
            self.logger.add(
                sys.stderr,
                format=console_format,
                level=level,
            )

        sink = RotatingLogSink(
            base_path="/tmp/vlogs/app_{timestamp}.log"
        )

        # Add file sink with custom serializer 
        self.logger.add(
            level=level,
            sink=sink.get_sink_func(),
            enqueue=True,  # Makes logging thread-safe
        )

    def log(self, level, message, **kwargs):
        log_method = getattr(self.logger, level)
        log_method(message, **kwargs)

    def info(self, message, **kwargs):
        self.log("info", message, **kwargs)

    def error(self, message, **kwargs):
        self.log("error", message, **kwargs)

    def debug(self, message, **kwargs):
        self.log("debug", message, **kwargs)


# Example of using the EventLogger
def example():
    import time
    current_timestamp = int(time.time())
    try:
        json_logger = EventLogger()
        # Create sample EventData instance
        from dippy_validation_api.validation_api import EventData
        event_data = EventData(
            commit="test123",
            btversion="1.0.0", 
            uid="12345",
            hotkey="0xabc...",
            coldkey="0xdef...",
            payload={
                "event_type": "test_event",
                "status": "success"
            },
            signature={
                "r": "0x123...",
                "s": "0x456...", 
                "v": 27
            }
        )
        
        # Log the EventData instance
        json_logger.info("event_data_test", extra=event_data.to_dict())
        json_logger.info(f"info_message {current_timestamp}", extra={"user": "admin", "status": "active"})
        json_logger.error(f"error_message {current_timestamp}", extra={"user": "guest", "error_code": 500})
        json_logger.debug(f"debug_message {current_timestamp}", extra={"user": "developer", "debug_mode": "on"})
        print("finished")

    except PermissionError as e:
        print(f"Failed to initialize logger: {e}")

if __name__ == "__main__":
    example()
