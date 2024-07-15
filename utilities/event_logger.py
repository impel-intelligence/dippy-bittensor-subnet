from loguru import logger
import os, sys


class EventLogger:
    def __init__(
        self,
        filepath="/tmp/valapi_event_logs/validator_api_{time:UNIX}.log",
        level="INFO",
        stderr=False,
    ):
        self.logger = logger
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

        # Configure loguru logger for JSON output and file rotation
        format = "{time} | {level} | {message}"
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
        self.logger.add(
            filepath,
            rotation="100 MB",
            level=level,
            format=format,
            serialize=True,  # For JSON output in file with rotation
        )

    def log(self, level, message, **kwargs):
        # Include additional fields in the log
        log_method = getattr(self.logger, level)
        log_method(message, **kwargs)

    def info(self, message, **kwargs):
        self.log("info", message, **kwargs)

    def error(self, message, **kwargs):
        self.log("error", message, **kwargs)

    def debug(self, message, **kwargs):
        self.log("debug", message, **kwargs)


# Example of using the JsonLogger


def example():
    try:
        json_logger = EventLogger()
        json_logger.info("This is an info message", extra={"user": "admin", "status": "active"})
        json_logger.error("This is an error essage", extra={"user": "guest", "error_code": 500})
        json_logger.debug("This is a debug message", extra={"user": "developer", "debug_mode": "on"})
    except PermissionError as e:
        print(f"Failed to initialize logger: {e}")
