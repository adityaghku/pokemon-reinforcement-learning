import os
import sys
import logging
from datetime import datetime


def setup_logging(
    log_file_name,
    log_dir,
    log_level=logging.INFO,
):
    """
    Set up logging to a file and standard output with detailed formatting.

    Args:
        log_dir (str): Directory where logs will be stored.
        log_level (int): Logging level (e.g., logging.INFO).
        log_file_prefix (str): Prefix for the log file name.

    Returns:
        logging.Logger: Configured logger instance.
    """
    try:
        # Ensure the log directory exists
        os.makedirs(log_dir, exist_ok=True)

        # Create timestamped log filename
        timestamp = datetime.now().strftime("%Y%m%d_%H")
        log_filename = f"{log_file_name}_{timestamp}.log"
        log_path = os.path.join(log_dir, log_filename)

        # Define detailed format
        log_format = (
            "%(asctime)s — [%(levelname)s] — "
            "%(name)s:%(lineno)d — %(funcName)s() — %(message)s"
        )
        formatter = logging.Formatter(log_format)

        # Create logger
        logger = logging.getLogger()
        logger.setLevel(log_level)

        # Clear existing handlers to avoid duplicates
        if logger.hasHandlers():
            logger.handlers.clear()

        # File handler
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Stdout handler
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        logger.debug("Logging system initialized.")

        return logger

    except Exception as e:
        print(f"Failed to set up logging: {e}")
        raise
