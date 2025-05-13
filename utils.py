import os
import sys
import logging
from datetime import datetime
from config import Config
import random
from multiprocessing import Lock


def setup_logging(log_file_name, log_dir, log_level=logging.DEBUG, process_id=None):
    """
    Set up logging to a file and standard output with detailed formatting.

    Args:
        log_file_name (str): Base name for the log file.
        log_dir (str): Directory where logs will be stored.
        log_level (int): Logging level (e.g., logging.INFO).
        process_id (int, optional): Process ID for unique log file names in multiprocessing.

    Returns:
        logging.Logger: Configured logger instance.
    """
    try:
        # Ensure the log directory exists
        os.makedirs(log_dir, exist_ok=True)

        # Create timestamped log filename
        timestamp = datetime.now().strftime("%Y%m%d_%H")
        if process_id is not None:
            log_filename = f"{log_file_name}_p{process_id}_{timestamp}.log"
        else:
            log_filename = f"{log_file_name}_{timestamp}.log"
        log_path = os.path.join(log_dir, log_filename)

        # Define detailed format
        log_format = (
            "%(asctime)s — [%(levelname)s] — "
            "%(name)s:%(lineno)d — %(funcName)s() — %(message)s"
        )
        formatter = logging.Formatter(log_format)

        # Create logger
        logger = logging.getLogger(
            f"ppo_training_p{process_id}" if process_id is not None else "ppo_training"
        )
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

        logger.debug(
            f"Logging system initialized for {'main process' if process_id is None else f'process {process_id}'}."
        )

        return logger

    except Exception as e:
        print(f"Failed to set up logging: {e}")
        raise


def clean_saves():
    # Get list of save files in the "rom" directory
    files = [
        f"rom/{f}"
        for f in os.listdir("rom")
        if f.endswith("_save.state") and f != Config.start_state
    ]

    # Check if there are more than 200 files
    if len(files) > 200:
        # Calculate how many files to delete
        num_to_delete = len(files) - 200

        # Randomly select files to delete
        files_to_delete = random.sample(files, num_to_delete)

        # Delete the selected files
        for file in files_to_delete:
            try:
                os.remove(file)
                print(f"Deleted {file}")
            except OSError as e:
                print(f"Error deleting {file}: {e}")

        print(f"Remaining save files: {len(files) - len(files_to_delete)}")

    else:
        print("No files to delete")
