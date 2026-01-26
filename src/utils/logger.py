import logging
from pathlib import Path
from datetime import datetime
import tempfile

# Global variable to store the log file path for the current run
_LOG_FILE = None

def get_logger(name: str):
    global _LOG_FILE
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    # Create log file path only once for the entire pipeline run
    if _LOG_FILE is None:
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            log_dir = Path(tempfile.gettempdir()) / "email_classifier_logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            _LOG_FILE = log_dir / f"{timestamp}.log"
            handler = logging.FileHandler(_LOG_FILE, encoding="utf-8")
            print(f"Logging to file: {_LOG_FILE}")
        except Exception as e:
            # Fallback to console logging if file creation fails
            print(f"Warning: Failed to create log file. Logging to console. Error: {e}")
            handler = logging.StreamHandler()
    else:
        # File is already set, try to use it
        try:
            handler = logging.FileHandler(_LOG_FILE, encoding="utf-8")
        except Exception:
             handler = logging.StreamHandler()

    formatter = logging.Formatter(
        "[%(asctime)s]: %(filename)s - Line %(lineno)d: %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.propagate = False
    return logger