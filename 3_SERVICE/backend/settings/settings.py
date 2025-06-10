import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.resolve()
MODELS_DIR = BASE_DIR / "models"
LOG_FILE_PATH = BASE_DIR / 'logs' / 'backend' / 'backend.log'

file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] - %(message)s")
stream_formatter = logging.Formatter("%(levelname)s:     %(message)s")

timed_handler = TimedRotatingFileHandler(
    LOG_FILE_PATH,
    when="midnight",
    interval=1,
    backupCount=7,
    delay=True
)
timed_handler.setLevel(logging.INFO)
timed_handler.setFormatter(file_formatter)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(stream_formatter)

logger = logging.getLogger("toxic_comments_app")
logger.setLevel(logging.INFO)
logger.addHandler(timed_handler)
logger.addHandler(stream_handler)
