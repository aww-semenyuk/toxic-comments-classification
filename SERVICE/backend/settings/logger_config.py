import logging
import os
from logging.handlers import TimedRotatingFileHandler

log_file_path = os.getenv("LOG_FILE_PATH")
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

timed_handler = TimedRotatingFileHandler(
    log_file_path,
    when="midnight",
    interval=1,
    backupCount=7
)
timed_handler.setLevel(logging.INFO)
timed_handler.setFormatter(formatter)

logger = logging.getLogger("toxic_comments_app")
logger.setLevel(logging.INFO)
logger.addHandler(timed_handler)
