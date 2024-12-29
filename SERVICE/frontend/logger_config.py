import os
import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

import streamlit as st

LOG_FILE_PATH = (
    Path(__file__).parent.resolve()
    / "logs" / "frontend" / "frontend.log"
)


@st.cache_resource
def get_logger():
    """Настраивает логирование с записью в файл и выводом в консоль."""
    os.makedirs(LOG_FILE_PATH.parent, exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )

    timed_handler = TimedRotatingFileHandler(
        LOG_FILE_PATH,
        when="midnight",
        interval=1,
        backupCount=7,
        delay=True
    )
    timed_handler.setLevel(logging.INFO)
    timed_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger = logging.getLogger("toxic_comments_app")
    logger.setLevel(logging.INFO)
    logger.addHandler(timed_handler)
    logger.addHandler(stream_handler)

    return logger
