import os
import logging


def setup_logging(
        log_dir: str = "logs/frontend",
        log_file_name: str = "app.log"
):
    """
    Настраивает логирование с записью в файл и выводом в консоль.

    Args:
        log_dir (str): Путь к директории для логов.
        log_file_name (str): Имя файла лога.
    """

    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, log_file_name)

    logging.basicConfig(
        filename=log_file,
        filemode="a",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logging.getLogger().addHandler(console_handler)
