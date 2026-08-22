"""Logging setup, called once when a script or the app starts.

Other modules do not configure anything - they just ask for a logger:

    import logging
    log = logging.getLogger(__name__)
    log.info("indexed %d chunks", count)

Using __name__ tags each line with the module it came from, so a log file
shows which part of the system produced which message.
"""

import logging
from pathlib import Path

from src.config import PROJECT_ROOT

LOG_DIR = PROJECT_ROOT / "logs"


def setup_logging(filename="app.log", level=logging.INFO):
    """Send log messages to the console and to logs/<filename>."""
    LOG_DIR.mkdir(exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setFormatter(formatter)

    # encoding matters: log lines contain Arabic text
    to_file = logging.FileHandler(LOG_DIR / filename, encoding="utf-8")
    to_file.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)

    # clear existing handlers so calling this twice does not duplicate output
    root.handlers.clear()
    root.addHandler(console)
    root.addHandler(to_file)

    # chromadb and sentence-transformers are chatty at INFO level
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)