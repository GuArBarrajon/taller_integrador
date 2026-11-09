import logging
from logging.handlers import RotatingFileHandler
from config import LOGS_DIR

FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:            # evita duplicar handlers al reimportar
        return logger
    logger.setLevel(logging.DEBUG)

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter(FORMAT)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)

    file = RotatingFileHandler(LOGS_DIR / "pipeline.log", maxBytes=2_000_000,
                               backupCount=3, encoding="utf-8")
    file.setLevel(logging.DEBUG)
    file.setFormatter(fmt)

    logger.addHandler(console)
    logger.addHandler(file)
    # Más adelante: acá se agrega el MongoHandler
    return logger