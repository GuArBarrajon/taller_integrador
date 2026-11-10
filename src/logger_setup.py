import logging
from logging.handlers import RotatingFileHandler

from config import LOGS_DIR, MONGO_URI, MONGO_DB, RUN_ID

FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

_mongo_handler = None
_mongo_tried = False


def _get_mongo_handler():
    """Un único handler compartido por todos los loggers."""
    global _mongo_handler, _mongo_tried
    if not _mongo_tried:
        _mongo_tried = True
        if MONGO_URI:
            from src.mongo_log_handler import MongoLogHandler
            h = MongoLogHandler(MONGO_URI, MONGO_DB, RUN_ID)
            _mongo_handler = h if h.enabled else None
    return _mongo_handler


def flush_mongo_logs():
    """Llamar al final de un notebook o del main para enviar lo pendiente."""
    if _mongo_handler:
        _mongo_handler.flush()


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
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

    mongo = _get_mongo_handler()
    if mongo:
        logger.addHandler(mongo)
    return logger