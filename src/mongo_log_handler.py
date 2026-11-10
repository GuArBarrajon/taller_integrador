import atexit
import logging
from datetime import datetime, timezone

from pymongo import MongoClient


class MongoLogHandler(logging.Handler):
    def __init__(self, uri, db_name, run_id, collection="logs",
                 batch_size=10, level=logging.INFO):
        super().__init__(level)
        self.run_id = run_id
        self.batch_size = batch_size
        self.buffer = []
        self.enabled = True
        try:
            client = MongoClient(uri, serverSelectionTimeoutMS=5000)
            client.admin.command("ping")
            self.col = client[db_name][collection]
        except Exception as e:
            self.enabled = False
            print(f"[MongoLogHandler] Sin conexión a Atlas, logs solo en archivo: {e}")
        atexit.register(self.flush)

    def emit(self, record):
        if not self.enabled:
            return
        try:
            self.buffer.append({
                "run_id": self.run_id,
                "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc),
                "level": record.levelname,
                "module": record.name,
                "message": record.getMessage(),
            })
            if len(self.buffer) >= self.batch_size:
                self.flush()
        except Exception:
            self.handleError(record)

    def flush(self):
        self.acquire()
        try:
            if self.enabled and self.buffer:
                docs, self.buffer = self.buffer, []
                try:
                    self.col.insert_many(docs)
                except Exception as e:
                    print(f"[MongoLogHandler] No se pudieron enviar logs: {e}")
        finally:
            self.release()