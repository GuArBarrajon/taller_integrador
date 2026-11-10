import sqlite3

import pandas as pd
from sqlalchemy import create_engine

from config import SQLITE_PATH, TABLE_CLEAN, DATA_PROCESSED
from src.logger_setup import get_logger

logger = get_logger(__name__)


def save_clean_data(df: pd.DataFrame, table: str = TABLE_CLEAN) -> None:
    """Tabla 1: datos limpios en la base relacional."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{SQLITE_PATH}")
    df.to_sql(table, engine, if_exists="replace", index=False, chunksize=10_000)

    with sqlite3.connect(SQLITE_PATH) as con:
        n = con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
    logger.info("Tabla '%s' guardada en %s: %d filas", table, SQLITE_PATH.name, n)
    if n != len(df):
        raise ValueError(f"Se esperaban {len(df)} filas y la tabla tiene {n}")


def load_clean_data(table: str = TABLE_CLEAN) -> pd.DataFrame:
    engine = create_engine(f"sqlite:///{SQLITE_PATH}")
    df = pd.read_sql_table(table, engine)
    logger.info("Tabla '%s' leída: %s filas, %s columnas", table, *df.shape)
    return df