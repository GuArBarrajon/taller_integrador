import numpy as np
import pandas as pd

from config import (DATA_RAW, RAW_FILE_PATTERN, CSV_PARAMS,
                    COLUMN_RENAME_MAP, TARGET_COLUMNS)
from src.logger_setup import get_logger

logger = get_logger(__name__)


def _warn_bad_line(fields):
    # pandas entrega la línea problemática como lista de campos
    logger.warning("Línea descartada por formato inválido: %s", ";".join(fields)[:80])
    return None


def get_raw_path(year: int):
    """Devuelve la ruta del archivo del año y verifica que sea un CSV real."""
    path = DATA_RAW / RAW_FILE_PATTERN.format(year=year)
    if not path.exists():
        raise FileNotFoundError(f"[{year}] No se encontró {path}. Descargalo a mano y copialo en data/raw/")

    with open(path, "rb") as f:
        head = f.read(300).lower()
    if b"<html" in head or b"request rejected" in head:
        raise ValueError(f"[{year}] {path.name} contiene HTML y no datos. Descargalo de nuevo a mano.")

    logger.info("[%s] archivo encontrado: %s", year, path.name)
    return path


def standardize_year(year: int) -> pd.DataFrame:
    """Lee el CSV original de un año y lo deja con las columnas estándar."""
    path = get_raw_path(year)
    df = pd.read_csv(path, engine="python", on_bad_lines=_warn_bad_line, **CSV_PARAMS)
    raw_shape = df.shape

    df.columns = df.columns.str.strip().str.upper()
    df = df.rename(columns=COLUMN_RENAME_MAP)
    df["AÑO"] = year

    missing = [c for c in TARGET_COLUMNS if c not in df.columns]
    if missing:
        logger.info("[%s] columnas ausentes (se rellenan con NaN): %s", year, missing)
    extra = [c for c in df.columns if c not in TARGET_COLUMNS]
    if extra:
        logger.warning("[%s] columnas no contempladas (se descartan): %s", year, extra)

    for c in missing:
        df[c] = np.nan
    df = fix_lat_lon(df, year)[TARGET_COLUMNS]

    logger.info("[%s] estandarizado: %s -> %s", year, raw_shape, df.shape)
    return df


def build_unified_dataset(years) -> pd.DataFrame:
    frames = [standardize_year(y) for y in years]
    df = pd.concat(frames, ignore_index=True)
    assert list(df.columns) == TARGET_COLUMNS, "Columnas inesperadas tras unificar"
    logger.info("Dataset unificado: %s filas, %s columnas", *df.shape)
    
    check_coordinates(df)
    
    return df


def fix_lat_lon(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Corrige filas donde LAT y LON están invertidas (CABA: lat≈-34.6, lon≈-58.4)."""
    lat = pd.to_numeric(df["LAT"], errors="coerce")
    lon = pd.to_numeric(df["LON"], errors="coerce")
    swapped = lat.abs() > lon.abs()
    n = int(swapped.sum())
    if n:
        logger.warning("[%s] %d filas con LAT/LON invertidas, se corrigen", year, n)
    df["LAT"] = np.where(swapped, lon, lat)
    df["LON"] = np.where(swapped, lat, lon)
    return df

def log_null_report(df: pd.DataFrame) -> pd.DataFrame:
    """Reemplaza los bucles de nulos: nulos por columna y por año."""
    nulls = df.isnull().sum()
    logger.info("Nulos por columna:\n%s", nulls[nulls > 0].to_string())

    por_año = df.drop(columns="AÑO").isnull().groupby(df["AÑO"]).sum()
    # columnas 100% nulas en algún año = esa variable no existe ese año
    for y, fila in por_año.iterrows():
        total = (df["AÑO"] == y).sum()
        vacias = fila[fila == total].index.tolist()
        if vacias:
            logger.warning("[%s] columnas completamente nulas: %s", y, vacias)
    return por_año

def check_coordinates(df: pd.DataFrame) -> None:
    """Verifica que LAT/LON caigan dentro del rectángulo aproximado de CABA."""
    lat_ok = df["LAT"].between(-34.72, -34.52)
    lon_ok = df["LON"].between(-58.54, -58.33)
    fuera = df["LAT"].notna() & ~(lat_ok & lon_ok)
    por_año = fuera.groupby(df["AÑO"]).sum()
    logger.info("Coordenadas fuera de CABA en total: %d", int(fuera.sum()))
    for y, n in por_año[por_año > 0].items():
        logger.warning("[%s] %d filas con coordenadas fuera de CABA", y, n)