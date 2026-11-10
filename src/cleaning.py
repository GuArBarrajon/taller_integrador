import numpy as np
import pandas as pd

from config import (DROP_COLUMNS, BINARY_COLUMNS, CABA_LAT, CABA_LON, M2_RANGE,
                    DOLARES_RANGE, USM2_RANGE, MAX_AMBIENTES, MAX_BANOS,
                    MAX_ANTIGUEDAD, TARGET_BASE, TARGET_NAME, SEGMENT_LABELS,
                    BARRIO_FIXES, ORIENT_FIXES, ORIENT_INVALID)
from src.logger_setup import get_logger

logger = get_logger(__name__)


def _log_rows(step, before, after):
    logger.info("%s: %d -> %d filas (%d eliminadas)", step, before, after, before - after)


def drop_unused_columns(df):
    cols = [c for c in DROP_COLUMNS if c in df.columns]
    logger.info("Se descartan columnas casi vacías: %s", cols)
    return df.drop(columns=cols)


def drop_duplicates(df):
    n0 = len(df)
    df = df.drop_duplicates()
    _log_rows("Duplicados exactos", n0, len(df))
    return df


def clean_prices_and_surface(df):
    for c in ["M2", "DOLARES", "U_S_M2"]:
        s = df[c].astype("string").str.strip().str.replace(",", ".", regex=False)
        df[c] = pd.to_numeric(s, errors="coerce").astype("float64")

    n0 = len(df)
    df = df.dropna(subset=["M2", "DOLARES"])
    _log_rows("Sin M2 o precio", n0, len(df))

    # rangos de superficie y precio total
    ok = df["M2"].between(*M2_RANGE) & df["DOLARES"].between(*DOLARES_RANGE)
    n0 = len(df)
    df = df[ok].copy()
    _log_rows("Outliers de M2/DOLARES", n0, len(df))

    # precio por m² propio, consistente en todos los años
    df["PRECIO_M2"] = df["DOLARES"] / df["M2"]
    ok = df["PRECIO_M2"].between(*USM2_RANGE)
    por_año = (~ok).groupby(df["AÑO"]).sum()
    for y, n in por_año[por_año > 0].items():
        logger.warning("[%s] %d filas con PRECIO_M2 fuera de rango", y, n)
    n0 = len(df)
    df = df[ok]
    _log_rows("Outliers de PRECIO_M2", n0, len(df))

    # solo informativo: U_S_M2 publicado vs. recalculado
    desvio = ((df["PRECIO_M2"] - df["U_S_M2"]).abs() / df["U_S_M2"]) > 0.05
    for y, n in desvio.groupby(df["AÑO"]).sum().loc[lambda s: s > 100].items():
        logger.warning("[%s] U_S_M2 publicado difiere >5%% de DOLARES/M2 en %d filas "
                       "(se usa PRECIO_M2)", y, n)
    return df


def clean_text_columns(df):
    for c, fixes in (("BARRIO", BARRIO_FIXES), ("ORIENT", ORIENT_FIXES)):
        txt = df[c].astype("string").str.strip().str.upper()
        txt = txt.replace(fixes)
        if c == "ORIENT":
            txt = txt.mask(txt.isin(ORIENT_INVALID))
        txt = txt.mask(txt == "")
        df[c] = txt.astype(object).where(txt.notna(), np.nan)

    n0 = len(df)
    df = df.dropna(subset=["BARRIO"])
    _log_rows("Sin BARRIO", n0, len(df))
    df["COMUNA"] = pd.to_numeric(df["COMUNA"], errors="coerce")
    n_barrios = df["BARRIO"].nunique()
    logger.info("BARRIOS distintos: %d (se esperan 48)", n_barrios)
    if n_barrios != 48:
        logger.warning("Cantidad de barrios distinta de 48: revisar variantes de escritura")
    return df


def clean_coordinates(df):
    for c in ["LAT", "LON"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    ok = df["LAT"].between(*CABA_LAT) & df["LON"].between(*CABA_LON)
    bad = df["LAT"].notna() & ~ok
    for y, n in bad.groupby(df["AÑO"]).sum().loc[lambda s: s > 0].items():
        logger.warning("[%s] %d coordenadas inválidas pasan a NaN", y, n)
    df.loc[bad, ["LAT", "LON"]] = np.nan
    return df


def _binary_flag(s: pd.Series) -> pd.Series:
    txt = s.astype("string").str.strip().str.upper()
    num = pd.to_numeric(txt, errors="coerce")
    si = txt.str.match(r"^S[IÍ]?\b", na=False) | (num > 0).fillna(False)
    return pd.Series(np.where(si.to_numpy(dtype=bool), 1.0, 0.0), index=s.index)


def clean_amenities(df):
    for c in BINARY_COLUMNS:
        no_disponible = df[c].isna().groupby(df["AÑO"]).transform("all")
        df[c] = _binary_flag(df[c]).where(~no_disponible, np.nan)
        años = sorted(df.loc[no_disponible, "AÑO"].unique().tolist())
        logger.info("%s -> 0/1. Sin dato en los años: %s. Proporción de 1: %.1f%%",
                    c, años, 100 * df[c].mean())
    return df


def clean_antiguedad(df):
    txt = df["ANTIGUEDAD"].astype("string").str.strip().str.upper()
    estrenar = (txt == "A ESTRENAR").fillna(False).to_numpy(dtype=bool)

    ant = pd.to_numeric(txt, errors="coerce").astype("float64")
    n_cero = int((ant == 0).sum())
    ant = ant.mask(ant == 0)            # 0 numérico = sin dato
    ant = ant.mask(estrenar, 0.0)       # "A ESTRENAR" = dato real
    logger.info("ANTIGUEDAD: %d 'A ESTRENAR' -> 0; %d ceros numéricos -> NaN", int(estrenar.sum()), n_cero)

    invalid = (ant < 0) | (ant > MAX_ANTIGUEDAD)
    ant = ant.mask(invalid)
    logger.info("ANTIGUEDAD: %d valores inválidos -> NaN; %d nulos en total",
                int(invalid.sum()), int(ant.isna().sum()))
    df["ANTIGUEDAD"] = ant
    return df


def clean_counts(df):
    for col, max_val in (("AMBIENTES", MAX_AMBIENTES), ("BAÑOS", MAX_BANOS)):
        x = pd.to_numeric(df[col], errors="coerce")
        invalid = (x <= 0) | (x > max_val)
        logger.info("%s: %d valores (<=0 o >%d) pasan a NaN", col, int(invalid.sum()), max_val)
        df[col] = x.mask(invalid)
    return df


def create_target(df):
    """Terciles de TARGET_BASE calculados dentro de cada año."""
    pct = df.groupby("AÑO")[TARGET_BASE].rank(pct=True, method="first")
    df[TARGET_NAME] = pd.cut(pct, bins=[0, 1/3, 2/3, 1.0], labels=SEGMENT_LABELS,
                             include_lowest=True).astype(str)
    logger.info("Distribución del target:\n%s", df[TARGET_NAME].value_counts().to_string())
    return df


def clean_dataset(df_raw: pd.DataFrame) -> pd.DataFrame:
    logger.info("=== Inicio de limpieza: %s filas, %s columnas ===", *df_raw.shape)
    df = df_raw.copy()
    for step in (drop_unused_columns, drop_duplicates, clean_prices_and_surface,
                 clean_text_columns, clean_coordinates, clean_amenities,
                 clean_antiguedad, clean_counts, create_target):
        df = step(df)

    df = df.reset_index(drop=True)
    assert df[TARGET_NAME].notna().all(), "Hay filas sin target"
    assert df[["M2", "DOLARES", "U_S_M2", "BARRIO"]].notna().all().all(), "Nulos en columnas clave"
    logger.info("=== Fin de limpieza: %s filas, %s columnas ===", *df.shape)
    return df