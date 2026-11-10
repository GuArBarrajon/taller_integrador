from datetime import datetime, timezone

import pandas as pd
from pymongo import MongoClient

from config import (MONGO_URI, MONGO_DB, RUN_ID, COL_PREDICTIONS, COL_METRICS, COL_PARAMS,
                    RANDOM_STATE, TEST_SIZE, CV_FOLDS, SCORING, SEARCH_SAMPLE_SIZE,
                    NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET_NAME, TARGET_BASE, TEMPORAL_CUTOFF, COL_LEAKAGE)
from src.logger_setup import get_logger

logger = get_logger(__name__)


def get_db():
    if not MONGO_URI:
        raise RuntimeError("Falta MONGO_URI en el .env")
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=8000)
    client.admin.command("ping")
    return client[MONGO_DB]


def _now():
    return datetime.now(timezone.utc)


def _replace_run(col, docs, escenario, batch=5000):
    filtro = {"run_id": RUN_ID, "escenario": escenario}
    col.delete_many(filtro)
    for i in range(0, len(docs), batch):
        col.insert_many(docs[i:i + batch])
    n = col.count_documents(filtro)
    logger.info("Mongo '%s' [%s]: %d documentos insertados (run_id=%s)", col.name, escenario, n, RUN_ID)
    if n != len(docs):
        raise ValueError(f"'{col.name}': se esperaban {len(docs)} documentos y hay {n}")


def save_params(results: dict, db=None):
    db = db or get_db()
    docs = [{
        "run_id": RUN_ID, "escenario": "comun", "timestamp": _now(), "modelo": name,
        "mejores_parametros": r["best_params"],
        "cv_score": r["cv_score"], "scoring": SCORING, "cv_folds": CV_FOLDS,
        "segundos_busqueda": r["search_seconds"], "segundos_ajuste_final": r["fit_seconds"],
        "random_state": RANDOM_STATE, "test_size": TEST_SIZE, "corte_temporal": TEMPORAL_CUTOFF,
        "filas_busqueda": SEARCH_SAMPLE_SIZE,
        "predictoras_numericas": NUMERIC_FEATURES,
        "predictoras_categoricas": CATEGORICAL_FEATURES,
        "target": TARGET_NAME, "target_base": TARGET_BASE,
    } for name, r in results.items()]
    _replace_run(db[COL_PARAMS], docs, "comun")


def save_metrics(all_metrics: list, escenario: str, db=None):
    db = db or get_db()
    docs = [{**m, "run_id": RUN_ID, "escenario": escenario, "timestamp": _now()} for m in all_metrics]
    _replace_run(db[COL_METRICS], docs, escenario)


def save_predictions(preds: pd.DataFrame, escenario: str, db=None):
    db = db or get_db()
    docs = preds.to_dict("records")
    for d in docs:
        d["run_id"], d["escenario"] = RUN_ID, escenario
    _replace_run(db[COL_PREDICTIONS], docs, escenario)


def save_leakage(df_leak: pd.DataFrame, db=None):
    """Prueba de leakage: una fila por variante de predictoras."""
    db = db or get_db()
    docs = []
    for r in df_leak.to_dict("records"):
        r["accuracy"], r["f1_macro"] = float(r["accuracy"]), float(r["f1_macro"])
        r["delta_vs_modelo_real"] = float(r["delta_vs_modelo_real"])
        r.update({"run_id": RUN_ID, "escenario": "leakage", "timestamp": _now(),
                  "modelo": "random_forest", "hiperparametros_rf": {"n_estimators": 100,
                  "min_samples_leaf": 5}, "random_state": RANDOM_STATE})
        docs.append(r)
    _replace_run(db[COL_LEAKAGE], docs, "leakage")

def load_run(collection: str, escenario: str, run_id: str = None, db=None) -> pd.DataFrame:
    """Lee una colección; si no se da run_id, usa la ejecución más reciente."""
    db = db or get_db()
    col = db[collection]
    if run_id is None:
        last = col.find_one({"escenario": escenario}, sort=[("run_id", -1)], projection={"run_id": 1})
        if last is None:
            raise ValueError(f"'{collection}' no tiene documentos con escenario '{escenario}'")
        run_id = last["run_id"]
    df = pd.DataFrame(list(col.find({"run_id": run_id, "escenario": escenario}, {"_id": 0})))
    logger.info("Mongo '%s' [%s]: %d documentos leídos (run_id=%s)", collection, escenario, len(df), run_id)
    return df