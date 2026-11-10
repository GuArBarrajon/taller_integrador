import time

from sklearn.base import clone

from config import TEMPORAL_CUTOFF
from src.logger_setup import get_logger

logger = get_logger(__name__)


def temporal_split(df, X, y, cutoff=TEMPORAL_CUTOFF):
    """Entrena con años <= cutoff y prueba con los posteriores."""
    es_train = (df["AÑO"] <= cutoff).to_numpy()
    assert df.loc[es_train, "AÑO"].max() < df.loc[~es_train, "AÑO"].min()
    X_tr, X_te, y_tr, y_te = X.loc[es_train], X.loc[~es_train], y.loc[es_train], y.loc[~es_train]
    logger.info("Split temporal: train %d filas (%d-%d) | test %d filas (%d-%d)",
                len(X_tr), df.loc[es_train, "AÑO"].min(), cutoff,
                len(X_te), cutoff + 1, df.loc[~es_train, "AÑO"].max())
    return X_tr, X_te, y_tr, y_te


def fit_temporal(name, base_model, X_tr, y_tr, rf_trees=100):
    """Reentrena con los mejores hiperparámetros ya encontrados (sin nueva búsqueda)."""
    model = clone(base_model)
    if name == "random_forest":
        model.set_params(clf__n_estimators=rf_trees, clf__n_jobs=-1)
    t0 = time.time()
    model.fit(X_tr, y_tr)
    logger.info("[%s] ajuste temporal con %d filas en %.0f s", name, len(X_tr), time.time() - t0)
    return model