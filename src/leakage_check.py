import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from config import NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET_NAME, RANDOM_STATE
from src.features import build_preprocessor
from src.logger_setup import get_logger

logger = get_logger(__name__)

VARIANTES = {
    "sin DOLARES (modelo real)": [],
    "sin DOLARES, con AÑO": ["AÑO"],
    "con DOLARES": ["DOLARES"],
    "con DOLARES + AÑO": ["DOLARES", "AÑO"],
    "con PRECIO_M2 + AÑO": ["PRECIO_M2", "AÑO"],
}
RF_PARAMS = dict(n_estimators=100, min_samples_leaf=5)


def leakage_check(df, sample=40_000):
    """Mismo split y mismo modelo; solo cambian las predictoras."""
    d = df.sample(min(sample, len(df)), random_state=RANDOM_STATE)
    y = d[TARGET_NAME].astype(str)
    cols = NUMERIC_FEATURES + ["DOLARES", "AÑO", "PRECIO_M2"] + CATEGORICAL_FEATURES
    X = d[cols].copy()
    for c in CATEGORICAL_FEATURES:
        X[c] = X[c].astype(object)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y,
                                              random_state=RANDOM_STATE)
    logger.info("Leakage check: muestra de %d filas (train %d, test %d)",
                len(d), len(X_tr), len(X_te))

    rows = []
    for label, extra in VARIANTES.items():
        num = NUMERIC_FEATURES + extra
        feats = num + CATEGORICAL_FEATURES
        pipe = Pipeline([("prep", build_preprocessor(num, CATEGORICAL_FEATURES)),
                         ("clf", RandomForestClassifier(n_jobs=-1, random_state=RANDOM_STATE,
                                                        **RF_PARAMS))])
        pipe.fit(X_tr[feats], y_tr)
        pred = pipe.predict(X_te[feats])
        acc = accuracy_score(y_te, pred)
        f1 = f1_score(y_te, pred, average="macro")
        rows.append({"variante": label, "predictoras_extra": extra,
                     "accuracy": acc, "f1_macro": f1,
                     "filas_train": len(X_tr), "filas_test": len(X_te)})
        logger.info("Leakage check [%s]: accuracy=%.3f f1_macro=%.3f", label, acc, f1)

    out = pd.DataFrame(rows)
    base = out.loc[out["variante"] == "sin DOLARES (modelo real)", "accuracy"].iloc[0]
    out["delta_vs_modelo_real"] = out["accuracy"] - base
    return out