import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import train_test_split

from config import SEGMENT_LABELS, RANDOM_STATE
from src.logger_setup import get_logger

logger = get_logger(__name__)


def evaluate_model(name, model, X_train, y_train, X_test, y_test, overfit_sample=20_000):
    y_pred = model.predict(X_test)
    assert len(y_pred) == len(y_test), "Predicciones y test con distinto largo"

    acc = accuracy_score(y_test, y_pred)
    p, r, f, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=SEGMENT_LABELS, average="macro", zero_division=0)
    pw, rw, fw, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=SEGMENT_LABELS, average="weighted", zero_division=0)
    pc, rc, fc, sc = precision_recall_fscore_support(
        y_test, y_pred, labels=SEGMENT_LABELS, zero_division=0)

    # control de sobreajuste: accuracy en una submuestra del train
    X_s, _, y_s, _ = train_test_split(X_train, y_train, train_size=min(overfit_sample, len(X_train)),
                                      stratify=y_train, random_state=RANDOM_STATE)
    acc_train = accuracy_score(y_s, model.predict(X_s))
    gap = acc_train - acc
    if gap > 0.05:
        logger.warning("[%s] posible sobreajuste: accuracy train %.3f vs test %.3f", name, acc_train, acc)

    cm = confusion_matrix(y_test, y_pred, labels=SEGMENT_LABELS)
    metrics = {
        "model": name,
        "accuracy": acc, "precision_macro": p, "recall_macro": r, "f1_macro": f,
        "precision_weighted": pw, "recall_weighted": rw, "f1_weighted": fw,
        "accuracy_train_sample": acc_train,
        "per_class": {l: {"precision": float(pc[i]), "recall": float(rc[i]),
                          "f1": float(fc[i]), "support": int(sc[i])}
                      for i, l in enumerate(SEGMENT_LABELS)},
        "confusion_matrix": cm.tolist(),
        "labels": SEGMENT_LABELS,
    }
    logger.info("[%s] TEST accuracy=%.4f precision=%.4f recall=%.4f f1_macro=%.4f (train acc %.4f)",
                name, acc, p, r, f, acc_train)
    logger.info("[%s] matriz de confusión (filas=real, columnas=predicho %s):\n%s",
                name, SEGMENT_LABELS, cm)

    proba = model.predict_proba(X_test)
    preds = pd.DataFrame({"id_fila": X_test.index, "modelo": name,
                          "y_real": y_test.values, "y_pred": y_pred})
    for i, c in enumerate(model.classes_):
        preds[f"proba_{c}"] = proba[:, i].round(4)
    preds["acierto"] = preds["y_real"] == preds["y_pred"]
    return metrics, preds