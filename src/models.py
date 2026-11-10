import time

import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     StratifiedKFold, train_test_split)
from sklearn.pipeline import Pipeline

from config import (PARAM_SPACES, RF_N_ITER, CV_FOLDS, SCORING,
                    SEARCH_SAMPLE_SIZE, RANDOM_STATE, DUMMY_STRATEGY)
from src.features import build_preprocessor
from src.logger_setup import get_logger

from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score

logger = get_logger(__name__)

def fit_baseline(X_train, y_train, strategy=DUMMY_STRATEGY) -> dict:
    """Línea base: ¿cuánto se logra sin aprender nada de las predictoras?"""
    model = DummyClassifier(strategy=strategy, random_state=RANDOM_STATE)
    cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_score = float(cross_val_score(model, X_train, y_train, scoring=SCORING, cv=cv).mean())
    t0 = time.time()
    model.fit(X_train, y_train)
    logger.info("[dummy] estrategia=%s | %s en CV: %.4f", strategy, SCORING, cv_score)
    return {"name": "dummy", "model": model, "best_params": {"strategy": strategy},
            "cv_score": cv_score, "cv_results": None, "search_seconds": 0.0,
            "fit_seconds": round(time.time() - t0, 2)}


def build_pipeline(name: str) -> Pipeline:
    if name == "logreg":
        clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    elif name == "random_forest":
        clf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1)
    else:
        raise ValueError(f"Modelo desconocido: {name}")
    return Pipeline([("prep", build_preprocessor()), ("clf", clf)])


def tune_and_fit(name: str, X_train, y_train) -> dict:
    """Busca hiperparámetros (CV sobre una submuestra) y reentrena con todo el train."""
    pipe = build_pipeline(name)
    cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    space = PARAM_SPACES[name]

    if name == "logreg":
        search = GridSearchCV(pipe, space, scoring=SCORING, cv=cv, n_jobs=-1, refit=False)
    else:
        search = RandomizedSearchCV(pipe, space, n_iter=RF_N_ITER, scoring=SCORING,
                                    cv=cv, n_jobs=-1, random_state=RANDOM_STATE, refit=False)

    if SEARCH_SAMPLE_SIZE and len(X_train) > SEARCH_SAMPLE_SIZE:
        X_s, _, y_s, _ = train_test_split(X_train, y_train, train_size=SEARCH_SAMPLE_SIZE,
                                          stratify=y_train, random_state=RANDOM_STATE)
    else:
        X_s, y_s = X_train, y_train

    logger.info("[%s] búsqueda de hiperparámetros sobre %d filas (%s, cv=%d, scoring=%s)",
                name, len(X_s), type(search).__name__, CV_FOLDS, SCORING)
    t0 = time.time()
    search.fit(X_s, y_s)
    t_search = time.time() - t0
    logger.info("[%s] mejores parámetros: %s | %s en CV: %.4f | %.0f s",
                name, search.best_params_, SCORING, search.best_score_, t_search)

    final = clone(pipe).set_params(**search.best_params_)
    
    if name == "random_forest":
        final.set_params(clf__n_jobs=-1)   # el ajuste final sí puede usar todos los núcleos

    t0 = time.time()
    final.fit(X_train, y_train)
    t_fit = time.time() - t0
    logger.info("[%s] reentrenado con %d filas en %.0f s", name, len(X_train), t_fit)

    return {
        "name": name,
        "model": final,
        "best_params": search.best_params_,
        "cv_score": float(search.best_score_),
        "cv_results": pd.DataFrame(search.cv_results_),
        "search_seconds": round(t_search, 1),
        "fit_seconds": round(t_fit, 1),
    }