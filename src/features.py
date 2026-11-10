from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import (NUMERIC_FEATURES, CATEGORICAL_FEATURES, NON_FEATURES,
                    TARGET_NAME, TEST_SIZE, RANDOM_STATE)
from src.logger_setup import get_logger

logger = get_logger(__name__)


def get_xy(df):
    """Separa predictoras y target; verifica que no haya leakage por columnas."""
    feats = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    prohibidas = set(feats) & set(NON_FEATURES)
    if prohibidas:
        raise ValueError(f"Predictoras prohibidas (leakage/identificadores): {prohibidas}")

    X = df[feats].copy()
    for c in CATEGORICAL_FEATURES:
        X[c] = X[c].astype(object)          # evita problemas con el dtype 'str' de pandas
    y = df[TARGET_NAME].astype(str)
    logger.info("X: %s | y: %s | predictoras: %s", X.shape, y.shape, feats)
    return X, y


def split_data(X, y):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    logger.info("Train: %d filas | Test: %d filas", len(X_tr), len(X_te))
    logger.info("Target en train:\n%s", y_tr.value_counts(normalize=True).round(3).to_string())
    logger.info("Target en test:\n%s", y_te.value_counts(normalize=True).round(3).to_string())
    return X_tr, X_te, y_tr, y_te


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="SIN_DATO")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20)),
    ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])