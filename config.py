from pathlib import Path
import os
from datetime import datetime
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
LOGS_DIR = BASE_DIR / "logs"

load_dotenv(BASE_DIR / ".env")
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB", "taller_integrador")
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")   # identifica cada ejecución

ROOT_URL = "https://data.buenosaires.gob.ar/dataset/departamentos-venta/resource/juqdkmgo-70"
RESOURCE_IDS = list(range(1, 32, 2))          # 1,3,5,...,31
YEARS = list(range(2001, 2017))               # 2001..2016
YEAR_TO_RESOURCE = dict(zip(YEARS, RESOURCE_IDS))

RAW_FILE_PATTERN = "departamentos-en-venta-{year}.csv"

CSV_PARAMS = dict(encoding="utf-8-sig", sep=";")

COLUMN_RENAME_MAP = {
    "LATITUD": "LAT",
    "LONGITUD": "LON",
    "M2CUB": "M2_CUBIERTOS",
    "BAÃOS": "BAÑOS",
    "BAÑO": "BAÑOS",
    "BAUL": "BAULERA",
    "BARRIOS": "BARRIO",
    "A_ESTRENAR": "ES_ESTRENAR",
}

TARGET_COLUMNS = [
    "AÑO", "CALLE", "NUMERO", "M2", "M2_CUBIERTOS", "DOLARES", "U_S_M2",
    "AMBIENTES", "DORMITORIO", "PISO", "ANTIGUEDAD", "ORIENT", "BAULERA",
    "COCHERA", "BAÑOS", "LAVADERO", "TERRAZA", "BARRIO", "COMUNA",
    "DIRECCION_NORMALIZADA", "LAT", "LON",
]

DROP_COLUMNS = ["M2_CUBIERTOS", "DORMITORIO", "PISO", "DIRECCION_NORMALIZADA"]
BINARY_COLUMNS = ["BAULERA", "COCHERA", "LAVADERO", "TERRAZA"]

CABA_LAT = (-34.72, -34.52)
CABA_LON = (-58.54, -58.33)

# Umbrales iniciales: ajustarlos mirando el EDA
M2_RANGE = (15, 500)
DOLARES_RANGE = (5_000, 5_000_000)
USM2_RANGE = (100, 10_000)
MAX_AMBIENTES = 10
MAX_BANOS = 8
MAX_ANTIGUEDAD = 120

TARGET_NAME = "SEGMENTO"
SEGMENT_LABELS = ["bajo", "medio", "alto"]

TARGET_BASE = "PRECIO_M2"
NON_FEATURES = ["CALLE", "NUMERO", "DOLARES", "U_S_M2", "PRECIO_M2", "SEGMENTO", "AÑO"]

BARRIO_FIXES = {
    "NUNEZ": "NUÑEZ",
    "MONTSERRAT": "MONSERRAT",
    "PARQUE AVELLANED": "PARQUE AVELLANEDA",
    "VILLA GRAL. MITR": "VILLA GRAL. MITRE",
}
ORIENT_FIXES = {"FRENTE Y CONTRA": "FRENTE Y CONTRAFRENTE",
                "DOBLE FRENTE": "FRENTE Y CONTRAFRENTE"}
ORIENT_INVALID = ["0", "SI"]
NON_FEATURES = ["CALLE", "NUMERO", "DOLARES", "U_S_M2", "SEGMENTO", "AÑO"]

SQLITE_PATH = DATA_PROCESSED / "deptos.db"
TABLE_CLEAN = "datos_limpios"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 3
SCORING = "f1_macro"
SEARCH_SAMPLE_SIZE = 20_000      # None = buscar sobre todo el train

NUMERIC_FEATURES = ["M2", "AMBIENTES", "ANTIGUEDAD", "BAÑOS", "BAULERA",
                    "COCHERA", "LAVADERO", "TERRAZA", "LAT", "LON"]
CATEGORICAL_FEATURES = ["BARRIO", "ORIENT"]

RF_N_ITER = 5
PARAM_SPACES = {
    "logreg": {"clf__C": [0.01, 0.1, 1, 10]},
    "random_forest": {
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [10, 20, None],
        "clf__min_samples_leaf": [1, 5, 10],
        "clf__max_features": ["sqrt", 0.5],
    },
}

MODELS_DIR = BASE_DIR / "models"

COL_PREDICTIONS = "predicciones"
COL_METRICS = "metricas"
COL_PARAMS = "parametros"

DUMMY_STRATEGY = "stratified"
TEMPORAL_CUTOFF = 2013        # train <= 2013, test >= 2014

COL_LEAKAGE = "leakage"

FIGURES_DIR = BASE_DIR / "reports" / "figures"