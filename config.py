from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
LOGS_DIR = BASE_DIR / "logs"

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