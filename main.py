import argparse
from datetime import datetime

import papermill as pm

from config import BASE_DIR
from src.logger_setup import get_logger, flush_mongo_logs

logger = get_logger("main")
NB_DIR = BASE_DIR / "notebooks"
OUT_DIR = NB_DIR / "ejecutados"
KERNEL = "taller_integrador"


def run_notebook(nombre, parametros=None, stamp=""):
    OUT_DIR.mkdir(exist_ok=True)
    salida = OUT_DIR / f"{nombre}_{stamp}.ipynb"
    logger.info("Ejecutando %s.ipynb con parámetros %s", nombre, parametros or {})
    pm.execute_notebook(str(NB_DIR / f"{nombre}.ipynb"), str(salida),
                        parameters=parametros or {}, kernel_name=KERNEL,
                        cwd=str(NB_DIR), log_output=True)   # cwd: para que funcione sys.path.append("..")
    logger.info("Copia ejecutada guardada en %s", salida)


def main(rapido: bool, reusar: bool):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    params = {}
    if rapido:      # prueba de humo: no toca Mongo ni la caché completa
        params = dict(search_sample_size=5_000, rf_n_iter=2, models_to_run=["logreg"],
                      save_to_mongo=False, cache_name="results_rapido.joblib")
    elif reusar:
        params = dict(reuse_results=True)
    run_notebook("01_eda", stamp=stamp)
    run_notebook("02_modelado", params, stamp)
    flush_mongo_logs()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rapido", action="store_true", help="prueba de humo (logreg, sin Mongo)")
    ap.add_argument("--reusar", action="store_true", help="reutiliza models/results.joblib")
    a = ap.parse_args()
    main(a.rapido, a.reusar)