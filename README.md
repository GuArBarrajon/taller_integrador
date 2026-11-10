# Taller integrador: clasificación del segmento de valor de departamentos en venta (CABA)

Proyecto final de **Programación Avanzada en Ciencia de Datos** (Licenciatura en Ciencia de Datos).

Pipeline de punta a punta: unifica 16 años (2001-2016) de publicaciones de departamentos en venta de CABA, los limpia, entrena modelos de clasificación con scikit-learn y guarda datos, resultados y logs en dos bases de datos. Incluye un dashboard en Streamlit.

- **Autor/es:** Luciano Asís, Gustavo Barrajón, Fernando Galera
- **Repositorio:** _(completar URL)_
- **Informe técnico** (limpieza, análisis y conclusiones): _(completar enlace o archivo)_

## Qué hace

1. **Carga y unifica** los 16 CSV anuales (nombres de columnas distintos, encoding, coordenadas invertidas).
2. **Limpia** los datos y crea el target `SEGMENTO` (bajo / medio / alto): terciles de `PRECIO_M2 = DOLARES / M2` calculados dentro de cada año.
3. **Guarda los datos limpios** en SQLite (Tabla 1).
4. **Entrena y compara** Regresión Logística y Random Forest con un `Pipeline` de scikit-learn (imputación, escalado, one-hot) y búsqueda de hiperparámetros (`GridSearchCV` / `RandomizedSearchCV`), contra un `DummyClassifier`.
5. **Evalúa** con Accuracy, Precision, Recall, F1 y matriz de confusión, en dos escenarios: split aleatorio y split temporal (train 2001-2013, test 2014-2016).
6. **Verifica leakage** (qué pasa si se incluyen variables de precio).
7. **Persiste en MongoDB Atlas** las predicciones, métricas, parámetros, la prueba de leakage y los logs.
8. **Visualiza** con matplotlib/seaborn (notebook `03_visualizacion`) y con un dashboard en Streamlit.

Todos los módulos escriben logs a consola, a archivo (`logs/pipeline.log`) y a MongoDB (colección `logs`).

## Bases de datos

| Qué | Dónde |
|---|---|
| Datos limpios y preprocesados (Tabla 1) | SQLite: `data/processed/deptos.db`, tabla `datos_limpios` |
| Predicciones y métricas (Tabla 2) | MongoDB: colecciones `predicciones` y `metricas` |
| Parámetros del modelo (Tabla 3) | MongoDB: colección `parametros` |
| Prueba de leakage | MongoDB: colección `leakage` |
| Logs de comprobaciones | MongoDB: colección `logs` (y `logs/pipeline.log`) |

Cada ejecución tiene un `run_id` (fecha y hora). Los documentos de métricas y predicciones tienen además el campo `escenario` (`aleatorio` o `temporal`).

## Estructura

```
taller_integrador/
├── config.py                 # rutas, parámetros, listas de columnas, URI de Mongo
├── main.py                   # ejecuta los notebooks con Papermill
├── requirements.txt
├── .env                      # credenciales (NO se sube a Git)
├── data/
│   ├── raw/                  # CSV originales (descarga manual)
│   └── processed/            # deptos.db (SQLite)
├── src/
│   ├── logger_setup.py       # logger a consola, archivo y Mongo
│   ├── mongo_log_handler.py
│   ├── data_loader.py        # lectura y unificación de los 16 años
│   ├── cleaning.py           # limpieza y creación del target
│   ├── db_sql.py             # SQLite (Tabla 1)
│   ├── features.py           # ColumnTransformer y split
│   ├── models.py             # modelos, búsqueda de hiperparámetros, baseline
│   ├── evaluation.py         # métricas y matriz de confusión
│   ├── temporal.py           # escenario temporal
│   ├── leakage_check.py      # prueba de leakage
│   ├── db_mongo.py           # MongoDB (Tablas 2 y 3, leakage, lectura)
│   └── visualization.py      # gráficos
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_modelado.ipynb
│   ├── 03_visualizacion.ipynb
│   └── ejecutados/           # copias ejecutadas por Papermill
├── app/dashboard.py          # dashboard Streamlit
├── models/                   # modelos entrenados (joblib)
├── reports/figures/          # gráficos exportados
└── logs/
```

## Requisitos

- Python 3.10 o superior (completar con la versión usada).
- Una cuenta gratuita de [MongoDB Atlas](https://www.mongodb.com/atlas) con un cluster M0.
- Los 16 CSV del dataset (ver más abajo).

## Instalación

```bash
git clone <URL-DEL-REPOSITORIO>
cd taller_integrador

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / Mac:
source .venv/bin/activate

pip install -r requirements.txt
python -m ipykernel install --user --name=taller_integrador
```

El nombre del kernel (`taller_integrador`) tiene que coincidir con el de `KERNEL` en `main.py`.

Un detalle sobre pip freeze: en Windows a veces guarda el archivo en UTF-16 si lo generás con PowerShell usando >, y eso rompe pip install -r. Si al instalar desde otro entorno te da un error de codificación, regeneralo desde cmd, o con:

```bash
pip freeze | Out-File -Encoding utf8 requirements.txt
```
## Configuración

### 1. Datos

Descargá a mano cada año desde el portal [BA Data - Departamentos en venta](https://data.buenosaires.gob.ar/dataset/departamentos-venta) (formato CSV; el portal bloquea las descargas automáticas) y copialos en `data/raw/` con este nombre:

```
departamentos-en-venta-2001.csv
departamentos-en-venta-2002.csv
...
departamentos-en-venta-2016.csv
```

El patrón se define en `config.py` (`RAW_FILE_PATTERN`).

### 2. MongoDB Atlas

1. Creá un cluster gratuito (M0).
2. En *Database Access*, creá un usuario de base de datos con contraseña (solo letras y números es lo más simple).
3. En *Network Access*, agregá tu IP.
4. En *Connect → Drivers*, copiá la cadena `mongodb+srv://...`.
5. Creá un archivo `.env` en la raíz del proyecto:

```
MONGO_URI=mongodb+srv://USUARIO:PASSWORD@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority
MONGO_DB=taller_integrador
```

Reemplazá `USUARIO` y `PASSWORD` por los reales, sin `< >`, sin comillas y sin espacios. Si la contraseña tiene símbolos, codificala con `urllib.parse.quote_plus`. **Nunca subas el `.env` a Git** (ya está en `.gitignore`).

Si Atlas no responde, el pipeline sigue funcionando y los logs quedan solo en `logs/pipeline.log`; los pasos que escriben en Mongo (`save_*`) sí fallan.

## Ejecución

Con el entorno virtual activado y desde la raíz del proyecto:

```bash
python main.py --rapido    # prueba de humo: solo Regresión Logística, sin escribir resultados en Mongo
python main.py             # corrida completa (el Random Forest tarda varios minutos)
python main.py --reusar    # reutiliza los modelos ya entrenados (models/results.joblib)
```

`main.py` usa Papermill: ejecuta `01_eda.ipynb` (carga, limpieza y guardado en SQLite) y `02_modelado.ipynb` (modelos, evaluación, escenario temporal, leakage y guardado en Mongo), y deja una copia con las salidas en `notebooks/ejecutados/`.

Los parámetros que se pueden cambiar están en la primera celda de `02_modelado.ipynb` (celda con tag `parameters`): tamaño de la muestra de búsqueda, cantidad de iteraciones, modelos a correr, escenario temporal, leakage, guardado en Mongo y uso de caché.

Para los gráficos, ejecutá `notebooks/03_visualizacion.ipynb` (lee los resultados desde Mongo y guarda las figuras en `reports/figures/`).

También podés abrir los notebooks en Jupyter o VS Code con el kernel `taller_integrador` y correrlos celda por celda.

### Dashboard

```bash
streamlit run app/dashboard.py
```

Necesita que ya se haya corrido el pipeline completo (SQLite y Mongo con datos) y el `.env` configurado.

## Resultados de referencia

Corrida `run_id` 20261110_022250 (test: 34.824 filas en el escenario aleatorio, 28.266 en el temporal).

| Modelo | Accuracy (aleatorio) | F1 macro (aleatorio) | Accuracy (temporal) | F1 macro (temporal) |
|---|---|---|---|---|
| Dummy (stratified) | 0,333 | 0,333 | 0,333 | 0,333 |
| Regresión Logística | 0,619 | 0,615 | 0,608 | 0,604 |
| Random Forest | 0,678 | 0,678 | 0,641 | 0,641 |

El detalle está en el informe técnico.

## Decisiones y limitaciones principales

- **Target relativo al año:** terciles de `PRECIO_M2` por año. No predice precios absolutos.
- **`PRECIO_M2` recalculado:** en 2016 el `U_S_M2` publicado usa los m² cubiertos, así que no es comparable con los otros años.
- **Sin leakage:** `DOLARES`, `U_S_M2`, `PRECIO_M2`, `AÑO`, `CALLE` y `NUMERO` no son predictoras. La prueba de leakage muestra cuánto subirían las métricas si se incluyeran.
- **Faltantes:** `AMBIENTES`, `BAÑOS`, `ANTIGUEDAD` y las comodidades no existen en todos los años. Los nulos se imputan dentro del `Pipeline` (con indicadores de faltante). Cerca del 50% de `ANTIGUEDAD` es imputado.
- **Random Forest sobreajusta** (train ≈ 0,80, test ≈ 0,68). La búsqueda de hiperparámetros usa una muestra de 20.000 filas y 8 combinaciones.
- **Avisos duplicados:** un mismo departamento puede estar publicado más de una vez y caer en train y test (no medido).
- **Descarga manual** de los datos por el bloqueo del portal.

## Aviso conocido en Windows

Al terminar `main.py` puede aparecer un `KeyError` en `joblib\externals\loky\backend\resource_tracker.py`. Es un mensaje de limpieza de carpetas temporales de joblib que ocurre después de guardar los resultados; no afecta a los datos ni a las bases.             