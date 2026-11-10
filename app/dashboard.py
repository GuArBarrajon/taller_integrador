"""Dashboard de Streamlit. Ejecutar desde la raíz del proyecto:
 
    streamlit run app/dashboard.py
 
Lee los datos limpios de SQLite y los resultados de MongoDB (corrida más reciente).
"""
import sys
from pathlib import Path
 
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
 
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
 
import config
from src.db_mongo import get_db, load_run
from src.db_sql import load_clean_data
 
LABELS = config.SEGMENT_LABELS
COLORS = {"bajo": "#4C78A8", "medio": "#F2B134", "alto": "#E45756"}
 
st.set_page_config(page_title="Departamentos en venta CABA", layout="wide")
 
 
# ---------- carga de datos (con caché) ----------
@st.cache_data(show_spinner="Leyendo SQLite...")
def get_clean() -> pd.DataFrame:
    return load_clean_data()
 
 
@st.cache_data(ttl=600, show_spinner="Leyendo MongoDB...")
def get_mongo(collection: str, escenario: str) -> pd.DataFrame:
    return load_run(collection, escenario)
 
 
@st.cache_data(ttl=300, show_spinner="Leyendo logs...")
def get_logs(limit: int = 2000) -> pd.DataFrame:
    col = get_db()["logs"]
    last = col.find_one(sort=[("timestamp", -1)], projection={"run_id": 1})
    if last is None:
        return pd.DataFrame()
    docs = col.find({"run_id": last["run_id"]}, {"_id": 0}).sort("timestamp", 1).limit(limit)
    return pd.DataFrame(list(docs))
 
 
def mongo_or_stop(collection: str, escenario: str) -> pd.DataFrame:
    try:
        return get_mongo(collection, escenario)
    except Exception as e:
        st.error(f"No se pudo leer '{collection}' ({escenario}) desde MongoDB: {e}")
        st.stop()
 
 
# ---------- encabezado y filtros ----------
st.title("Departamentos en venta en CABA (2001-2016)")
st.caption("Clasificación del segmento de valor (bajo / medio / alto) a partir de las características del departamento.")
 
df = get_clean()
 
st.sidebar.header("Filtros (pestaña Datos)")
y0, y1 = st.sidebar.slider("Años", int(df["AÑO"].min()), int(df["AÑO"].max()),
                           (int(df["AÑO"].min()), int(df["AÑO"].max())))
barrios = st.sidebar.multiselect("Barrios", sorted(df["BARRIO"].unique()))
segmentos = st.sidebar.multiselect("Segmento", LABELS, default=LABELS)
 
f = df[df["AÑO"].between(y0, y1) & df["SEGMENTO"].isin(segmentos)]
if barrios:
    f = f[f["BARRIO"].isin(barrios)]
 
tab_datos, tab_modelos, tab_pred, tab_leak, tab_logs = st.tabs(
    ["Datos", "Modelos", "Predicciones", "Leakage", "Logs"])
 
# ---------- pestaña Datos ----------
with tab_datos:
    if f.empty:
        st.warning("Los filtros no dejan ninguna fila.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Avisos", f"{len(f):,}".replace(",", "."))
        c2.metric("Barrios", f["BARRIO"].nunique())
        c3.metric("Superficie mediana (m²)", f"{f['M2'].median():.0f}")
        c4.metric("USD/m² mediano", f"{f['PRECIO_M2'].median():,.0f}".replace(",", "."))
 
        left, right = st.columns(2)
        with left:
            cnt = f.groupby(["AÑO", "SEGMENTO"]).size().reset_index(name="avisos")
            st.plotly_chart(px.bar(cnt, x="AÑO", y="avisos", color="SEGMENTO",
                                   color_discrete_map=COLORS, category_orders={"SEGMENTO": LABELS},
                                   title="Avisos por año y segmento"), use_container_width=True)
        with right:
            med = f.groupby("AÑO")["PRECIO_M2"].median().reset_index()
            st.plotly_chart(px.line(med, x="AÑO", y="PRECIO_M2", markers=True,
                                    title="USD/m² mediano por año"), use_container_width=True)
 
        left, right = st.columns(2)
        with left:
            top = (f.groupby("BARRIO")["PRECIO_M2"].median().sort_values(ascending=False)
                   .head(15).reset_index())
            st.plotly_chart(px.bar(top, x="PRECIO_M2", y="BARRIO", orientation="h",
                                   title="Barrios con mayor USD/m² mediano (top 15)")
                            .update_yaxes(autorange="reversed"), use_container_width=True)
        with right:
            feats = config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES
            nulos = (f[feats].isnull().groupby(f["AÑO"]).mean() * 100).round(0)
            st.plotly_chart(px.imshow(nulos.T, text_auto=True, aspect="auto", color_continuous_scale="Reds",
                                      title="% de nulos por variable y año", labels={"color": "% nulos"}),
                            use_container_width=True)
 
        geo = f.dropna(subset=["LAT", "LON"])
        geo = geo.sample(min(5000, len(geo)), random_state=config.RANDOM_STATE)
        if not geo.empty:
            sin_fondo = st.checkbox("Mapa sin fondo (por si no cargan los mosaicos)")
            kw = dict(color="SEGMENTO", color_discrete_map=COLORS,
                    category_orders={"SEGMENTO": LABELS}, hover_data=["BARRIO", "M2", "PRECIO_M2"])
            titulo = "Ubicación de los avisos (muestra de hasta 5.000)"
            if sin_fondo:
                fig = px.scatter(geo, x="LON", y="LAT", title=titulo, height=520, **kw)
                fig.update_yaxes(scaleanchor="x", scaleratio=1)   # evita que se deforme
            elif hasattr(px, "scatter_map"):
                fig = px.scatter_map(geo, lat="LAT", lon="LON", zoom=10.5, height=520, title=titulo, **kw)
                fig.update_layout(map_style="open-street-map", margin=dict(l=0, r=0, t=40, b=0))
            else:
                fig = px.scatter_mapbox(geo, lat="LAT", lon="LON", zoom=10.5, height=520, title=titulo, **kw)
                fig.update_layout(mapbox_style="open-street-map", margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Con los filtros elegidos no hay filas con coordenadas.")
 
# ---------- pestaña Modelos ----------
with tab_modelos:
    escenario = st.radio("Escenario", ["aleatorio", "temporal"], horizontal=True,
                         help="aleatorio: split 80/20 | temporal: train 2001-2013, test 2014-2016")
    met = mongo_or_stop("metricas", escenario)
    cols = ["model", "accuracy", "precision_macro", "recall_macro", "f1_macro"]
    st.dataframe(met[cols].round(4), use_container_width=True, hide_index=True)
 
    long = met.melt(id_vars="model", value_vars=cols[1:], var_name="métrica", value_name="valor")
    st.plotly_chart(px.bar(long, x="métrica", y="valor", color="model", barmode="group",
                           range_y=[0, 1], title=f"Comparación de modelos ({escenario})"),
                    use_container_width=True)
 
    modelo = st.selectbox("Matriz de confusión del modelo", met["model"].tolist(),
                          index=max(0, met["model"].tolist().index("random_forest"))
                          if "random_forest" in met["model"].tolist() else 0)
    row = met[met["model"] == modelo].iloc[0]
    cm = np.array(row["confusion_matrix"])
    pct = cm / cm.sum(axis=1, keepdims=True)
    left, right = st.columns([3, 2])
    with left:
        fig = px.imshow(pct, x=LABELS, y=LABELS, text_auto=".0%", color_continuous_scale="Blues",
                        zmin=0, zmax=1, labels=dict(x="Predicho", y="Real", color="% por fila"),
                        title=f"{modelo}: matriz de confusión (porcentaje por fila)")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        pc = pd.DataFrame(row["per_class"]).T.round(3)
        pc["support"] = pc["support"].astype(int)
        st.markdown("**Métricas por clase**")
        st.dataframe(pc, use_container_width=True)
        st.caption(f"Matriz en cantidades (filas = real): {cm.tolist()}")
 
    with st.expander("Parámetros del modelo (Mongo: colección parametros)"):
        par = mongo_or_stop("parametros", "comun")
        st.dataframe(par[["modelo", "mejores_parametros", "cv_score", "scoring", "cv_folds",
                          "segundos_busqueda", "segundos_ajuste_final", "random_state"]],
                     use_container_width=True, hide_index=True)
 
    fig_path = config.FIGURES_DIR / "importancia_variables.png"
    if fig_path.exists():
        with st.expander("Importancia de variables (permutación, Random Forest)"):
            st.image(str(fig_path))
 
# ---------- pestaña Predicciones ----------
with tab_pred:
    st.caption("Carga las predicciones guardadas en MongoDB (decenas de miles de documentos).")
    if st.checkbox("Cargar predicciones"):
        esc = st.radio("Escenario", ["aleatorio", "temporal"], horizontal=True, key="esc_pred")
        pred = mongo_or_stop("predicciones", esc)
        modelos = sorted(pred["modelo"].unique())
        mod = st.selectbox("Modelo", modelos, index=modelos.index("random_forest") if "random_forest" in modelos else 0)
        p = pred[pred["modelo"] == mod].copy()
        p["AÑO"] = df.loc[p["id_fila"], "AÑO"].to_numpy()
        p["BARRIO"] = df.loc[p["id_fila"], "BARRIO"].to_numpy()
 
        left, right = st.columns(2)
        with left:
            por_anio = p.groupby("AÑO")["acierto"].mean().reset_index()
            st.plotly_chart(px.bar(por_anio, x="AÑO", y="acierto", range_y=[0, 1],
                                   title=f"Accuracy por año ({mod}, {esc})"), use_container_width=True)
        with right:
            g = p.groupby("BARRIO")["acierto"].agg(["mean", "size"])
            g = g[g["size"] >= 100].sort_values("mean").reset_index()
            sel = pd.concat([g.head(8), g.tail(8)])
            st.plotly_chart(px.bar(sel, x="mean", y="BARRIO", orientation="h", range_x=[0, 1],
                                   title="Accuracy por barrio (8 más bajos y 8 más altos; ≥100 avisos)",
                                   labels={"mean": "accuracy"}), use_container_width=True)
 
        st.markdown("**Muestra de predicciones**")
        cols_p = ["id_fila", "AÑO", "BARRIO", "y_real", "y_pred", "acierto"] + \
                 [c for c in p.columns if c.startswith("proba_")]
        solo_errores = st.checkbox("Mostrar solo errores")
        ver = p[~p["acierto"]] if solo_errores else p
        st.dataframe(ver[cols_p].head(300), use_container_width=True, hide_index=True)
 
# ---------- pestaña Leakage ----------
with tab_leak:
    st.write("Mismo Random Forest y mismo split; solo cambian las predictoras. "
             "Las variantes con variables de precio no se usan en el modelo real.")
    leak = mongo_or_stop("leakage", "leakage").sort_values("accuracy")
    fig = px.bar(leak, x="accuracy", y="variante", orientation="h", range_x=[0, 1],
                 title="Accuracy según las predictoras incluidas")
    fig.add_vline(x=1 / 3, line_dash="dash", annotation_text="azar (0,333)")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(leak[["variante", "accuracy", "f1_macro", "delta_vs_modelo_real"]].round(3),
                 use_container_width=True, hide_index=True)
 
# ---------- pestaña Logs ----------
with tab_logs:
    try:
        logs = get_logs()
    except Exception as e:
        st.error(f"No se pudieron leer los logs: {e}")
        logs = pd.DataFrame()
    if logs.empty:
        st.info("No hay logs en MongoDB.")
    else:
        st.caption(f"Última ejecución con logs: run_id {logs['run_id'].iloc[0]} ({len(logs)} mensajes, máximo 2.000)")
        niveles = st.multiselect("Nivel", sorted(logs["level"].unique()),
                                 default=[n for n in ("WARNING", "ERROR", "CRITICAL") if n in set(logs["level"])])
        texto = st.text_input("Buscar en el mensaje")
        v = logs[logs["level"].isin(niveles)] if niveles else logs
        if texto:
            v = v[v["message"].str.contains(texto, case=False, na=False)]
        st.dataframe(v[["timestamp", "level", "module", "message"]], use_container_width=True, hide_index=True)
