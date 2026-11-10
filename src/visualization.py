import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance

from config import FIGURES_DIR, SEGMENT_LABELS, RANDOM_STATE
from src.logger_setup import get_logger

logger = get_logger(__name__)
sns.set_theme(style="whitegrid")


def _save(fig, name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    logger.info("Figura guardada: %s", path.name)


def plot_price_by_year(df):
    """Por qué los terciles se calculan por año: el precio por m² cambia mucho en el tiempo."""
    q = df.groupby("AÑO")["PRECIO_M2"].quantile([1/3, 0.5, 2/3]).unstack()
    q.columns = ["tercil 1/3", "mediana", "tercil 2/3"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    q.plot(ax=ax, marker="o")
    ax.set(title="PRECIO_M2 (USD/m²) por año y cortes del target", ylabel="USD/m²")
    _save(fig, "precio_m2_por_anio")
    return q


def plot_confusion_matrices(metrics: pd.DataFrame, escenario: str):
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.2))
    axes = np.atleast_1d(axes)
    for ax, (_, r) in zip(axes, metrics.iterrows()):
        cm = np.array(r["confusion_matrix"])
        pct = cm / cm.sum(axis=1, keepdims=True)
        annot = np.array([[f"{cm[i, j]}\n{pct[i, j]:.0%}" for j in range(3)] for i in range(3)])
        sns.heatmap(pct, annot=annot, fmt="", cmap="Blues", vmin=0, vmax=1, cbar=False,
                    xticklabels=SEGMENT_LABELS, yticklabels=SEGMENT_LABELS, ax=ax)
        ax.set(title=f"{r['model']} (acc {r['accuracy']:.3f})", xlabel="Predicho", ylabel="Real")
    fig.suptitle(f"Matrices de confusión - escenario {escenario} (porcentaje por fila)")
    _save(fig, f"matrices_confusion_{escenario}")


def plot_model_comparison(metrics_all: pd.DataFrame):
    cols = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    long = metrics_all.melt(id_vars=["escenario", "model"], value_vars=cols,
                            var_name="métrica", value_name="valor")
    g = sns.catplot(data=long, x="métrica", y="valor", hue="model", col="escenario",
                    kind="bar", height=4, aspect=1.4)
    g.set(ylim=(0, 1))
    g.figure.suptitle("Comparación de modelos", y=1.03)
    _save(g.figure, "comparacion_modelos")


def plot_accuracy_by_year(preds: pd.DataFrame, df: pd.DataFrame):
    p = preds.copy()
    p["AÑO"] = df.loc[p["id_fila"], "AÑO"].to_numpy()
    t = p.groupby(["AÑO", "modelo"])["acierto"].mean().unstack()
    fig, ax = plt.subplots(figsize=(7, 4))
    t.plot(kind="bar", ax=ax)
    ax.set(title="Accuracy por año (escenario temporal)", ylabel="accuracy", ylim=(0, 1))
    ax.tick_params(axis="x", rotation=0)
    _save(fig, "accuracy_por_anio_temporal")
    return t.round(3)


def plot_permutation_importance(model, X_test, y_test, n=8000):
    """Importancia por permutación sobre las 12 predictoras originales."""
    idx = X_test.sample(min(n, len(X_test)), random_state=RANDOM_STATE).index
    res = permutation_importance(model, X_test.loc[idx], y_test.loc[idx], scoring="accuracy",
                                 n_repeats=3, random_state=RANDOM_STATE, n_jobs=1)
    imp = (pd.DataFrame({"variable": X_test.columns, "caida_accuracy": res.importances_mean,
                         "desvio": res.importances_std})
           .sort_values("caida_accuracy", ascending=True))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(imp["variable"], imp["caida_accuracy"], xerr=imp["desvio"])
    ax.set(title="Importancia por permutación (Random Forest)", xlabel="caída de accuracy")
    _save(fig, "importancia_variables")
    return imp.iloc[::-1].reset_index(drop=True)


def plot_leakage(leak: pd.DataFrame):
    d = leak.sort_values("accuracy")
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.barh(d["variante"], d["accuracy"])
    ax.axvline(1 / 3, color="gray", ls="--", label="azar (0,333)")
    ax.set(title="Prueba de leakage: accuracy según predictoras", xlabel="accuracy", xlim=(0, 1))
    ax.legend()
    _save(fig, "leakage")