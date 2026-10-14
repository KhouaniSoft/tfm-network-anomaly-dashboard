# -*- coding: utf-8 -*-
"""Pipeline TFM: detección de anomalías en tráfico de red (UNSW-NB15) con Isolation Forest.

Protocolo: entrenamiento (solo normal) -> validación (fijar umbral) -> prueba (una sola vez).
Ver preprocesado.py para el detalle del preprocesado común con el cuadro de mando.
"""
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             confusion_matrix, roc_auc_score, roc_curve)
import preprocesado as pp

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "font.family": "sans-serif"})
OUT = {}


# ---------- Formato español en los ejes numéricos ----------
def _es(x, pos=None):
    if abs(x - round(x)) < 1e-9:
        return f"{int(round(x)):,}".replace(",", ".")
    return f"{x:.2f}".rstrip("0").rstrip(".").replace(".", ",")
ES = FuncFormatter(_es)
def es_x(*axs):
    for a in axs: a.xaxis.set_major_formatter(ES)
def es_y(*axs):
    for a in axs: a.yaxis.set_major_formatter(ES)


def metricas(y, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
            "precision": round(precision_score(y, pred, zero_division=0), 4),
            "recall": round(recall_score(y, pred, zero_division=0), 4),
            "f1": round(f1_score(y, pred, zero_division=0), 4),
            "accuracy": round((tp + tn) / (tp + tn + fp + fn), 4),
            "far": round(fp / (fp + tn), 4) if fp + tn else 0.0,
            "fnr": round(fn / (fn + tp), 4) if fn + tp else 0.0}


# ======================= 1. DESCRIPCIÓN BÁSICA (partición oficial de entrenamiento) =======================
# El análisis exploratorio completo está en eda.py.
train_raw, test_raw = pp.cargar()
# El análisis exploratorio se limita a la partición de entrenamiento: la de prueba
# no se inspecciona hasta que el modelo y el umbral están congelados (apartado 4).
OUT["shape_train"] = list(train_raw.shape)
OUT["nulls_train"] = int(train_raw.isna().sum().sum())
OUT["dups_train"] = int(train_raw.drop(columns=["id"]).duplicated().sum())  # sin "id", único por fila
OUT["dups_train_pct"] = round(OUT["dups_train"] / len(train_raw) * 100, 2)
OUT["label_dist_train"] = train_raw["label"].value_counts().to_dict()
OUT["attack_cat_train"] = train_raw["attack_cat"].value_counts().to_dict()
OUT["service_dash_pct"] = round(float((train_raw["service"] == "-").mean() * 100), 2)
OUT["n_proto"] = int(train_raw["proto"].nunique()); OUT["n_service"] = int(train_raw["service"].nunique())
OUT["n_state"] = int(train_raw["state"].nunique())
OUT["desc_numeric"] = (train_raw[["dur", "spkts", "dpkts", "sbytes", "dbytes", "rate"]].describe().T
                       [["mean", "std", "min", "50%", "max"]].round(3).to_dict("index"))
OUT["skew"] = train_raw[["dur", "sbytes", "dbytes", "spkts", "dpkts"]].skew().round(2).to_dict()

# Figura 1: clases y categorías
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
v = [OUT["label_dist_train"][0], OUT["label_dist_train"][1]]
ax[0].bar(["Normal", "Ataque"], v, color=["#4472C4", "#C00000"])
ax[0].set_title("Distribución de clases (entrenamiento)"); ax[0].set_ylabel("Registros"); es_y(ax[0])
for i, n in enumerate(v):
    ax[0].text(i, n, f"{n:,}".replace(",", "."), ha="center", va="bottom", fontsize=8)
train_raw["attack_cat"].value_counts().plot(kind="barh", ax=ax[1], color="#4472C4"); ax[1].invert_yaxis()
ax[1].set_title("Registros por categoría (entrenamiento)"); ax[1].set_xlabel("Registros")
ax[1].set_ylabel(""); es_x(ax[1])
plt.tight_layout(); plt.savefig("fig1_clases.png", bbox_inches="tight"); plt.close()

# Figura 2: distribuciones asimétricas
NOMBRES = {"dur": "Duración del flujo", "sbytes": "Bytes origen → destino", "rate": "Tasa de paquetes"}
fig, ax = plt.subplots(1, 3, figsize=(9, 3.0))
for a, col in zip(ax, ["dur", "sbytes", "rate"]):
    a.hist(np.log1p(train_raw[col]), bins=60, color="#4472C4", edgecolor="none")
    a.set_title(f"{NOMBRES[col]}\nlog(1 + {col})"); a.set_ylabel(""); es_x(a); es_y(a)
ax[0].set_ylabel("Frecuencia")
plt.tight_layout(); plt.savefig("fig2_distribuciones.png", bbox_inches="tight"); plt.close()

# ======================= 2. PROTOCOLO: deduplicación, división, preprocesado =======================
d = pp.preparar_todo()
dedup, tr, va, test, prep, modelo, umbral = (d[k] for k in
                                            ["dedup", "tr", "va", "test", "prep", "modelo", "umbral"])
normales_tr = tr[tr.label == 0]
OUT["protocolo"] = {
    "registros_tras_deduplicar_y_depurar": int(len(dedup)),
    "filas_eliminadas_por_etiquetas_contradictorias": d["n_contradictorias"],
    "entrenamiento": {"total": int(len(tr)), "normales": int((tr.label == 0).sum()),
                      "ataques_no_usados": int((tr.label == 1).sum())},
    "validacion": {"total": int(len(va)), "normales": int((va.label == 0).sum()),
                   "ataques": int((va.label == 1).sum())},
    "fraccion_validacion": pp.FRACCION_VALIDACION, "semilla": pp.SEMILLA,
}
OUT["preprocesado"] = {
    "ajustado_con": "tráfico normal del subconjunto de entrenamiento",
    "excluidas_explicitamente": pp.EXCLUIDAS,
    "constantes_en_normal": prep.constantes,
    "eliminadas_por_correlacion": prep.correladas,
    "transformadas_log": prep.sesgadas,
    "n_variables_finales": len(prep.variables),
    "variables_finales": prep.variables,
}

# Figura 3: correlación usada para la selección (tráfico normal de entrenamiento)
num_cols = [c for c in normales_tr.select_dtypes("number").columns
            if c != "label" and c not in prep.constantes]
fig, ax = plt.subplots(figsize=(7.5, 6.5))
im = ax.imshow(normales_tr[num_cols].corr(), cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(num_cols))); ax.set_xticklabels(num_cols, rotation=90, fontsize=6)
ax.set_yticks(range(len(num_cols))); ax.set_yticklabels(num_cols, fontsize=6)
cb = fig.colorbar(im, shrink=0.8); cb.ax.yaxis.set_major_formatter(ES)
ax.set_title("Matriz de correlación (Pearson), tráfico normal de entrenamiento")
plt.tight_layout(); plt.savefig("fig3_correlacion.png", bbox_inches="tight"); plt.close()

# ======================= 3. VALIDACIÓN: análisis de umbrales y elección =======================
sc_va_norm = va.loc[va.label == 0, "score"].values
barrido = {}
for pct in [80, 85, 90, 95, 99]:
    thr = float(np.percentile(sc_va_norm, pct))
    m = metricas(va.label, (va.score >= thr).astype(int))
    barrido[pct] = {"thr": round(thr, 4), **{k: m[k] for k in ["precision", "recall", "f1", "fp", "fn", "far"]}}
OUT["validacion_barrido_umbral"] = barrido

# Análisis de sensibilidad: dependencia del modelo respecto a sttl/dttl (solo en VALIDACIÓN).
# El análisis exploratorio muestra que el TTL separa casi por completo las clases en este
# conjunto de datos (artefacto del banco de pruebas). El modelo principal las conserva; aquí
# se mide cuánto rendimiento depende de ellas. No interviene en la selección del modelo.
from sklearn.metrics import roc_auc_score as _auc
def _variante(excluir):
    p = pp.Preprocesado(excluir_extra=excluir).ajustar(normales_tr)
    m = pp.entrenar(p.transformar(normales_tr))
    s_va = pp.puntuar(m, p.transformar(va))
    u = pp.umbral_por_far(s_va[va.label.values == 0])
    r = metricas(va.label, (s_va >= u).astype(int))
    return {"auc_validacion": round(float(_auc(va.label, s_va)), 4),
            **{k: r[k] for k in ["precision", "recall", "f1", "far"]}}
OUT["sensibilidad_ttl_validacion"] = {"con_ttl (modelo principal)": _variante([]),
                                      "sin_ttl": _variante(["sttl", "dttl"])}

m_va = metricas(va.label, (va.score >= umbral).astype(int))
OUT["umbral_elegido"] = {
    "criterio": f"tasa de falsas alarmas en validación <= {pp.OBJETIVO_FAR:.0%}",
    "objetivo_far": pp.OBJETIVO_FAR,
    "umbral": round(umbral, 4),
    "metricas_validacion": m_va,
}

# ======================= 4. PRUEBA: evaluación única con el umbral congelado =======================
# Modelo y umbral ya congelados: a partir de aquí se describe y evalúa la partición de prueba.
OUT["composicion_test"] = {
    "shape": list(test_raw.shape),
    "nulls": int(test_raw.isna().sum().sum()),
    "label_dist": test_raw["label"].value_counts().to_dict(),
    "attack_cat": test_raw["attack_cat"].value_counts().to_dict(),
}
pred = (test.score >= umbral).astype(int).values
y = test.label.values
OUT["roc_auc_test"] = round(float(roc_auc_score(y, test.score)), 4)
OUT["final_test"] = metricas(y, pred)
rc = {}
for cat in test["attack_cat"].unique():
    if cat == "Normal": continue
    mask = (test["attack_cat"] == cat).values
    rc[cat] = {"n": int(mask.sum()), "recall": round(float(pred[mask].mean()), 4)}
OUT["recall_por_categoria_test"] = dict(sorted(rc.items(), key=lambda x: -x[1]["recall"]))

# Figura 4: ROC y puntuaciones en prueba con el umbral congelado
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
fpr, tpr, _ = roc_curve(y, test.score)
ax[0].plot(fpr, tpr, color="#C00000", lw=1.5,
           label=f"Isolation Forest (AUC = {OUT['roc_auc_test']:.3f})".replace(".", ","))
ax[0].plot([0, 1], [0, 1], "--", color="gray", lw=1, label="Clasificador aleatorio")
ax[0].plot(OUT["final_test"]["far"], OUT["final_test"]["recall"], "o", color="black", ms=5,
           label="Umbral fijado en validación")
ax[0].set_xlabel("Tasa de falsas alarmas"); ax[0].set_ylabel("Cobertura (recall)"); es_x(ax[0]); es_y(ax[0])
ax[0].set_title("Curva ROC (conjunto de prueba)"); ax[0].legend(fontsize=8, loc="lower right")
n1, _, _ = ax[1].hist(test.score[y == 0], bins=80, alpha=0.65, label="Normal (etiqueta real)",
                      color="#4472C4", density=True)
n2, _, _ = ax[1].hist(test.score[y == 1], bins=80, alpha=0.65, label="Ataque (etiqueta real)",
                      color="#C00000", density=True)
ax[1].axvline(umbral, color="black", ls="--", lw=1, label="Umbral fijado en validación")
ax[1].set_ylim(0, max(n1.max(), n2.max()) * 1.35)
ax[1].set_xlabel("Puntuación de anomalía"); ax[1].set_ylabel("Densidad"); es_x(ax[1]); es_y(ax[1])
ax[1].set_title("Puntuaciones según la etiqueta real (prueba)")
ax[1].legend(fontsize=8, loc="upper right", framealpha=0.95)
plt.tight_layout(); plt.savefig("fig4_roc_scores.png", bbox_inches="tight"); plt.close()

# Figura 5: matriz de confusión y cobertura por familia (prueba)
f = OUT["final_test"]
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
cm = np.array([[f["tn"], f["fp"]], [f["fn"], f["tp"]]])
ax[0].imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        ax[0].text(j, i, f"{cm[i, j]:,}".replace(",", "."), ha="center", va="center",
                   color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=11)
ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(["Predicho normal", "Predicho anómalo"])
ax[0].set_yticks([0, 1]); ax[0].set_yticklabels(["Real: normal", "Real: ataque"])
ax[0].set_title("Matriz de confusión (conjunto de prueba)")
rcv = OUT["recall_por_categoria_test"]
ax[1].barh(list(rcv.keys()), [v["recall"] for v in rcv.values()], color="#4472C4")
ax[1].invert_yaxis(); ax[1].set_xlim(0, 1); ax[1].set_xlabel("Cobertura (recall)"); es_x(ax[1])
ax[1].set_title("Cobertura por categoría de ataque")
plt.tight_layout(); plt.savefig("fig5_confusion_categorias.png", bbox_inches="tight"); plt.close()

# ======================= 5. KPI (prueba, umbral congelado) =======================
det = test.assign(pred=pred)
OUT["kpis_test"] = {
    "volumen_total_GB": round(float((det.sbytes + det.dbytes).sum()) / 1e9, 2),
    "flujos_analizados": int(len(det)),
    "flujos_anomalos": int(det.pred.sum()),
    "pct_flujos_anomalos": round(float(det.pred.mean() * 100), 2),
    "tasa_falsas_alarmas_pct": round(f["far"] * 100, 2),
    "top_proto_anom": det[det.pred == 1]["proto"].value_counts().head(5).to_dict(),
    "top_service_anom": det[det.pred == 1]["service"].value_counts().head(5).to_dict(),
}

# Figura 6: anomalías por protocolo y servicio
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
det[det.pred == 1]["proto"].value_counts().head(6).plot(kind="bar", ax=ax[0], color="#C00000")
ax[0].set_title("Anomalías detectadas por protocolo"); ax[0].set_ylabel("Flujos anómalos")
ax[0].set_xlabel("Protocolo"); ax[0].tick_params(axis="x", rotation=30); es_y(ax[0])
det[det.pred == 1]["service"].value_counts().head(6).plot(kind="bar", ax=ax[1], color="#C00000")
ax[1].set_title("Anomalías detectadas por servicio"); ax[1].set_ylabel("Flujos anómalos")
ax[1].set_xlabel("Servicio"); ax[1].tick_params(axis="x", rotation=30); es_y(ax[1])
plt.tight_layout(); fig.align_xlabels(ax)   # "Protocolo" y "Servicio" a la misma altura
plt.savefig("fig6_kpis.png", bbox_inches="tight"); plt.close()

with open("resultados.json", "w", encoding="utf-8") as fh:
    json.dump(OUT, fh, indent=2, ensure_ascii=False, default=str)
print(json.dumps({k: OUT[k] for k in ["protocolo", "umbral_elegido", "roc_auc_test", "final_test"]},
                 indent=2, ensure_ascii=False, default=str))
