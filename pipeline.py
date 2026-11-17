# -*- coding: utf-8 -*-
"""Pipeline TFM: detección de anomalías en tráfico de red (UNSW-NB15) con Isolation Forest.

Protocolo: entrenamiento (solo normal) -> validación (fijar umbral) -> prueba (una sola vez).
El análisis exploratorio está en eda.py y los análisis complementarios en analisis.py.
El preprocesado, el modelo y los KPI son comunes con el cuadro de mando (preprocesado.py).

Ejecución:  python pipeline.py
Salida:     resultados.json y figuras fig1_roc_puntuaciones.png y fig2_confusion_familias.png
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             confusion_matrix, roc_auc_score, roc_curve)
import preprocesado as pp

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


# ======================= 1. PROTOCOLO: deduplicación, división, preprocesado =======================
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

# ======================= 2. VALIDACIÓN: barrido de umbrales, sensibilidad y umbral =======================
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
def _variante(excluir):
    p = pp.Preprocesado(excluir_extra=excluir).ajustar(normales_tr)
    m = pp.entrenar(p.transformar(normales_tr))
    s_va = pp.puntuar(m, p.transformar(va))
    u = pp.umbral_por_far(s_va[va.label.values == 0])
    r = metricas(va.label, (s_va >= u).astype(int))
    return {"auc_validacion": round(float(roc_auc_score(va.label, s_va)), 4),
            **{k: r[k] for k in ["precision", "recall", "f1", "far"]}}


OUT["sensibilidad_ttl_validacion"] = {"con_ttl (modelo principal)": _variante([]),
                                      "sin_ttl": _variante(["sttl", "dttl"])}

PCT_REF = int(round((1 - pp.OBJETIVO_FAR) * 100))
m_va = metricas(va.label, (va.score >= umbral).astype(int))
OUT["umbral_elegido"] = {
    "criterio": f"percentil {PCT_REF} de la puntuación del tráfico normal de validación "
                f"(objetivo de falsas alarmas del {pp.OBJETIVO_FAR * 100:.0f} %)",
    "objetivo_far": pp.OBJETIVO_FAR,
    "umbral": round(umbral, 4),
    "metricas_validacion": m_va,
}

# ======================= 3. PRUEBA: evaluación única con el umbral congelado =======================
# Modelo y umbral ya congelados: a partir de aquí se describe y evalúa la partición de prueba.
test_raw = d["test_raw"]
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

# Figura 1: ROC y puntuaciones en prueba con el umbral congelado.
# Las dos clases se distinguen por el tipo de línea, no solo por el color.
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
fpr, tpr, _ = roc_curve(y, test.score)
ax[0].plot(fpr, tpr, color="#C00000", lw=1.5,
           label=f"Isolation Forest (AUC = {OUT['roc_auc_test']:.3f})".replace(".", ","))
ax[0].plot([0, 1], [0, 1], "--", color="gray", lw=1, label="Clasificador aleatorio")
ax[0].plot(OUT["final_test"]["far"], OUT["final_test"]["recall"], "o", color="black", ms=5,
           label="Umbral fijado en validación")
ax[0].set_xlabel("Tasa de falsas alarmas"); ax[0].set_ylabel("Cobertura (recall)"); es_x(ax[0]); es_y(ax[0])
ax[0].set_title("Curva ROC (conjunto de prueba)"); ax[0].legend(fontsize=8, loc="lower right")
bins = np.linspace(test.score.min(), test.score.max(), 81)
centros = (bins[:-1] + bins[1:]) / 2
dens_n, _ = np.histogram(test.score[y == 0], bins=bins, density=True)
dens_a, _ = np.histogram(test.score[y == 1], bins=bins, density=True)
ax[1].fill_between(centros, dens_n, step="mid", color="#4472C4", alpha=0.25, lw=0)
ax[1].plot(centros, dens_n, drawstyle="steps-mid", color="#4472C4", lw=1.3, ls="-",
           label="Normal (etiqueta real)")
ax[1].plot(centros, dens_a, drawstyle="steps-mid", color="#C00000", lw=1.5, ls=(0, (4, 2)),
           label="Ataque (etiqueta real)")
ax[1].axvline(umbral, color="black", ls=":", lw=1.2, label="Umbral fijado en validación")
ax[1].set_ylim(0, max(dens_n.max(), dens_a.max()) * 1.35)
ax[1].set_xlabel("Puntuación de anomalía"); ax[1].set_ylabel("Densidad"); es_x(ax[1]); es_y(ax[1])
ax[1].set_title("Puntuaciones según la etiqueta real (prueba)")
ax[1].legend(fontsize=8, loc="upper right", framealpha=0.95)
plt.tight_layout(); plt.savefig("fig1_roc_puntuaciones.png", bbox_inches="tight"); plt.close()

# Figura 2: matriz de confusión y cobertura por familia (prueba)
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
plt.tight_layout(); plt.savefig("fig2_confusion_familias.png", bbox_inches="tight"); plt.close()

# ======================= 4. KPI (prueba completa, umbral de referencia) =======================
# Misma función que el cuadro de mando (apartado 2.6 de la memoria): las cifras coinciden.
k = pp.kpis(test, umbral)
anomalos = test[pred == 1]
OUT["kpis_test"] = {
    "base": f"partición de prueba completa; umbral de referencia (P{PCT_REF})",
    "I1_flujos_analizados": k["flujos"],
    "I2_volumen_GB": round(k["volumen_gb"], 2),
    "I3_flujos_anomalos": k["anomalos"],
    "I4_pct_flujos_anomalos": round(k["pct_anomalos"], 2),
    "I5_tasa_falsas_alarmas_pct": round(k["tasa_falsas_alarmas_pct"], 2),
    "I6_cobertura_pct": round(k["cobertura_pct"], 2),
    "anomalos_por_protocolo_top5": anomalos["proto"].value_counts().head(5).to_dict(),
    "anomalos_por_servicio_top5": anomalos["service"].value_counts().head(5).to_dict(),
}

with open("resultados.json", "w", encoding="utf-8") as fh:
    json.dump(OUT, fh, indent=2, ensure_ascii=False, default=str)
print(json.dumps({k: OUT[k] for k in ["protocolo", "umbral_elegido", "roc_auc_test", "final_test", "kpis_test"]},
                 indent=2, ensure_ascii=False, default=str))
