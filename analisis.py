# -*- coding: utf-8 -*-
"""Análisis complementarios del modelo (PEC3). No modifican el modelo ni el umbral.

  A. Sensibilidad a la configuración de Isolation Forest, solo en VALIDACIÓN:
     número de árboles (t) y tamaño de submuestra (psi), con diez semillas.
     La estabilidad de las decisiones entre semillas no usa etiquetas de ataque;
     el AUC y la cobertura son descriptivos (la configuración no se elige con ellos).
  B. Cobertura por familia de ataque en validación frente a prueba.
  C. Duplicados en la partición de prueba y métricas sin ellos.
  D. Tráfico normal: composición por TTL de origen (sttl), tasa de falsas alarmas y
     tramos de tráfico normal en el orden del fichero de prueba.

B, C y D se calculan con el modelo y el umbral ya congelados: son análisis
descriptivos a posteriori y no intervienen en ninguna decisión del modelo.

Ejecución:  python analisis.py
Salida:     analisis_resultados.json
"""
import itertools
import json
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
import preprocesado as pp

ARBOLES = [100, 300]              # 100: valor por defecto en Liu et al. (2008)
SUBMUESTRAS = [256, 512, 1024]    # 256: valor por defecto en Liu et al. (2008)
SEMILLAS = range(10)
OUT = {}

d = pp.preparar_todo()
tr, va, test, prep, umbral = (d[k] for k in ["tr", "va", "test", "prep", "umbral"])
normales_tr = tr[tr.label == 0]
normal_va = (va.label == 0).values

# ---------------- A. Sensibilidad a la configuración (validación) ----------------
X_tr, X_va = prep.transformar(normales_tr), prep.transformar(va)
filas, alertas = [], {}
for t, psi in itertools.product(ARBOLES, SUBMUESTRAS):
    for semilla in SEMILLAS:
        inicio = time.perf_counter()
        modelo = IsolationForest(n_estimators=t, max_samples=psi,
                                 random_state=semilla, n_jobs=-1).fit(X_tr)
        s = pp.puntuar(modelo, X_va)
        segundos = time.perf_counter() - inicio
        alerta = s >= pp.umbral_por_far(s[normal_va])      # mismo criterio: P90 del normal
        alertas[t, psi, semilla] = alerta
        filas.append({"arboles": t, "submuestra": psi, "auc": roc_auc_score(va.label, s),
                      "cobertura": alerta[~normal_va].mean(), "segundos": segundos})

# Estabilidad (sin etiquetas): proporción de flujos de validación cuya decisión
# (alerta / no alerta) cambia entre dos semillas, promediada sobre todos los pares.
cambios = {(t, psi): float(np.mean([(alertas[t, psi, a] != alertas[t, psi, b]).mean()
                                     for a, b in itertools.combinations(SEMILLAS, 2)]))
           for t, psi in itertools.product(ARBOLES, SUBMUESTRAS)}
res = pd.DataFrame(filas).groupby(["arboles", "submuestra"])
OUT["A_sensibilidad_configuracion"] = {
    "base": "validación; diez semillas por configuración; umbral P90 del tráfico normal",
    "configuracion_principal": {"arboles": 300, "submuestra": 512, "semilla": pp.SEMILLA},
    "resultados": [
        {"arboles": t, "submuestra": psi,
         "cambio_decision_entre_semillas_pct": round(cambios[t, psi] * 100, 2),
         "auc_media": round(g.auc.mean(), 4), "auc_desv": round(g.auc.std(), 4),
         "cobertura_media": round(g.cobertura.mean(), 4), "cobertura_desv": round(g.cobertura.std(), 4),
         "segundos_ajuste_y_puntuacion": round(g.segundos.mean(), 2)}
        for (t, psi), g in res],
}

# ---------------- B. Cobertura por familia: validación frente a prueba ----------------
va = va.assign(alerta=va.score >= umbral)
test = test.assign(alerta=test.score >= umbral)


def por_familia(df):
    ataques = df[df.label == 1]
    return ataques.groupby("attack_cat").alerta.agg(n="size", cobertura="mean")


fam = por_familia(va).join(por_familia(test), lsuffix="_val", rsuffix="_prueba", how="outer")
fam["peso_val"] = fam.n_val / fam.n_val.sum()
fam["peso_prueba"] = fam.n_prueba / fam.n_prueba.sum()
OUT["B_cobertura_por_familia"] = {
    "cobertura_validacion": round(float(va.loc[va.label == 1, "alerta"].mean()), 4),
    "cobertura_prueba": round(float(test.loc[test.label == 1, "alerta"].mean()), 4),
    # Cobertura que tendría la validación si sus familias pesaran como en la prueba
    "validacion_con_pesos_de_prueba": round(float((fam.cobertura_val * fam.peso_prueba).sum()), 4),
    "por_familia": fam.sort_values("n_prueba", ascending=False).round(4).to_dict("index"),
}

# ---------------- C. Duplicados en la partición de prueba ----------------
dup = test.drop(columns=["score", "alerta"]).duplicated(keep="first")
sin_dup = test[~dup]
y, a = sin_dup.label.values, sin_dup.alerta.values
OUT["C_prueba_sin_duplicados"] = {
    "duplicados": int(dup.sum()), "duplicados_pct": round(float(dup.mean() * 100), 2),
    "duplicados_pct_por_familia": (test[dup].attack_cat.value_counts() / test.attack_cat.value_counts()
                                   ).fillna(0).mul(100).round(2).to_dict(),
    "flujos": int(len(sin_dup)), "normales": int((y == 0).sum()), "ataques": int((y == 1).sum()),
    "auc": round(float(roc_auc_score(y, sin_dup.score)), 4),
    "cobertura": round(float(a[y == 1].mean()), 4),
    "tasa_falsas_alarmas": round(float(a[y == 0].mean()), 4),
    "precision": round(float(a[y == 1].sum() / a.sum()), 4),
    "cobertura_por_familia": sin_dup[sin_dup.label == 1].groupby("attack_cat").alerta.mean().round(4).to_dict(),
}

# ---------------- D. Tráfico normal: composición por sttl y falsas alarmas ----------------
normal = {"entrenamiento": normales_tr, "validacion": va[va.label == 0], "prueba": test[test.label == 0]}
peso = pd.DataFrame({k: v.sttl.value_counts(normalize=True) for k, v in normal.items()}).fillna(0)
far_val = normal["validacion"].groupby("sttl").alerta.mean()
far_prueba = normal["prueba"].groupby("sttl").alerta.mean()
# Falsas alarmas que tendría la validación con la composición por sttl del normal de prueba
w = peso["prueba"]
comunes = w.index.intersection(far_val.index)
principales = peso.max(axis=1).sort_values(ascending=False).head(4).index   # > 99 % del normal
OUT["D_trafico_normal_sttl"] = {
    "far_validacion": round(float(normal["validacion"].alerta.mean()), 4),
    "far_prueba": round(float(normal["prueba"].alerta.mean()), 4),
    "far_validacion_con_composicion_de_prueba": round(float((far_val[comunes] * w[comunes]).sum()
                                                            / w[comunes].sum()), 4),
    "peso_de_prueba_cubierto_pct": round(float(w[comunes].sum() * 100), 2),
    "por_valor_de_sttl": {
        int(v): {"normal_entrenamiento_pct": round(float(peso.loc[v, "entrenamiento"] * 100), 2),
                 "normal_prueba_pct": round(float(peso.loc[v, "prueba"] * 100), 2),
                 "far_validacion_pct": round(float(far_val.get(v, np.nan) * 100), 1),
                 "far_prueba_pct": round(float(far_prueba.get(v, np.nan) * 100), 1)}
        for v in principales},
}
# Tramos contiguos de tráfico normal en el orden del fichero de prueba (al menos 1.000 flujos):
# el fichero no tiene marcas de tiempo, pero el tráfico normal aparece en bloques con perfiles distintos.
pos = np.flatnonzero(test.label.to_numpy() == 0)
tramos = [t for t in np.split(pos, np.flatnonzero(np.diff(pos) > 1) + 1) if len(t) >= 1000]
OUT["D_trafico_normal_sttl"]["tramos_normales_en_el_fichero"] = [
    {"ids": f"{t[0] + 1}-{t[-1] + 1}", "flujos": int(len(t)),
     "far_pct": round(float(test.alerta.iloc[t].mean() * 100), 1),
     "sttl_31_pct": round(float((test.sttl.iloc[t] == 31).mean() * 100), 1)}
    for t in tramos]

with open("analisis_resultados.json", "w", encoding="utf-8") as fh:
    json.dump(OUT, fh, indent=2, ensure_ascii=False, default=str)

# Resumen breve en consola
for r in OUT["A_sensibilidad_configuracion"]["resultados"]:
    print(f"t={r['arboles']:>3} psi={r['submuestra']:>4} | cambio de decisión "
          f"{r['cambio_decision_entre_semillas_pct']:>5} % | AUC {r['auc_media']} ± {r['auc_desv']} | "
          f"cobertura {r['cobertura_media']} | {r['segundos_ajuste_y_puntuacion']} s")
b, c, dd = OUT["B_cobertura_por_familia"], OUT["C_prueba_sin_duplicados"], OUT["D_trafico_normal_sttl"]
print(f"Cobertura: validación {b['cobertura_validacion']} | prueba {b['cobertura_prueba']} | "
      f"prueba sin duplicados {c['cobertura']}")
print(f"Falsas alarmas: validación {dd['far_validacion']} | prueba {dd['far_prueba']} | "
      f"validación con la composición de prueba {dd['far_validacion_con_composicion_de_prueba']}")
for t in dd["tramos_normales_en_el_fichero"]:
    print(f"Tramo normal {t['ids']}: {t['flujos']} flujos | falsas alarmas {t['far_pct']} % | "
          f"sttl = 31 en el {t['sttl_31_pct']} %")
print("Detalle en analisis_resultados.json")
