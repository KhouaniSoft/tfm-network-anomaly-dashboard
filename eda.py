# -*- coding: utf-8 -*-
"""Análisis exploratorio de datos (T2.2, OE2) sobre la partición oficial de ENTRENAMIENTO.

La partición de prueba no se inspecciona aquí: queda reservada para la evaluación final.

Cada bloque responde a una pregunta que justifica una decisión del preprocesado (2.4):
  P1. Estructura y calidad: ¿hay nulos, valores imposibles o variables constantes?
  P2. Duplicados: ¿cuántos hay y cómo se reparten por familia de ataque?
  P3. Etiquetas contradictorias: ¿hay flujos idénticos con etiquetas distintas?
  P4. Variables categóricas: ¿cuántas categorías tienen y cuántas son raras?
  P5. Asimetría: ¿qué variables continuas necesitan transformación?
  P6. Redundancia: ¿qué pares de variables están muy correlacionados?
  P7. Variables a revisar: stcpb/dtcpb (secuencia TCP) y sttl/dttl (TTL).

Ejecución:  python eda.py
Salida:     eda_resultados.json y figuras eda_fig*.png
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

RUTA = "data/UNSW_NB15_training-set.csv"
CATEGORICAS = ["proto", "service", "state"]
ETIQUETAS = ["label", "attack_cat"]
UMBRAL_SESGO = 3
UMBRAL_CORRELACION = 0.95
UMBRAL_RARA = 0.001          # categoría "rara": menos del 0,1 % de los flujos
plt.rcParams.update({"figure.dpi": 150, "font.size": 9})
OUT = {}


def pct(x):
    return round(float(x) * 100, 2)


df = pd.read_csv(RUTA).drop(columns=["id"])   # "id" es un índice de fila, no un atributo
NUM = [c for c in df.select_dtypes("number").columns if c != "label"]
ATRIB = NUM + CATEGORICAS                        # atributos del flujo (sin etiquetas)

# ---------------- P1. Estructura y calidad ----------------
OUT["P1_estructura"] = {
    "filas": len(df), "columnas": df.shape[1],
    "numericas": len(NUM), "categoricas": CATEGORICAS, "etiquetas": ETIQUETAS,
    "nulos_por_columna": {c: int(n) for c, n in df.isna().sum().items() if n > 0},
    "negativos_por_columna": {c: int((df[c] < 0).sum()) for c in NUM if (df[c] < 0).any()},
    "constantes": [c for c in NUM if df[c].nunique() <= 1],
    "valores_distintos": {c: int(df[c].nunique()) for c in NUM},
    "binarias_o_casi": [c for c in NUM if df[c].nunique() <= 2],
    "clases": df["label"].value_counts().to_dict(),
    "familias": df["attack_cat"].value_counts().to_dict(),
}

# ---------------- P2. Duplicados por familia ----------------
dup = df.duplicated(keep="first")
dedup = df[~dup]
fam = pd.DataFrame({
    "original": df["attack_cat"].value_counts(),
    "tras_deduplicar": dedup["attack_cat"].value_counts(),
})
fam["duplicados_pct"] = ((1 - fam["tras_deduplicar"] / fam["original"]) * 100).round(2)
fam["peso_original_pct"] = (fam["original"] / fam["original"].sum() * 100).round(2)
fam["peso_dedup_pct"] = (fam["tras_deduplicar"] / fam["tras_deduplicar"].sum() * 100).round(2)
fam = fam.sort_values("original", ascending=False)
OUT["P2_duplicados"] = {
    "total": int(dup.sum()), "total_pct": pct(dup.mean()),
    "filas_tras_deduplicar": len(dedup),
    "por_familia": fam.to_dict("index"),
}

fig, ax = plt.subplots(figsize=(7, 3.4))
y = np.arange(len(fam))
ax.barh(y - 0.2, fam["peso_original_pct"], 0.4, label="Original", color="#4472C4")
ax.barh(y + 0.2, fam["peso_dedup_pct"], 0.4, label="Tras deduplicar", color="#ED7D31", hatch="//")
ax.set_yticks(y); ax.set_yticklabels(fam.index); ax.invert_yaxis()
ax.set_xlabel("Peso en la partición (%)"); ax.legend()
ax.set_title("Efecto de la deduplicación por familia (entrenamiento)")
plt.tight_layout(); plt.savefig("eda_fig1_duplicados.png", bbox_inches="tight"); plt.close()

# ---------------- P3. Etiquetas contradictorias ----------------
# Flujos con atributos idénticos y etiqueta binaria distinta
grupos = dedup.groupby(ATRIB, dropna=False)["label"].nunique()
conflictivos = grupos[grupos > 1]
claves = conflictivos.index.to_frame(index=False)
filas_conf = dedup.merge(claves, on=ATRIB, how="inner")
OUT["P3_etiquetas_contradictorias"] = {
    "grupos_conflictivos": int(len(conflictivos)),
    "filas_implicadas": int(len(filas_conf)),
    "filas_implicadas_pct_dedup": pct(len(filas_conf) / len(dedup)),
    "familias_implicadas": filas_conf["attack_cat"].value_counts().to_dict(),
}

# ---------------- P4. Variables categóricas ----------------
cat = {}
for c in CATEGORICAS:
    frec = df[c].value_counts(normalize=True)
    cat[c] = {
        "categorias": int(frec.size),
        "top5_pct": (frec.head(5) * 100).round(2).to_dict(),
        "raras_n": int((frec < UMBRAL_RARA).sum()),
        "raras_flujos_pct": pct(frec[frec < UMBRAL_RARA].sum()),
        "categorias_solo_en_ataque": int(len(set(df.loc[df.label == 1, c]) - set(df.loc[df.label == 0, c]))),
    }
cat["service"]["sin_identificar_pct"] = pct((df["service"] == "-").mean())
OUT["P4_categoricas"] = cat

# ---------------- P5. Asimetría (tráfico normal deduplicado, solo variables no binarias) ----------------
# Misma base que P6 y que el preprocesado, que ajusta las transformaciones con tráfico normal.
# En la partición completa los ataques cambian la asimetría (p. ej., trans_depth: 167,34 frente a 2,85).
normal = dedup[dedup.label == 0]
no_bin = [c for c in NUM if normal[c].nunique() > 2]
sesgo = normal[no_bin].skew().round(2).sort_values(ascending=False)
candidatas = sesgo[sesgo.abs() > UMBRAL_SESGO].index.tolist()
# Efecto de la transformación: asimetría de las candidatas después de aplicar log(1 + x)
tras_log = np.log1p(normal[candidatas].clip(lower=0)).skew().round(2)
OUT["P5_asimetria"] = {
    "base": "tráfico normal deduplicado",
    "criterio": f"|asimetría| > {UMBRAL_SESGO}, solo variables con más de dos valores",
    "asimetria": sesgo.to_dict(),
    "candidatas_log": candidatas,
    "asimetria_tras_log": tras_log.to_dict(),
    "mediana_abs_candidatas_antes": round(float(sesgo[candidatas].abs().median()), 2),
    "mediana_abs_candidatas_tras_log": round(float(tras_log.abs().median()), 2),
    "siguen_por_encima_del_umbral_tras_log": tras_log[tras_log.abs() > UMBRAL_SESGO].index.tolist(),
}

fig, ax = plt.subplots(figsize=(7, 6.4))
posicion = np.arange(len(sesgo))
colores = ["#C00000" if c in candidatas else "#A6A6A6" for c in sesgo.index]
ax.barh(posicion, sesgo.values, color=colores)
ax.scatter(tras_log[candidatas].values, [sesgo.index.get_loc(c) for c in candidatas],
           marker="D", s=14, color="black", zorder=3)
ax.set_yticks(posicion); ax.set_yticklabels(sesgo.index); ax.invert_yaxis()
ax.axvline(UMBRAL_SESGO, ls="--", color="black", lw=1)
ax.set_xscale("symlog"); ax.set_xlabel("Coeficiente de asimetría (escala symlog)")
ax.tick_params(axis="y", labelsize=6)
ax.legend(handles=[
    Patch(color="#C00000", label=f"Candidata a log(1 + x) (|asimetría| > {UMBRAL_SESGO})"),
    Patch(color="#A6A6A6", label="Sin transformar"),
    Line2D([], [], marker="D", ls="", color="black", ms=4, label="Asimetría tras log(1 + x)"),
    Line2D([], [], ls="--", color="black", lw=1, label=f"Umbral = {UMBRAL_SESGO}")],
    fontsize=7, loc="lower right")
ax.set_title("Asimetría antes y después de log(1 + x) (tráfico normal deduplicado)")
plt.tight_layout(); plt.savefig("eda_fig2_asimetria.png", bbox_inches="tight"); plt.close()

# ---------------- P6. Redundancia (tráfico normal deduplicado) ----------------
corr = normal[[c for c in NUM if normal[c].nunique() > 1]].corr()
sup = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool)).stack()
pares = sup[sup.abs() > UMBRAL_CORRELACION].sort_values(key=abs, ascending=False)
OUT["P6_redundancia"] = {
    "base": "tráfico normal deduplicado",
    "criterio": f"|r de Pearson| > {UMBRAL_CORRELACION}",
    "pares": [{"a": a, "b": b, "r": round(float(r), 3)} for (a, b), r in pares.items()],
}

# ---------------- P7. Variables a revisar ----------------
OUT["P7_secuencia_tcp"] = {
    c: {"valores_distintos": int(df[c].nunique()),
        "ceros_pct": pct((df[c] == 0).mean()),
        "media_normal": round(float(df.loc[df.label == 0, c].mean())),
        "media_ataque": round(float(df.loc[df.label == 1, c].mean()))}
    for c in ["stcpb", "dtcpb"]}

ttl = {}
for c in ["sttl", "dttl"]:
    tabla = pd.crosstab(df[c], df["label"], normalize="columns").mul(100).round(2)
    tabla.columns = ["normal_pct", "ataque_pct"]
    principales = df[c].value_counts().head(6).index
    ttl[c] = tabla.loc[principales].to_dict("index")
OUT["P7_ttl"] = ttl

fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
for a, c in zip(ax, ["sttl", "dttl"]):
    t = pd.crosstab(df[c], df["label"], normalize="columns").mul(100)
    t = t.loc[df[c].value_counts().head(6).index]
    xx = np.arange(len(t))
    a.bar(xx - 0.2, t[0], 0.4, label="Normal", color="#4472C4")
    a.bar(xx + 0.2, t[1], 0.4, label="Ataque", color="#C00000", hatch="//")
    a.set_xticks(xx); a.set_xticklabels(t.index.astype(int))
    a.set_xlabel(f"Valor de {c}"); a.set_ylabel("% de la clase"); a.set_title(c)
ax[0].legend()
fig.suptitle("Distribución del TTL según la clase (entrenamiento)")
plt.tight_layout(); plt.savefig("eda_fig3_ttl.png", bbox_inches="tight"); plt.close()

with open("eda_resultados.json", "w", encoding="utf-8") as fh:
    json.dump(OUT, fh, indent=2, ensure_ascii=False, default=str)

# Resumen breve en consola
print(f"Filas: {len(df):,} | duplicados: {OUT['P2_duplicados']['total_pct']} %")
print(f"Etiquetas contradictorias: {OUT['P3_etiquetas_contradictorias']['filas_implicadas']} filas")
print(f"Candidatas a log: {len(OUT['P5_asimetria']['candidatas_log'])}")
print(f"Pares redundantes: {len(OUT['P6_redundancia']['pares'])}")
print("Detalle en eda_resultados.json")
