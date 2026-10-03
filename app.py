# -*- coding: utf-8 -*-
"""Cuadro de mando inteligente para el análisis de tráfico de red (TFM).

Ejecución:  streamlit run app.py
Requiere:   resultados del pipeline (test con columnas 'pred' y 'score') o lo
            recalcula desde los CSV originales si no existe el fichero cacheado.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import roc_curve, roc_auc_score
import preprocesado as pp

st.set_page_config(page_title="Cuadro de mando · Tráfico de red", layout="wide")


@st.cache_resource(show_spinner="Entrenando el modelo y puntuando el tráfico…")
def cargar_y_puntuar():
    """Mismo protocolo que pipeline.py: preprocesado y modelo ajustados solo con
    tráfico normal de entrenamiento; umbral de referencia fijado en validación."""
    d = pp.preparar_todo()
    sc_val_normal = d["va"].loc[d["va"].label == 0, "score"].values
    return d["test"], sc_val_normal


test, score_norm = cargar_y_puntuar()
test = test.copy()
PCT_REF = int(round((1 - pp.OBJETIVO_FAR) * 100))   # percentil de referencia (P90)


# ---------------- Utilidades de formato (español) ----------------
def es_num(x, dec=0):
    """Número con formato español: punto para miles, coma para decimales."""
    return f"{x:,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def es_pct(x, dec=1):
    return es_num(x * 100, dec) + " %"


def estilo(fig, alto):
    """Separadores españoles, números completos y leyenda visible en todos los gráficos."""
    return fig.update_layout(height=alto, margin=dict(t=30, b=10), separators=",.",
                             yaxis_tickformat=",~r",
                             legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))


# ---------------- Barra lateral: umbral y filtros ----------------
st.sidebar.title("Configuración")
pct = st.sidebar.slider("Umbral de decisión (percentil de la puntuación en tráfico normal de validación)",
                        50, 99, PCT_REF,
                        help="Percentiles altos = menos alertas y menos falsas alarmas, "
                             "pero menor cobertura de ataques.")
umbral = float(np.percentile(score_norm, pct))
st.sidebar.caption(f"Umbral de referencia: P{PCT_REF}, fijado de antemano en validación para un objetivo "
                   f"de falsas alarmas del {int(pp.OBJETIVO_FAR*100)} %. Los demás valores del deslizador "
                   f"son una exploración del compromiso entre cobertura y falsas alarmas y no se usan "
                   f"para seleccionar el modelo.")
test["pred"] = (test["score"] >= umbral).astype(int)

st.sidebar.markdown("### Filtros")
protos = st.sidebar.multiselect("Protocolo", sorted(test["proto"].unique()),
                                placeholder="Todos los protocolos")
servs = st.sidebar.multiselect("Servicio", sorted(test["service"].unique()),
                               placeholder="Todos los servicios")
solo_anom = st.sidebar.checkbox("Solo tráfico marcado como anómalo")

vista = test.copy()
if protos:
    vista = vista[vista["proto"].isin(protos)]
if servs:
    vista = vista[vista["service"].isin(servs)]
if solo_anom:
    vista = vista[vista["pred"] == 1]

# ---------------- Cabecera de KPI ----------------
st.title("Cuadro de mando inteligente · Análisis de tráfico de red")
st.caption("Detección de anomalías con Isolation Forest sobre UNSW-NB15 · Prueba de concepto de BI")

tn = int(((test.label == 0) & (test.pred == 0)).sum())
fp = int(((test.label == 0) & (test.pred == 1)).sum())
fn = int(((test.label == 1) & (test.pred == 0)).sum())
tp = int(((test.label == 1) & (test.pred == 1)).sum())
far = fp / (fp + tn) if fp + tn else 0.0
recall = tp / (tp + fn) if tp + fn else 0.0
gb = (test.sbytes + test.dbytes).sum() / 1e9

c = st.columns(6)
c[0].metric("Volumen total", es_num(gb, 2) + " GB")
c[1].metric("Flujos analizados", es_num(len(test)))
c[2].metric("Flujos anómalos", es_num(int(test.pred.sum())))
c[3].metric("Flujos anómalos (%)", es_pct(test.pred.mean()))
c[4].metric("Tasa de falsas alarmas", es_pct(far))
c[5].metric("Cobertura (recall)", es_pct(recall))

# ---------------- Anomalías por bloques de flujos ----------------
st.subheader("Anomalías por bloques de flujos")
ventana = 2000
st.caption(f"La partición oficial de UNSW-NB15 no incluye marcas de tiempo: cada bloque agrupa "
           f"{es_num(ventana)} flujos consecutivos según su orden en el fichero, no según el tiempo.")
serie = (test.assign(bloque=np.arange(len(test)) // ventana)
             .groupby("bloque").agg(anomalias=("pred", "sum"),
                                    flujos=("pred", "size")).reset_index())
fig = go.Figure()
fig.add_bar(x=serie.bloque, y=serie.flujos, name="Flujos del bloque", marker_color="#B7C7E8")
fig.add_scatter(x=serie.bloque, y=serie.anomalias, name="Anomalías detectadas",
                mode="lines", line=dict(color="#C00000", width=2))
fig.update_layout(xaxis_title=f"Bloque de {es_num(ventana)} flujos (orden del fichero)",
                  yaxis_title="Flujos")
st.plotly_chart(estilo(fig, 300), width="stretch")

# ---------------- Distribuciones ----------------
col1, col2 = st.columns(2)
anom = vista[vista.pred == 1]
with col1:
    st.subheader("Anomalías por protocolo")
    d = anom.proto.value_counts().head(8).reset_index()
    fig = px.bar(d, x="proto", y="count", color_discrete_sequence=["#C00000"],
                 labels={"proto": "Protocolo", "count": "Flujos anómalos"})
    st.plotly_chart(estilo(fig, 300), width="stretch")
with col2:
    st.subheader("Anomalías por servicio")
    d = anom.service.value_counts().head(8).reset_index()
    fig = px.bar(d, x="service", y="count", color_discrete_sequence=["#C00000"],
                 labels={"service": "Servicio", "count": "Flujos anómalos"})
    st.plotly_chart(estilo(fig, 300), width="stretch")

# ---------------- Comparación normal vs anómalo ----------------
st.subheader("Comparación entre tráfico clasificado como normal y como anómalo")
st.caption("Clasificación según la predicción del modelo con el umbral seleccionado.")
VARIABLES = {"dur": "Duración del flujo (dur)", "spkts": "Paquetes origen → destino (spkts)",
             "dpkts": "Paquetes destino → origen (dpkts)", "rate": "Tasa de paquetes (rate)",
             "smean": "Tamaño medio de paquete origen (smean)",
             "dmean": "Tamaño medio de paquete destino (dmean)"}
var = st.selectbox("Variable a comparar", list(VARIABLES), format_func=VARIABLES.get)
fig = go.Figure()
for etiqueta, color, nombre in [(0, "#4472C4", "Clasificado como normal"),
                                (1, "#C00000", "Clasificado como anómalo")]:
    fig.add_histogram(x=np.log1p(test.loc[test.pred == etiqueta, var]), name=nombre,
                      opacity=0.6, marker_color=color, histnorm="probability density")
fig.update_layout(barmode="overlay", xaxis_title=f"log(1 + {var})", yaxis_title="Densidad")
st.plotly_chart(estilo(fig, 320), width="stretch")

# ---------------- Rendimiento del modelo ----------------
st.subheader("Rendimiento del modelo (conjunto de prueba, contra las etiquetas reales)")
fpr, tpr, _ = roc_curve(test.label, test.score)
auc = roc_auc_score(test.label, test.score)
col1, col2 = st.columns([1, 1])
with col1:
    fig = go.Figure()
    fig.add_scatter(x=fpr, y=tpr, name=f"ROC (AUC = {es_num(auc, 3)})",
                    line=dict(color="#C00000"))
    fig.add_scatter(x=[0, 1], y=[0, 1], name="Azar",
                    line=dict(dash="dash", color="gray"))
    fig.add_scatter(x=[far], y=[recall], mode="markers", name=f"Umbral (P{pct})",
                    marker=dict(size=10, color="black"))
    fig.update_layout(xaxis_title="Tasa de falsas alarmas", yaxis_title="Cobertura (recall)")
    st.plotly_chart(estilo(fig, 340), width="stretch")
with col2:
    st.dataframe(pd.DataFrame({
        "Métrica": ["Precisión", "Cobertura (recall)", "F1", "Tasa de falsas alarmas",
                    "Verdaderos positivos", "Falsos positivos", "Falsos negativos",
                    "Verdaderos negativos"],
        "Valor": [es_num(tp / (tp + fp), 3) if tp + fp else "—", es_num(recall, 3),
                  es_num(2 * tp / (2 * tp + fp + fn), 3), es_num(far, 3),
                  es_num(tp), es_num(fp), es_num(fn), es_num(tn)]}),
        width="stretch", hide_index=True)

# ---------------- Detalle ----------------
st.subheader("Detalle de flujos (ordenado por puntuación de anomalía)")
cols = ["proto", "service", "state", "dur", "spkts", "dpkts", "rate", "score", "pred", "attack_cat"]
tabla = vista.sort_values("score", ascending=False)[cols].head(500).copy()
tabla["pred"] = tabla["pred"].map({1: "Anómalo", 0: "Normal"})
tabla = tabla.rename(columns={
    "proto": "Protocolo", "service": "Servicio", "state": "Estado", "dur": "Duración (s)",
    "spkts": "Paquetes orig.", "dpkts": "Paquetes dest.", "rate": "Tasa",
    "score": "Puntuación", "pred": "Predicción", "attack_cat": "Familia real"})
st.dataframe(tabla, width="stretch", height=350, column_config={
    col: st.column_config.NumberColumn(col, format="localized")
    for col in ["Duración (s)", "Paquetes orig.", "Paquetes dest.", "Tasa", "Puntuación"]})
st.caption("Prueba de concepto académica. Las alertas requieren verificación humana: "
           f"con el umbral actual, la tasa de falsas alarmas es del {es_pct(far)}.")
