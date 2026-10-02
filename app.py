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
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_curve, roc_auc_score

st.set_page_config(page_title="Cuadro de mando · Tráfico de red", layout="wide")
RS = 42


@st.cache_data(show_spinner="Entrenando el modelo y puntuando el tráfico…")
def cargar_y_puntuar():
    train = pd.read_csv("data/UNSW_NB15_training-set.csv")
    test = pd.read_csv("data/UNSW_NB15_testing-set.csv")
    for df in (train, test):
        df.drop(columns=["id"], inplace=True)
        df["service"] = df["service"].replace("-", "unknown")
    train = train.drop_duplicates().reset_index(drop=True)

    num = [c for c in train.select_dtypes("number").columns if c != "label"]
    corr = train[num].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool))
    num = [c for c in num if not any(upper[c] > 0.95)]
    for c in ["proto", "service", "state"]:
        f = train[c].value_counts(normalize=True)
        train[c + "_freq"] = train[c].map(f)
        test[c + "_freq"] = test[c].map(f).fillna(0.0)
    feats = num + ["proto_freq", "service_freq", "state_freq"]
    skewed = [c for c in num if abs(train[c].skew()) > 3]
    Xtr, Xte = train[feats].copy(), test[feats].copy()
    for c in skewed:
        Xtr[c] = np.log1p(Xtr[c].clip(lower=0))
        Xte[c] = np.log1p(Xte[c].clip(lower=0))
    sc = RobustScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)

    modelo = IsolationForest(n_estimators=300, max_samples=512,
                             random_state=RS, n_jobs=-1).fit(Xtr_s[train.label == 0])
    test["score"] = -modelo.score_samples(Xte_s)
    score_normal_train = -modelo.score_samples(Xtr_s[train.label == 0])
    return test, score_normal_train


test, score_norm = cargar_y_puntuar()

# ---------------- Barra lateral: umbral y filtros ----------------
st.sidebar.title("Configuración")
pct = st.sidebar.slider("Umbral de decisión (percentil del score en tráfico normal)",
                        50, 99, 80,
                        help="Percentiles altos = menos alertas y menos falsas alarmas, "
                             "pero menor cobertura de ataques.")
umbral = float(np.percentile(score_norm, pct))
test["pred"] = (test["score"] >= umbral).astype(int)

st.sidebar.markdown("### Filtros")
protos = st.sidebar.multiselect("Protocolo", sorted(test["proto"].unique()))
servs = st.sidebar.multiselect("Servicio", sorted(test["service"].unique()))
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

c = st.columns(4)
c[0].metric("Volumen total", f"{gb:,.2f} GB")
c[1].metric("Conexiones", f"{len(test):,}")
c[2].metric("Anomalías detectadas", f"{int(test.pred.sum()):,}",
            f"{test.pred.mean()*100:.1f} % del tráfico")
c[3].metric("Tasa de falsas alarmas", f"{far*100:.1f} %",
            f"recall {recall*100:.1f} %", delta_color="inverse")

# ---------------- Evolución temporal (ventanas secuenciales) ----------------
st.subheader("Evolución de las anomalías")
ventana = 2000
serie = (test.assign(bloque=np.arange(len(test)) // ventana)
             .groupby("bloque").agg(anomalias=("pred", "sum"),
                                    conexiones=("pred", "size")).reset_index())
fig = go.Figure()
fig.add_bar(x=serie.bloque, y=serie.conexiones, name="Conexiones", marker_color="#B7C7E8")
fig.add_scatter(x=serie.bloque, y=serie.anomalias, name="Anomalías",
                mode="lines", line=dict(color="#C00000", width=2))
fig.update_layout(height=280, margin=dict(t=10, b=10),
                  xaxis_title=f"Ventana secuencial ({ventana} flujos)", yaxis_title="Flujos")
st.plotly_chart(fig, width="stretch")

# ---------------- Distribuciones ----------------
col1, col2 = st.columns(2)
anom = vista[vista.pred == 1]
with col1:
    st.subheader("Anomalías por protocolo")
    d = anom.proto.value_counts().head(8).reset_index()
    st.plotly_chart(px.bar(d, x="proto", y="count", color_discrete_sequence=["#C00000"])
                    .update_layout(height=300, margin=dict(t=10)), width="stretch")
with col2:
    st.subheader("Anomalías por servicio")
    d = anom.service.value_counts().head(8).reset_index()
    st.plotly_chart(px.bar(d, x="service", y="count", color_discrete_sequence=["#C00000"])
                    .update_layout(height=300, margin=dict(t=10)), width="stretch")

# ---------------- Comparación normal vs anómalo ----------------
st.subheader("Comparación tráfico normal vs. anómalo")
var = st.selectbox("Variable a comparar", ["dur", "spkts", "dpkts", "rate", "smean", "dmean"])
fig = go.Figure()
for etiqueta, color, nombre in [(0, "#4472C4", "Normal (pred.)"), (1, "#C00000", "Anómalo (pred.)")]:
    fig.add_histogram(x=np.log1p(test.loc[test.pred == etiqueta, var]), name=nombre,
                      opacity=0.6, marker_color=color, histnorm="probability density")
fig.update_layout(barmode="overlay", height=300, margin=dict(t=10),
                  xaxis_title=f"log(1+{var})", yaxis_title="Densidad")
st.plotly_chart(fig, width="stretch")

# ---------------- Rendimiento del modelo ----------------
st.subheader("Rendimiento del modelo (contra etiquetas reales del dataset)")
fpr, tpr, _ = roc_curve(test.label, test.score)
auc = roc_auc_score(test.label, test.score)
col1, col2 = st.columns([1, 1])
with col1:
    fig = go.Figure()
    fig.add_scatter(x=fpr, y=tpr, name=f"AUC = {auc:.3f}", line=dict(color="#C00000"))
    fig.add_scatter(x=[0, 1], y=[0, 1], name="Azar", line=dict(dash="dash", color="gray"))
    fig.add_scatter(x=[far], y=[recall], mode="markers", name=f"Umbral P{pct}",
                    marker=dict(size=10, color="black"))
    fig.update_layout(height=320, margin=dict(t=10),
                      xaxis_title="Tasa de falsos positivos", yaxis_title="Recall")
    st.plotly_chart(fig, width="stretch")
with col2:
    st.dataframe(pd.DataFrame({
        "Métrica": ["Precisión", "Recall", "F1", "FAR", "TP", "FP", "FN", "TN"],
        "Valor": [f"{tp/(tp+fp):.3f}" if tp + fp else "—", f"{recall:.3f}",
                  f"{2*tp/(2*tp+fp+fn):.3f}", f"{far:.3f}",
                  f"{tp:,}", f"{fp:,}", f"{fn:,}", f"{tn:,}"]}),
        width="stretch", hide_index=True)

# ---------------- Detalle ----------------
st.subheader("Detalle de flujos (ordenado por puntuación de anomalía)")
cols = ["proto", "service", "state", "dur", "spkts", "dpkts", "rate", "score", "pred", "attack_cat"]
st.dataframe(vista.sort_values("score", ascending=False)[cols].head(500),
             width="stretch", height=350)
st.caption("Prueba de concepto académica. Las alertas requieren verificación humana: "
           f"con el umbral actual, la tasa de falsas alarmas es del {far*100:.1f} %.")
