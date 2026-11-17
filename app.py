# -*- coding: utf-8 -*-
"""Cuadro de mando inteligente para el análisis de tráfico de red (TFM).

Ejecución:  streamlit run app.py

Al arrancar ejecuta el mismo protocolo que pipeline.py (preprocesado.preparar_todo):
preprocesado y modelo ajustados con tráfico normal de entrenamiento, umbral de referencia
fijado en validación y partición de prueba puntuada. El resultado queda en caché mientras
la aplicación está abierta. Los indicadores siguen el apartado 2.6 de la memoria.

Pestañas:
  Supervisión             KPI y distribución de las alertas, en una sola pantalla.
  Exploración             comparación entre tráfico normal y anómalo, y detalle de flujos.
  Rendimiento del modelo  curva ROC y métricas contra las etiquetas reales (global).
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import roc_auc_score, roc_curve
import preprocesado as pp

st.set_page_config(page_title="Cuadro de mando · Tráfico de red", layout="wide")
AZUL, ROJO, GRIS = "#4472C4", "#C00000", "#7F7F7F"
PCT_REF = int(round((1 - pp.OBJETIVO_FAR) * 100))      # percentil de referencia (P90)
OBJETIVO_PCT = pp.OBJETIVO_FAR * 100                   # objetivo de falsas alarmas (10 %)


@st.cache_resource(show_spinner="Entrenando el modelo y puntuando el tráfico…")
def cargar_y_puntuar():
    """Mismo protocolo que pipeline.py. La curva ROC no depende del umbral ni de los
    filtros, así que se calcula una sola vez."""
    d = pp.preparar_todo()
    test = d["test"].copy()
    # Servicio sin identificar con el mismo nombre que en la memoria (apartado 2.4)
    test["service"] = test["service"].replace("unknown", "no identificado")
    test.insert(0, "id", test.index + 1)               # identificador original del flujo
    sc_val_normal = d["va"].loc[d["va"].label == 0, "score"].to_numpy()
    fpr, tpr, _ = roc_curve(test["label"], test["score"])
    return test, sc_val_normal, fpr, tpr, roc_auc_score(test["label"], test["score"])


test, score_norm, roc_fpr, roc_tpr, auc = cargar_y_puntuar()


# ---------------- Utilidades de formato (español) ----------------
def es_num(x, dec=0):
    """Número con formato español: punto para miles, coma para decimales."""
    if pd.isna(x):
        return "—"
    return f"{x:,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def es_pct(x, dec=1):
    """x ya expresado en porcentaje."""
    return "—" if pd.isna(x) else es_num(x, dec) + " %"


def es_sig(x, cifras=3):
    """Número con formato español y 3 cifras significativas, sin ceros decimales sobrantes."""
    if pd.isna(x):
        return "—"
    if x == 0:
        return "0"
    nd = cifras - 1 - int(np.floor(np.log10(abs(x))))
    s = es_num(round(x, nd), max(nd, 0))
    return s.rstrip("0").rstrip(",") if "," in s else s


# Definición de los indicadores (apartado 2.6 de la memoria), visible en un desplegable
DEFINICIONES = pd.DataFrame({
    "ID": ["I1", "I2", "I3", "I4", "I5", "I6"],
    "Indicador": ["Flujos analizados", "Volumen analizado (GB)", "Flujos anómalos",
                  "Flujos anómalos (%) · KPI", "Tasa de falsas alarmas · KPI", "Cobertura"],
    "Cálculo": ["Número de flujos (N)", "Σ (sbytes + dbytes) / 10⁹",
                "Flujos con puntuación ≥ umbral (A)", "100 · A / N", "FP / (FP + VN)", "VP / (VP + FN)"],
    "Ámbito": ["Filtros", "Filtros", "Filtros", "Filtros",
               "Global, con etiquetas", "Global, con etiquetas"],
    "Objetivo o referencia": ["—", "—", "—",
                              "Si los ataques son poco frecuentes, se aproxima a la tasa de falsas alarmas",
                              f"{OBJETIVO_PCT:.0f} % (fijado en validación)",
                              "Contrapeso de la tasa de falsas alarmas"]})


def estilo(fig, alto):
    """Separadores españoles, números completos, márgenes automáticos (etiquetas sin cortar)
    y leyenda visible en todos los gráficos."""
    fig.update_xaxes(automargin=True, tickformat=",~r")
    fig.update_yaxes(automargin=True, tickformat=",~r")
    return fig.update_layout(height=alto, margin=dict(t=30, b=10), separators=",.",
                             legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))


# ---------------- Barra lateral: umbral y filtros ----------------
st.sidebar.title("Configuración")
pct = st.sidebar.slider("Umbral de decisión (percentil de la puntuación en tráfico normal de validación)",
                        50, 99, PCT_REF,
                        help="Percentiles altos = menos alertas y menos falsas alarmas, "
                             "pero menor cobertura de ataques.")
umbral = float(np.percentile(score_norm, pct))
umbral_ref = float(np.percentile(score_norm, PCT_REF))
st.sidebar.caption(f"Umbral actual: {es_num(umbral, 4)} (P{pct}). Umbral de referencia: P{PCT_REF} "
                   f"({es_num(umbral_ref, 4)}), fijado de antemano en validación para un objetivo de "
                   f"falsas alarmas del {es_num(OBJETIVO_PCT)} %. Los demás valores del deslizador "
                   f"exploran el compromiso entre cobertura y falsas alarmas y no se usan para "
                   f"seleccionar el modelo.")

st.sidebar.markdown("### Filtros")
protos = st.sidebar.multiselect("Protocolo", sorted(test["proto"].unique()),
                                placeholder="Todos los protocolos")
servs = st.sidebar.multiselect("Servicio", sorted(test["service"].unique()),
                               placeholder="Todos los servicios")
st.sidebar.caption("Los filtros se aplican a los indicadores operativos, a la supervisión y a la "
                   "exploración. El rendimiento del modelo es siempre global.")

# Datos: 'datos' es toda la prueba (global); 'base', el tráfico seleccionado con los filtros
datos = test.assign(pred=(test["score"] >= umbral).astype(int))
base = datos
if protos:
    base = base[base["proto"].isin(protos)]
if servs:
    base = base[base["service"].isin(servs)]
k_op = pp.kpis(base, umbral)       # I1-I4: según los filtros
k_gl = pp.kpis(datos, umbral)      # I5-I6: globales, con etiquetas

st.title("Cuadro de mando inteligente · Análisis de tráfico de red")
st.caption("Detección de anomalías con Isolation Forest sobre UNSW-NB15 · Prueba de concepto de BI")
tab_sup, tab_exp, tab_rend = st.tabs(["Supervisión", "Exploración", "Rendimiento del modelo"])

# ======================= Supervisión =======================
with tab_sup:
    # Una sola fila de seis indicadores (etiquetas completas); los dos grupos se rotulan encima
    g_op, g_cal = st.columns([4, 2])
    g_op.caption(f"**Indicadores operativos** · tráfico seleccionado con los filtros · umbral P{pct}")
    g_cal.caption(f"**Calidad del detector** · global, con etiquetas reales · objetivo de falsas "
                  f"alarmas: {es_num(OBJETIVO_PCT)} %")
    c = st.columns(6)
    c[0].metric("Flujos analizados", es_num(k_op["flujos"]))
    c[1].metric("Volumen analizado", es_num(k_op["volumen_gb"], 2) + " GB")
    c[2].metric("Flujos anómalos", es_num(k_op["anomalos"]))
    c[3].metric("Flujos anómalos (%)", es_pct(k_op["pct_anomalos"]))
    dif = k_gl["tasa_falsas_alarmas_pct"] - OBJETIVO_PCT
    c[4].metric("Tasa de falsas alarmas", es_pct(k_gl["tasa_falsas_alarmas_pct"]),
                delta=f"{'+' if dif >= 0 else '-'}{es_num(abs(dif), 1)} pp vs. {es_num(OBJETIVO_PCT)} %",
                delta_color="inverse")
    c[5].metric("Cobertura", es_pct(k_gl["cobertura_pct"]))
    with st.expander("Definición de los indicadores"):
        st.dataframe(DEFINICIONES, hide_index=True, width="stretch")

    if base.empty:
        st.info("No hay flujos con esta combinación de filtros.")
    else:
        g1, g2, g3 = st.columns(3)
        with g1:
            st.markdown("**Distribución de la puntuación de anomalía**")
            ini, fin = float(test["score"].min()), float(test["score"].max())
            cubos = dict(start=ini, end=fin, size=(fin - ini) / 60)
            fig = go.Figure()
            fig.add_histogram(x=base.loc[base["pred"] == 0, "score"], xbins=cubos,
                              name="Sin alerta", marker_color=AZUL)
            fig.add_histogram(x=base.loc[base["pred"] == 1, "score"], xbins=cubos,
                              name="Alerta", marker_color=ROJO)
            # Las líneas de umbral se identifican en la leyenda, sin rótulos sobre las barras
            lineas = [(umbral, "dash", "black", f"Umbral (P{pct})")]
            if pct != PCT_REF:
                lineas.append((umbral_ref, "dot", GRIS, f"Referencia (P{PCT_REF})"))
            for x, trazo, color, nombre in lineas:
                fig.add_vline(x=x, line_dash=trazo, line_color=color)
                fig.add_scatter(x=[None], y=[None], mode="lines", name=nombre,
                                line=dict(dash=trazo, color=color))
            fig.update_layout(barmode="stack", bargap=0.05,
                              xaxis_title="Puntuación de anomalía", yaxis_title="Flujos",
                              legend=dict(traceorder="normal", entrywidth=0.5,
                                          entrywidthmode="fraction", font=dict(size=11)))
            st.plotly_chart(estilo(fig, 300), width="stretch")
        anomalos = base[base["pred"] == 1]
        for col, campo, titulo in [(g2, "proto", "Flujos anómalos por protocolo"),
                                   (g3, "service", "Flujos anómalos por servicio")]:
            with col:
                st.markdown(f"**{titulo}**")
                if anomalos.empty:
                    st.caption("No hay flujos anómalos con estos filtros y este umbral.")
                    continue
                cuenta = anomalos[campo].value_counts().head(8).sort_values()
                fig = go.Figure(go.Bar(x=cuenta.values, y=cuenta.index, orientation="h",
                                       marker_color=ROJO))
                fig.update_layout(xaxis_title="Flujos anómalos", showlegend=False)
                st.plotly_chart(estilo(fig, 300), width="stretch")

# ======================= Exploración =======================
with tab_exp:
    st.subheader("Comparación entre tráfico clasificado como normal y como anómalo")
    st.caption("Clasificación según el modelo con el umbral seleccionado. Diagramas de caja en escala "
               "log(1 + x): la caja va del percentil 25 al 75, la línea interior es la mediana y los "
               "bigotes llegan al mínimo y al máximo. Bajo cada grupo se indica la mediana en unidades "
               "originales.")
    VARIABLES = {"dur": "Duración del flujo (dur, s)",
                 "spkts": "Paquetes origen → destino (spkts)",
                 "sbytes": "Bytes origen → destino (sbytes)",
                 "rate": "Tasa de paquetes (rate, paquetes/s)",
                 "smean": "Tamaño medio de paquete origen (smean, bytes)",
                 "dmean": "Tamaño medio de paquete destino (dmean, bytes)"}
    UNIDADES = {"dur": "s", "spkts": "paquetes", "sbytes": "bytes", "rate": "paquetes/s",
                "smean": "bytes", "dmean": "bytes"}
    var = st.selectbox("Variable a comparar", list(VARIABLES), format_func=VARIABLES.get)
    if base.empty:
        st.info("No hay flujos con esta combinación de filtros.")
    else:
        fig = go.Figure()
        for clase, color, nombre in [(0, AZUL, "Clasificado como normal"),
                                     (1, ROJO, "Clasificado como anómalo")]:
            serie = base.loc[base["pred"] == clase, var]
            fig.add_box(y=np.log1p(serie), marker_color=color, boxpoints=False,
                        name=f"{nombre}<br>{es_num(len(serie))} flujos · mediana "
                             f"{es_sig(serie.median())} {UNIDADES[var]}")
        fig.update_layout(yaxis_title=f"log(1 + {var})", showlegend=False)
        st.plotly_chart(estilo(fig, 340), width="stretch")

    st.subheader("Detalle de flujos (ordenados por puntuación de anomalía)")
    tabla = base.sort_values("score", ascending=False).head(500)
    tabla = pd.DataFrame({
        "id": tabla["id"], "Protocolo": tabla["proto"], "Servicio": tabla["service"],
        "Estado": tabla["state"], "Duración (s)": tabla["dur"],
        "Paquetes orig.": tabla["spkts"], "Paquetes dest.": tabla["dpkts"],
        "Tasa (paquetes/s)": tabla["rate"].map(lambda v: float(f"{v:.3g}")),   # 3 cifras significativas
        "Puntuación": tabla["score"].round(3),
        "Predicción": tabla["pred"].map({1: "Anómalo", 0: "Normal"}),
        "Familia real (solo evaluación)": tabla["attack_cat"]})
    st.dataframe(tabla, width="stretch", height=350, hide_index=True, column_config={
        "id": st.column_config.NumberColumn("id", format="%d"),
        **{col: st.column_config.NumberColumn(col, format="localized")
           for col in ["Duración (s)", "Paquetes orig.", "Paquetes dest.", "Tasa (paquetes/s)",
                       "Puntuación"]}})
    st.caption(f"Se muestran como máximo 500 flujos, del más al menos anómalo. Prueba de concepto "
               f"académica: las alertas "
               f"requieren verificación humana; con el umbral actual, la tasa de falsas alarmas "
               f"global es del {es_pct(k_gl['tasa_falsas_alarmas_pct'])}.")

# ======================= Rendimiento del modelo =======================
with tab_rend:
    st.subheader("Rendimiento del modelo")
    st.caption("Partición de prueba completa contra las etiquetas reales: no depende de los filtros.")
    y, p = datos["label"].to_numpy(), datos["pred"].to_numpy()
    tp, fp = int(((y == 1) & (p == 1)).sum()), int(((y == 0) & (p == 1)).sum())
    fn, tn = int(((y == 1) & (p == 0)).sum()), int(((y == 0) & (p == 0)).sum())
    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_scatter(x=roc_fpr, y=roc_tpr, name=f"ROC (AUC = {es_num(auc, 3)})",
                        line=dict(color=ROJO))
        fig.add_scatter(x=[0, 1], y=[0, 1], name="Azar", line=dict(dash="dash", color=GRIS))
        fig.add_scatter(x=[k_gl["tasa_falsas_alarmas_pct"] / 100], y=[k_gl["cobertura_pct"] / 100],
                        mode="markers", name=f"Umbral actual (P{pct})",
                        marker=dict(size=10, color="black", symbol="circle"))
        if pct != PCT_REF:
            k_ref = pp.kpis(datos, umbral_ref)
            fig.add_scatter(x=[k_ref["tasa_falsas_alarmas_pct"] / 100], y=[k_ref["cobertura_pct"] / 100],
                            mode="markers", name=f"Umbral de referencia (P{PCT_REF})",
                            marker=dict(size=10, color=GRIS, symbol="diamond"))
        fig.update_layout(xaxis_title="Tasa de falsas alarmas", yaxis_title="Cobertura (recall)")
        st.plotly_chart(estilo(fig, 340), width="stretch")
    with col2:
        st.dataframe(pd.DataFrame({
            "Métrica": ["Tasa de falsas alarmas", "Cobertura (recall)", "Precisión", "F1",
                        "Verdaderos positivos", "Falsos positivos", "Falsos negativos",
                        "Verdaderos negativos"],
            "Valor": [es_pct(k_gl["tasa_falsas_alarmas_pct"]), es_pct(k_gl["cobertura_pct"]),
                      es_pct(tp / (tp + fp) * 100) if tp + fp else "—",
                      es_num(2 * tp / (2 * tp + fp + fn), 3) if tp + fp + fn else "—",
                      es_num(tp), es_num(fp), es_num(fn), es_num(tn)]}),
            width="stretch", hide_index=True)
