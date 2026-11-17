# -*- coding: utf-8 -*-
"""Preprocesado, modelo y KPI comunes para pipeline.py, analisis.py y app.py.

Protocolo experimental:
  1. Partición oficial de entrenamiento -> deduplicación -> eliminación de flujos con
     etiquetas contradictorias -> división estratificada en entrenamiento (80 %) y
     validación (20 %). La partición de prueba no se modifica.
  2. Preprocesado e Isolation Forest ajustados EXCLUSIVAMENTE con el tráfico normal
     del subconjunto de entrenamiento.
  3. Umbral fijado en validación con un criterio operativo (objetivo de falsas alarmas),
     sin usar etiquetas de ataque.
  4. Partición oficial de prueba evaluada una sola vez con el umbral congelado.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

SEMILLA = 42
CATEGORICAS = ["proto", "service", "state"]
UMBRAL_CORRELACION = 0.95
UMBRAL_SESGO = 3
FRACCION_VALIDACION = 0.20
OBJETIVO_FAR = 0.10          # tasa máxima de falsas alarmas admitida en validación

# Variables excluidas de forma explícita, con su motivo (ver análisis exploratorio, eda.py)
EXCLUIDAS = {
    "stcpb": "número de secuencia TCP inicial: aleatorio en TCP y 0 fuera de TCP "
             "(sin información de comportamiento; redundante con proto)",
    "dtcpb": "número de secuencia TCP inicial: aleatorio en TCP y 0 fuera de TCP "
             "(sin información de comportamiento; redundante con proto)",
    "is_ftp_login": "indicador binario con valores imposibles (2, 4); idéntico a ct_ftp_cmd (r = 1)",
}


def cargar(ruta="data"):
    train = pd.read_csv(f"{ruta}/UNSW_NB15_training-set.csv")
    test = pd.read_csv(f"{ruta}/UNSW_NB15_testing-set.csv")
    return train, test


def limpiar(df):
    """Elimina el identificador y recodifica el servicio no identificado."""
    df = df.drop(columns=["id"]).copy()
    df["service"] = df["service"].replace("-", "unknown")
    return df


def quitar_contradictorias(df):
    """Elimina los flujos con atributos idénticos y etiqueta binaria distinta: no se
    puede saber cuál es la correcta y, en el tráfico normal, enseñarían al modelo
    patrones de ataque como normales. Devuelve el conjunto limpio y el nº de filas quitadas."""
    atributos = [c for c in df.columns if c not in ("label", "attack_cat")]
    n_etiquetas = df.groupby(atributos, dropna=False)["label"].transform("nunique")
    return df[n_etiquetas == 1].reset_index(drop=True), int((n_etiquetas > 1).sum())


def dividir(train_limpio):
    """Deduplica ANTES de dividir (evita copias exactas a ambos lados), elimina las
    etiquetas contradictorias y separa entrenamiento/validación estratificando por
    familia de ataque."""
    dedup = train_limpio.drop_duplicates().reset_index(drop=True)
    dedup, n_contradictorias = quitar_contradictorias(dedup)
    tr, va = train_test_split(dedup, test_size=FRACCION_VALIDACION,
                              stratify=dedup["attack_cat"], random_state=SEMILLA)
    return dedup, tr.reset_index(drop=True), va.reset_index(drop=True), n_contradictorias


class Preprocesado:
    """Transformaciones aprendidas solo con tráfico normal de entrenamiento."""

    def __init__(self, excluir_extra=()):
        # excluir_extra: solo para análisis de sensibilidad (p. ej., sin sttl/dttl)
        self.excluidas = list(EXCLUIDAS) + list(excluir_extra)

    def ajustar(self, normales):
        num = [c for c in normales.select_dtypes("number").columns
               if c != "label" and c not in self.excluidas]
        # Variables constantes en el tráfico normal: sin información para el perfil
        self.constantes = [c for c in num if normales[c].nunique() <= 1]
        num = [c for c in num if c not in self.constantes]
        # Redundancia: de cada par con |r| > 0,95 se elimina la segunda variable, aunque la
        # primera también se elimine por otro par (p. ej., dbytes, por dpkts)
        corr = normales[num].corr().abs()
        sup = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool))
        self.correladas = [c for c in num if (sup[c] > UMBRAL_CORRELACION).any()]
        self.numericas = [c for c in num if c not in self.correladas]
        # Codificación por frecuencia: categorías no vistas en normal -> 0 (máxima rareza)
        self.frecuencias = {c: normales[c].value_counts(normalize=True) for c in CATEGORICAS}
        # Transformación logarítmica de las variables muy asimétricas
        # (solo variables con más de dos valores: en indicadores binarios no tiene sentido)
        self.sesgadas = [c for c in self.numericas
                         if normales[c].nunique() > 2 and abs(normales[c].skew()) > UMBRAL_SESGO]
        self.variables = self.numericas + [c + "_freq" for c in CATEGORICAS]
        self.escalador = RobustScaler().fit(self._matriz(normales))
        return self

    def _matriz(self, df):
        X = df[self.numericas].astype(float).copy()
        for c in self.sesgadas:
            X[c] = np.log1p(X[c].clip(lower=0))
        for c in CATEGORICAS:
            X[c + "_freq"] = df[c].map(self.frecuencias[c]).fillna(0.0).astype(float)
        return X[self.variables]

    def transformar(self, df):
        return self.escalador.transform(self._matriz(df))


def entrenar(X_normales):
    return IsolationForest(n_estimators=300, max_samples=512,
                           random_state=SEMILLA, n_jobs=-1).fit(X_normales)


def puntuar(modelo, X):
    """Puntuación de anomalía: mayor valor = más anómalo."""
    return -modelo.score_samples(X)


def umbral_por_far(puntuaciones_normales_validacion, objetivo=OBJETIVO_FAR):
    """Umbral = percentil (1 - objetivo) de la puntuación del tráfico normal de validación:
    alrededor del `objetivo` de ese tráfico queda por encima (falsas alarmas).
    Solo usa tráfico normal."""
    return float(np.quantile(puntuaciones_normales_validacion, 1 - objetivo))


def kpis(df, umbral):
    """Indicadores del apartado 2.6 de la memoria (I1-I6) sobre flujos ya puntuados.

    I1-I4 no usan etiquetas (indicadores operativos); I5-I6 sí (calidad del detector).
    Si no hay flujos (o no hay flujos de una clase), el indicador vale NaN."""
    alerta = (df["score"] >= umbral).to_numpy()
    normal = (df["label"] == 0).to_numpy()
    n, a = len(df), int(alerta.sum())
    return {
        "flujos": n,                                                           # I1
        "volumen_gb": float((df["sbytes"] + df["dbytes"]).sum()) / 1e9,        # I2
        "anomalos": a,                                                         # I3
        "pct_anomalos": a / n * 100 if n else np.nan,                          # I4
        "tasa_falsas_alarmas_pct": alerta[normal].mean() * 100 if normal.any() else np.nan,  # I5
        "cobertura_pct": alerta[~normal].mean() * 100 if (~normal).any() else np.nan,        # I6
    }


def preparar_todo(ruta="data"):
    """Ejecuta el protocolo completo y devuelve todo lo necesario para evaluar."""
    train_raw, test_raw = cargar(ruta)
    train_l, test = limpiar(train_raw), limpiar(test_raw)
    dedup, tr, va, n_contradictorias = dividir(train_l)
    normales_tr = tr[tr.label == 0]
    prep = Preprocesado().ajustar(normales_tr)
    modelo = entrenar(prep.transformar(normales_tr))
    # Validación: puntuar y fijar el umbral
    va = va.assign(score=puntuar(modelo, prep.transformar(va)))
    umbral = umbral_por_far(va.loc[va.label == 0, "score"])
    # Modelo y umbral congelados: solo ahora se puntúa la partición de prueba
    test = test.assign(score=puntuar(modelo, prep.transformar(test)))
    return dict(train_raw=train_raw, test_raw=test_raw, dedup=dedup, tr=tr, va=va, test=test,
                prep=prep, modelo=modelo, umbral=umbral, n_contradictorias=n_contradictorias)
