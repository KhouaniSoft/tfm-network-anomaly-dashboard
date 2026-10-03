# -*- coding: utf-8 -*-
"""Preprocesado y modelo comunes para pipeline.py y app.py.

Protocolo experimental:
  1. Partición oficial de entrenamiento -> deduplicación -> división estratificada
     en entrenamiento (80 %) y validación (20 %).
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


def cargar(ruta="data"):
    train = pd.read_csv(f"{ruta}/UNSW_NB15_training-set.csv")
    test = pd.read_csv(f"{ruta}/UNSW_NB15_testing-set.csv")
    return train, test


def limpiar(df):
    """Elimina el identificador y recodifica el servicio no identificado."""
    df = df.drop(columns=["id"]).copy()
    df["service"] = df["service"].replace("-", "unknown")
    return df


def dividir(train_limpio):
    """Deduplica ANTES de dividir (evita copias exactas a ambos lados) y separa
    entrenamiento/validación estratificando por familia de ataque."""
    dedup = train_limpio.drop_duplicates().reset_index(drop=True)
    tr, va = train_test_split(dedup, test_size=FRACCION_VALIDACION,
                              stratify=dedup["attack_cat"], random_state=SEMILLA)
    return dedup, tr.reset_index(drop=True), va.reset_index(drop=True)


class Preprocesado:
    """Transformaciones aprendidas solo con tráfico normal de entrenamiento."""

    def ajustar(self, normales):
        num = [c for c in normales.select_dtypes("number").columns if c != "label"]
        # Variables constantes en el tráfico normal: sin información para el perfil
        self.constantes = [c for c in num if normales[c].nunique() <= 1]
        num = [c for c in num if c not in self.constantes]
        # Redundancia: |r| > 0,95 con una variable ya retenida
        corr = normales[num].corr().abs()
        sup = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool))
        self.correladas = [c for c in num if (sup[c] > UMBRAL_CORRELACION).any()]
        self.numericas = [c for c in num if c not in self.correladas]
        # Codificación por frecuencia: categorías no vistas en normal -> 0 (máxima rareza)
        self.frecuencias = {c: normales[c].value_counts(normalize=True) for c in CATEGORICAS}
        # Transformación logarítmica de las variables muy asimétricas
        self.sesgadas = [c for c in self.numericas if abs(normales[c].skew()) > UMBRAL_SESGO]
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
    """Umbral tal que, en validación, como máximo el `objetivo` del tráfico normal
    queda por encima (falsas alarmas). Solo usa tráfico normal."""
    return float(np.quantile(puntuaciones_normales_validacion, 1 - objetivo))


def preparar_todo(ruta="data"):
    """Ejecuta el protocolo completo y devuelve todo lo necesario para evaluar."""
    train_raw, test_raw = cargar(ruta)
    train_l, test = limpiar(train_raw), limpiar(test_raw)
    dedup, tr, va = dividir(train_l)
    normales_tr = tr[tr.label == 0]
    prep = Preprocesado().ajustar(normales_tr)
    modelo = entrenar(prep.transformar(normales_tr))
    # Validación: puntuar y fijar el umbral
    va = va.assign(score=puntuar(modelo, prep.transformar(va)))
    umbral = umbral_por_far(va.loc[va.label == 0, "score"])
    # Modelo y umbral congelados: solo ahora se puntúa la partición de prueba
    test = test.assign(score=puntuar(modelo, prep.transformar(test)))
    return dict(train_raw=train_raw, dedup=dedup, tr=tr, va=va, test=test,
                prep=prep, modelo=modelo, umbral=umbral)
