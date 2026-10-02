# Cuadro de mando inteligente para el análisis de tráfico de red y la detección de anomalías mediante aprendizaje automático

Trabajo Final de Máster · Máster Universitario en Ingeniería Informática · Universitat Oberta de Catalunya (UOC)

Prueba de concepto de **Business Intelligence** que integra un modelo de detección de anomalías no supervisado (**Isolation Forest**) dentro de un **cuadro de mando interactivo** (Streamlit + Plotly) orientado al apoyo a la decisión en operaciones TI, sobre el dataset público **UNSW-NB15**.

```
Datos de tráfico de red → Detección de anomalías → KPIs → Dashboard → Apoyo a decisiones
```

## Resultados principales

Modelo entrenado **exclusivamente con tráfico normal** (régimen semisupervisado, sin ver ningún ataque ni etiqueta), evaluado sobre la partición oficial de prueba (82.332 flujos):

| Métrica | Valor |
|---|---:|
| AUC-ROC | **0,826** |
| Recall (umbral P80) | 0,880 |
| Precisión (umbral P80) | 0,735 |
| F1 (umbral P80) | 0,801 |
| Tasa de falsas alarmas | 38,9 % |

Recall por familia de ataque: casi perfecto en Generic (0,994), DoS (0,947), Backdoor (0,938) y Analysis (0,926); limitado en familias sigilosas — Reconnaissance (0,629), Fuzzers (0,611), Shellcode (0,548) — un límite estructural del detector puntual discutido en la memoria.

El umbral de decisión se define como **percentil de la puntuación de anomalía del tráfico normal** (P80–P99) y se expone como control interactivo del dashboard: el compromiso recall/falsas alarmas es una decisión operativa visible, no un hiperparámetro oculto.

## Estructura del repositorio

```
.
├── pipeline.py        # Pipeline completo reproducible: EDA, limpieza, modelo,
│                      # evaluación, KPIs y generación de todas las figuras de la memoria
├── app.py             # Cuadro de mando interactivo (Streamlit + Plotly)
├── resultados.json    # Salida del pipeline: todas las métricas y KPIs calculados
├── requirements.txt   # Dependencias
├── data/              # (no incluida) Dataset UNSW-NB15 — ver instrucciones abajo
└── docs/              # Memoria del TFM y presentación de defensa
```

## Requisitos

- Python 3.12+
- Probado con: pandas 3.0, scikit-learn 1.8, matplotlib, streamlit, plotly

```bash
pip install -r requirements.txt
```

## Datos (no incluidos en el repositorio)

El dataset **UNSW-NB15 no se redistribuye** en este repositorio. Descarga la partición oficial desde la página de sus autores (UNSW Canberra):

- https://research.unsw.edu.au/projects/unsw-nb15-dataset

Coloca los dos ficheros en la carpeta `data/`:

```
data/UNSW_NB15_training-set.csv   (175.341 registros)
data/UNSW_NB15_testing-set.csv    (82.332 registros)
```

## Reproducción de los resultados

Todos los resultados de la memoria son reproducibles (semilla global `random_state = 42`):

```bash
# 1. Pipeline completo: métricas, KPIs y figuras (fig1–fig6, resultados.json)
python pipeline.py

# 2. Cuadro de mando interactivo (http://localhost:8501)
streamlit run app.py
```

La primera ejecución del dashboard tarda 1–2 minutos (entrena el modelo y puntúa el conjunto de prueba); las siguientes son inmediatas gracias a la caché de Streamlit.

Para verificar la reproducción, compara tu `resultados.json` con el incluido en el repositorio: los valores deben coincidir con las mismas versiones de librerías.

## El cuadro de mando

Tres niveles de lectura:

1. **Cabecera de KPIs** — volumen de tráfico, conexiones, anomalías detectadas, % anómalo, tasa de falsas alarmas y recall.
2. **Vistas analíticas** — evolución de anomalías por ventanas secuenciales, distribución por protocolo y servicio, comparador de variables entre tráfico normal y anómalo, y panel del modelo con curva ROC.
3. **Detalle accionable** — tabla de flujos filtrable y ordenada por puntuación de anomalía para priorizar el triaje.

Un control deslizante permite ajustar el umbral de decisión (P50–P99) y observar en tiempo real su efecto sobre alertas, precisión y tasa de falsas alarmas.

## Metodología (resumen)

1. **EDA**: 45 columnas, 0 nulos; 38,6 % de duplicados exactos en el entrenamiento; 53,7 % de flujos con servicio no identificado; asimetrías de hasta 45,3 en variables de volumen.
2. **Preparación**: deduplicación del entrenamiento; recodificación `service="-"` → `unknown`; reducción por correlación (|r| > 0,95); transformación log(1+x) en 25 variables sesgadas; *frequency encoding* de `proto`/`service`/`state`; escalado robusto. **36 variables finales.**
3. **Modelo**: Isolation Forest (300 árboles, `max_samples=512`) ajustado solo sobre los 51.890 flujos normales únicos.
4. **Evaluación**: partición oficial de prueba, nunca vista; barridos de `contamination` y de umbral por percentiles; desglose por las 9 familias de ataque.

## Alcance y limitaciones

Prueba de concepto académica de análisis **por lotes** con exploración interactiva. No incluye captura de tráfico privado, ingesta en tiempo real ni infraestructura productiva. La evolución hacia ingesta NetFlow/IPFIX con reentrenamiento periódico del perfil normal se documenta como línea futura en la memoria.

## Referencias principales

- Moustafa, N., & Slay, J. (2015). *UNSW-NB15: A comprehensive data set for network intrusion detection systems*. MilCIS 2015. https://doi.org/10.1109/MilCIS.2015.7348942
- Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). *Isolation Forest*. ICDM 2008. https://doi.org/10.1109/ICDM.2008.17
- Axelsson, S. (2000). *The base-rate fallacy and the difficulty of intrusion detection*. ACM TISSEC, 3(3). https://doi.org/10.1145/357830.357849

La bibliografía completa (APA 7) está en la memoria del TFM.

## Licencia y atribución

Código bajo licencia MIT. El dataset UNSW-NB15 es propiedad de sus autores (UNSW Canberra) y su uso académico requiere citar sus publicaciones; consulta las condiciones en su página oficial.

## Autor

**[Nombre y apellidos]** · KhouaniSoft
TFM dirigido por **[Nombre del tutor/a]** · UOC · 2026
