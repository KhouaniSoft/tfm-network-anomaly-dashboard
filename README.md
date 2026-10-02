# Cuadro de mando inteligente para el análisis de tráfico de red y la detección de anomalías mediante aprendizaje automático

Trabajo Final de Máster · Máster Universitario en Ingeniería Informática · Universitat Oberta de Catalunya (UOC) · Curso 2026-2027

Prueba de concepto de **Business Intelligence** que integra un modelo de detección de anomalías (**Isolation Forest**, en régimen semisupervisado) dentro de un **cuadro de mando interactivo** (Streamlit + Plotly) orientado al apoyo a la decisión en operaciones TI, sobre el conjunto de datos público **UNSW-NB15**.

```
Datos de tráfico de red → Detección de anomalías → KPI → Cuadro de mando → Apoyo a la decisión
```

## Estado del proyecto

En desarrollo. Los resultados se publicarán en este documento a medida que se obtengan y validen en cada entrega parcial.

## Estructura del repositorio

```
.
├── pipeline.py        # Proceso completo: análisis exploratorio, limpieza, modelo,
│                      # evaluación, KPI y generación de figuras
├── app.py             # Cuadro de mando interactivo (Streamlit + Plotly)
├── requirements.txt   # Dependencias con versiones fijadas
└── data/              # (no incluida) Conjunto de datos UNSW-NB15 — ver instrucciones
```

Al ejecutar `pipeline.py` se generan `resultados.json` (métricas y KPI) y las figuras del análisis.

## Requisitos

- Python 3.12
- Dependencias con versiones fijadas para garantizar la reproducibilidad:

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

## Datos (no incluidos en el repositorio)

El conjunto de datos **UNSW-NB15 no se redistribuye** en este repositorio. Descarga la partición oficial desde la página de sus autores (UNSW Canberra):

- https://research.unsw.edu.au/projects/unsw-nb15-dataset

Coloca los dos ficheros en la carpeta `data/`:

```
data/UNSW_NB15_training-set.csv   (175.341 registros)
data/UNSW_NB15_testing-set.csv    (82.332 registros)
```

Para comprobar que dispones exactamente de la misma versión empleada en este trabajo, verifica su huella SHA-256:

| Fichero | SHA-256 |
|---|---|
| `UNSW_NB15_training-set.csv` | `BEC7DD5EC88DC2A0CCC7A07879D338395ED7421750F675FD0339E07DFE0648FA` |
| `UNSW_NB15_testing-set.csv` | `734FE6642EDF758F7C94D7D9149426B49D202FE8E7BF0BEF47392489C3C0A559` |

```powershell
Get-FileHash data\UNSW_NB15_training-set.csv -Algorithm SHA256
```

## Ejecución

El proceso es reproducible (semilla global `random_state = 42`):

```bash
# 1. Proceso completo: métricas, KPI y figuras
python pipeline.py

# 2. Cuadro de mando interactivo (http://localhost:8501)
streamlit run app.py
```

La primera ejecución del cuadro de mando tarda unos minutos, porque entrena el modelo y puntúa el conjunto de prueba; las siguientes son inmediatas gracias a la caché de Streamlit.

## Metodología

- **Proceso analítico:** CRISP-DM (comprensión del negocio y de los datos, preparación, modelado, evaluación y despliegue).
- **Gestión del proyecto:** enfoque ágil adaptado a un trabajo individual; cada entrega parcial es una iteración.
- **Modelo:** Isolation Forest entrenado exclusivamente con tráfico normal, sin ver ataques ni etiquetas, y evaluado sobre la partición oficial de prueba.
- **Umbral de decisión:** definido como percentil de la puntuación de anomalía del tráfico normal y expuesto como control interactivo del cuadro de mando, para hacer visible el compromiso entre cobertura de detección y tasa de falsas alarmas.

## Alcance

Prueba de concepto académica de análisis **por lotes**. No incluye captura de tráfico privado, despliegue en tiempo real, infraestructura productiva ni comparación sistemática de algoritmos de detección.

## Referencias principales

- N. Moustafa y J. Slay, "UNSW-NB15: A comprehensive data set for network intrusion detection systems (UNSW-NB15 network data set)," *MilCIS*, 2015. doi: 10.1109/MilCIS.2015.7348942
- F. T. Liu, K. M. Ting y Z.-H. Zhou, "Isolation Forest," *IEEE ICDM*, 2008. doi: 10.1109/ICDM.2008.17

La bibliografía completa, en formato IEEE, se recoge en la memoria del TFM.

## Licencia y atribución

El conjunto de datos UNSW-NB15 es propiedad de sus autores (UNSW Canberra); su uso académico requiere citar sus publicaciones. Consulta las condiciones en su página oficial.

## Autoría

Abdelouadoud Khouani · TFM dirigido por David Amorós Alcaraz · UOC
