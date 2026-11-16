# TP Final — Laboratorio de Datos II

## Predicción del Consumo Energético de una Planta Cervecera

Proyecto de Machine Learning para predecir el consumo energético del sistema de refrigeración (`Frio (kW)`) del día siguiente en una planta cervecera mexicana.

El proyecto incluye análisis exploratorio, preprocesamiento, entrenamiento de modelos, optimización de hiperparámetros y evaluación de resultados.

## 1. Requisitos previos

Para ejecutar el proyecto se necesita:

- **Python 3.12**
- **Conda** (Miniforge o Miniconda)
- **uv** (gestor de dependencias)
- **Git**
- VS Code con las extensiones Python y Jupyter (recomendado)

En Windows, uv puede instalarse mediante:

```powershell
winget install --id astral-sh.uv -e
```

Verificar la instalación:

```powershell
conda --version
uv --version
git --version
```

## 2. Clonar el repositorio

```bash
git clone https://github.com/Geronimo472/TP-Final-LAB-DATOS.git
cd TP-Final-LAB-DATOS
```

## 3. Configuración del entorno virtual

Crear un entorno aislado con Python 3.12:

```powershell
conda create -n cervecera_env python=3.12 -y
```

Activarlo:

```powershell
conda activate cervecera_env
```

Comprobar la versión:

```powershell
python --version
```

Debe indicar Python 3.12.x.

## 4. Instalación de dependencias

Con el entorno activado, ejecutar desde la raíz del proyecto:

```powershell
uv pip install --python "$env:CONDA_PREFIX\python.exe" -r requirements.txt
```

El archivo `requirements.txt` contiene las dependencias y versiones utilizadas por el proyecto, congeladas con `uv pip freeze`. `pywinpty` solo se instala en Windows (lleva el marcador `sys_platform == "win32"`), así que el mismo archivo funciona en Linux y Mac.

Para comprobar la instalación:

```powershell
python -c "import pandas, sklearn, xgboost, lightgbm, optuna; print('Dependencias instaladas correctamente')"
```

**Importante:** todas las instalaciones deben realizarse sobre el entorno `cervecera_env`.

## 5. Configuración de Jupyter en VS Code

1. Abrir el proyecto en VS Code.
2. Seleccionar el intérprete Python del entorno `cervecera_env`.
3. Abrir cualquier notebook `.ipynb`.
4. Seleccionar `cervecera_env` como kernel de Jupyter.

Esto garantiza que los notebooks utilicen las mismas dependencias que los scripts del proyecto.

## 6. Datos del proyecto

Conseguí los archivos Excel originales de la cátedra y copiá los tres archivos de trabajo en la carpeta local `data/raw/`. Creá esa carpeta si hace falta. Git la ignora.

```text
data/
└── raw/
    ├── Totalizadores Planta - 2021_2023.xlsx
    ├── Totalizadores Planta 2020_2022.xlsx
    └── Totalizadores Planta 2022_2023.xlsx
```

El EDA prioriza esa carpeta. Para conservar compatibilidad con las ejecuciones anteriores, también acepta los mismos archivos en `notebooks/datos crudos`. Su paso 5.1 genera `data/dataset_unificado_inicial.csv`, que usa el notebook de preprocesamiento.

Después del procesamiento se genera:

```text
data/processed/dataset_final.csv
```

Este archivo contiene los datos preparados para entrenar los modelos. `data/processed/data_lineage.json` registra de qué archivos y con qué parámetros se generó.

### Verificación con checksums

`data/checksums.json` guarda el SHA-256 de `data/dataset_unificado_inicial.csv` y de `data/processed/dataset_final.csv`. Para comprobar que una ejecución reprodujo los mismos datos, comparar esos valores con:

```powershell
Get-FileHash data/processed/dataset_final.csv -Algorithm SHA256
Get-FileHash data/dataset_unificado_inicial.csv -Algorithm SHA256
```

En Linux o Mac: `sha256sum data/processed/dataset_final.csv data/dataset_unificado_inicial.csv`. Los hashes se comparan sin distinguir mayúsculas y minúsculas. El registro del modelo (`models/model_registry.json`) guarda además el hash del `dataset_final.csv` con el que se entrenó cada versión.

## 7. Estructura del proyecto

```text
TP-Final-LAB-DATOS/
├── data/
│   ├── raw/                        # Excel originales, solo en local
│   ├── checksums.json              # SHA-256 de los datos generados
│   └── processed/
│       ├── dataset_final.csv       # dataset sin escalar para modelar
│       └── data_lineage.json       # origen y parámetros del dataset
├── notebooks/
│   ├── eda.ipynb                   # análisis exploratorio
│   ├── preprocesamiento.ipynb      # limpieza, features y selección de variables
│   ├── optuna_optimization.ipynb   # búsqueda de hiperparámetros
│   └── modelado.ipynb              # comparación de modelos y baselines
├── src/
│   ├── preprocessing_pipeline.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── predict.py
├── models/
│   ├── modelo_v1.0.0.pkl           # modelo final con su preprocesamiento
│   ├── model_registry.json         # registro de versiones
│   └── best_params.json            # hiperparámetros de Optuna
├── results/
│   ├── experiment_logs.csv
│   ├── evaluacion_test_v1.0.0.json
│   └── seleccion_variables/        # comparación SelectKBest, RFE y Lasso
├── .gitattributes
├── .gitignore
├── consigna.md
├── preguntas_informe.md            # respuestas a las preguntas del informe
├── requirements.txt
└── README.md
```

## 8. Ejecución del proyecto

El flujo de trabajo se divide en las siguientes etapas.

### Etapa 1 — Preprocesamiento

1. Ejecutar `notebooks/eda.ipynb` hasta el paso 5.1 para exportar la tabla unificada.
2. Desde la raíz del repositorio, ejecutar:

```powershell
python src/preprocessing_pipeline.py
```

El notebook `notebooks/preprocesamiento.ipynb` documenta y permite inspeccionar los mismos criterios de preparación.

El pipeline limpia los datos, genera características (rezagos y medias móviles), realiza la división temporal de los conjuntos, selecciona 10 variables con SelectKBest (ANOVA F) ajustado solo en Train y exporta `data/processed/dataset_final.csv`. La ruta de entrada habitual es `data/dataset_unificado_inicial.csv`. Puede indicarse otra con `--data RUTA`.

El comando también genera `data/processed/preprocesador.joblib` con los imputadores, los límites de atípicos y las variables seleccionadas en Train. Para guardar un modelo apto para predecir desde Excel, generá el estado y el CSV juntos con el script.

### Etapa 2 — Optimización con Optuna

Ejecutar:

`notebooks/optuna_optimization.ipynb`

Se optimizan los hiperparámetros de los modelos y se generan gráficos de diagnóstico.

Los mejores parámetros se guardan en:

`models/best_params.json`

### Etapa 3 — Entrenamiento

Desde la raíz del repositorio:

```powershell
python src/train_model.py
```

Se entrenan y comparan los modelos:

- XGBoost
- Random Forest
- LightGBM
- Ridge Regression

Se calculan las métricas MAE, RMSE y R² utilizando el conjunto de validación.

Validation y Test usan los mismos baselines. Persistencia toma la última observación disponible. La media móvil promedia las observaciones disponibles de los siete días previos. En Test se reporta el número de fechas en las que se pudo comparar el modelo con ambos baselines.

### Etapa 4 — Modelo final

Una vez seleccionado el modelo definitivo:

```powershell
python src/train_model.py --finalizar
```

El modelo se reentrena con Train + Validation y se guarda versionado junto con su registro. El archivo del modelo incluye el preprocesamiento ajustado para que se pueda ejecutar `predict.py` en otro equipo.

### Etapa 5 — Evaluación final

Solamente después de finalizar el modelo:

```powershell
python src/evaluate_model.py --test
```

El conjunto de Test se reserva para la evaluación final y no debe utilizarse para seleccionar modelos o hiperparámetros.

### Etapa 6 — Predicción sobre un Excel nuevo

Con el modelo y `models/model_registry.json` generados, ejecutá desde la raíz:

```powershell
python src/predict.py 'ruta/al/archivo.xlsx'
```

El script usa la versión semántica más alta del registro. Para elegir otra, agregá `--version v1.0.0`. Escribe `results/predicciones.csv` con `fecha` (día D de entrada) y `prediccion_frio_kw` (pronóstico para D+1), una fila por día válido. `--salida` permite elegir otra ruta.

El Excel puede traer columnas y hojas adicionales. Las hojas necesarias se leen por nombre y deben tener `DIA`, `HORA` y las mediciones requeridas por el modelo. Incorpora el historial de los 7 días anteriores si está disponible para calcular los rezagos y las medias móviles desde el primer día. Los faltantes se procesan con los valores aprendidos en entrenamiento.

## 9. Reproducibilidad

El proyecto utiliza:

- Python 3.12.
- Conda para aislar el entorno.
- uv para gestionar dependencias.
- `requirements.txt` para fijar versiones.
- Semilla aleatoria `42` en los procedimientos que admiten `random_state`.
- Separación temporal Train (70 %), Validation (15 %) y Test (15 %).
- Git para el control de versiones.

## 10. Estado del proyecto

Modelo final: **Ridge v1.0.0** (`alpha` ≈ 13,72), con 10 variables seleccionadas por SelectKBest (ANOVA F). Está registrado en `models/model_registry.json` y su commit lleva la etiqueta `v1.0.0`.

Resultados en Test (165 días, del 27/04/2023 al 24/10/2023, `results/evaluacion_test_v1.0.0.json`):

| Método | MAE (kW) | RMSE (kW) | R² |
| --- | ---: | ---: | ---: |
| Persistencia | 3.369 | 4.410 | 0,40 |
| Media móvil 7 días | 3.033 | 3.889 | 0,54 |
| **Ridge v1.0.0** | **2.773** | **3.651** | **0,59** |

El modelo se eligió con Validation (MAE 2.265 kW). Test se usó una sola vez, para la evaluación final. Las decisiones y limitaciones se justifican en `preguntas_informe.md`.
