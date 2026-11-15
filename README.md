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

El archivo `requirements.txt` contiene las dependencias y versiones utilizadas por el proyecto.

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

Los archivos Excel originales se gestionan mediante Git LFS. Después de clonar el repositorio, ejecutar `git lfs pull` para descargarlos en:

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

Este archivo contiene los datos preparados para entrenar los modelos.

## 7. Estructura del proyecto

```text
TP-Final-LAB-DATOS/
├── data/
│   └── processed/
├── notebooks/
│   ├── eda.ipynb
│   ├── preprocesamiento.ipynb
│   └── optuna_optimization.ipynb
├── src/
│   ├── preprocessing_pipeline.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── predict.py
├── models/
├── results/
├── .gitignore
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

El pipeline limpia los datos, genera características, realiza la división temporal de los conjuntos y exporta `data/processed/dataset_final.csv`. La ruta de entrada habitual es `data/dataset_unificado_inicial.csv`. Puede indicarse otra con `--data RUTA`.

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
- Git LFS para versionar los Excel originales sin incorporarlos directamente al historial normal de Git.

Los datos originales quedan almacenados en Git LFS y se materializan en `data/raw` al ejecutar `git lfs pull`.

## 10. Estado del proyecto

El proyecto se encuentra en desarrollo.

Quedan pendientes la validación definitiva del preprocesamiento, la selección del modelo final y la comprobación completa de la reproducibilidad en otro equipo.
