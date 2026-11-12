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

Los archivos Excel originales no se incluyen en el repositorio.

Deben obtenerse de los datos proporcionados por la cátedra y colocarse en:

```text
notebooks/
└── datos crudos/
    ├── Totalizadores Planta - 2021_2023.xlsx
    ├── Totalizadores Planta 2020_2022.xlsx
    └── Totalizadores Planta 2022_2023.xlsx
```

Los nombres y las rutas deben coincidir con los que utiliza el notebook de preprocesamiento.

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
│   ├── train_model.py
│   └── evaluate_model.py
├── models/
├── results/
├── .gitignore
├── requirements.txt
└── README.md
```

## 8. Ejecución del proyecto

El flujo de trabajo se divide en las siguientes etapas.

### Etapa 1 — Preprocesamiento

Ejecutar:

`notebooks/preprocesamiento.ipynb`

El notebook limpia los datos, genera características, realiza la división temporal de los conjuntos y exporta el dataset procesado.

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

También se consideran los baselines de persistencia y media móvil de siete días.

### Etapa 4 — Modelo final

Una vez seleccionado el modelo definitivo:

```powershell
python src/train_model.py --finalizar
```

El modelo se reentrena con Train + Validation y se guarda versionado junto con su registro.

### Etapa 5 — Evaluación final

Solamente después de finalizar el modelo:

```powershell
python src/evaluate_model.py --test
```

El conjunto de Test se reserva para la evaluación final y no debe utilizarse para seleccionar modelos o hiperparámetros.

## 9. Reproducibilidad

El proyecto utiliza:

- Python 3.12.
- Conda para aislar el entorno.
- uv para gestionar dependencias.
- `requirements.txt` para fijar versiones.
- Semilla aleatoria `42` en los procedimientos que admiten `random_state`.
- Separación temporal Train (70 %), Validation (15 %) y Test (15 %).
- Git para el control de versiones.

Los datos originales se excluyen del repositorio y deben obtenerse por separado.

## 10. Estado del proyecto

El proyecto se encuentra en desarrollo.

Quedan pendientes la validación definitiva del preprocesamiento, la selección del modelo final y la comprobación completa de la reproducibilidad en otro equipo.