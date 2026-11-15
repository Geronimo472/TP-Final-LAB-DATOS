# Trabajo Práctico Final - Predicción de Consumo Energético Industrial

Versión en Markdown del enunciado «Trabajo Práctico Final 2026».

## Objetivo general

Desarrollar un pipeline de machine learning completo y reproducible para predecir el consumo energético del sistema de refrigeración, **Frio (kW)**, del día siguiente en una planta cervecera mexicana. Se evalúan la precisión del modelo y las prácticas de MLOps que aseguran la trazabilidad, la colaboración y la gestión del ciclo de vida del proyecto.

## Dataset y variable objetivo

Se proporcionan archivos Excel con registros horarios de métricas operacionales de la planta desde 2021 hasta 2024:

- `Totalizadores Planta de Cerveza 2023_2024.xlsx`
- `Totalizadores Planta de Cerveza - 2022_2023.xlsx`
- `Totalizadores Planta de Cerveza 2021_2022.xlsx`
- `Totalizadores Planta de Cerveza 2024_2025.xlsx`. No proporcionado. Corresponde al Test oculto.

**Paso 1. Calcular el consumo total diario.** Las variables son totalizadores horarios. Cada lectura mide el consumo acumulado desde el inicio del día. El último valor registrado de la jornada representa el consumo diario y se usa como característica de esa fecha.

Las lecturas de las 23:00 y las 23:59 difieren. Para los días con ambas, la diferencia promedio es del orden de 800 kW, cerca del 20 % del margen de error permitido. Elegir una definición, aplicarla consistentemente y justificarla en el informe.

**Paso 2. Construir la variable objetivo.** Alinear las características del día D con el consumo de Frio (kW) del día D+1. El enunciado recomienda `df.shift(-1)` sobre la columna de Frio en el dataframe diario.

## Fase 0 — Configuración del entorno y control de versiones

### 0.1. Dependencias y entorno reproducible

Crear un entorno aislado con Python 3.12. Se recomienda usar Conda y uv. Generar `requirements.txt` con versiones específicas.

```bash
conda create -n cervecera_env python=3.12
conda activate cervecera_env
pip install uv
uv pip install pandas numpy scikit-learn jupyter xgboost lightgbm
uv pip freeze > requirements.txt
```

### 0.2. Repositorio Git

- Inicializar un repositorio en la raíz del proyecto y sincronizarlo con un remoto, como GitHub o GitLab.
- Documentar en `README.md` cómo clonar, configurar el entorno a partir de `requirements.txt` y ejecutar los scripts principales.

### 0.3. Estrategia de ramas

Mantener `main` estable. Desarrollar cada fase en su rama e integrarla mediante Pull Request a `develop` o directamente a `main`.

- `feature/eda`
- `feature/preprocessing`
- `feature/modeling`
- `feature/prediction-pipeline`

## Fase 1 — Análisis exploratorio de datos (EDA)

Trabajar en `feature/eda`. Se recomienda analizar los Excel hoja por hoja y formular hipótesis sobre qué procesos de la planta se relacionan con el consumo de frío. Evaluar la variabilidad de las métricas y cuantificar sus relaciones directas e inversas con el objetivo para justificar un subconjunto de variables informativo y manejable.

### 1.1. Análisis descriptivo y versionado inicial

- Unificar todos los Excel en un dataframe. Cuantificar las fechas solapadas, verificar si coinciden los totales diarios y justificar una regla de desempate. Verificar que no haya más de 25 registros por fecha antes de agregar por día.
- Calcular el SHA-256 del dataset unificado inicial leyendo por bloques, como en `huella()` del capítulo 10. Guardar el par archivo y hash en `data/checksums.json` y justificar por qué se usa SHA-256 en lugar de MD5.
- Documentar dimensiones, tipos, rangos y valores faltantes.
- Detectar atípicos y registros inválidos. Revisar días con menos de 24 lecturas horarias, valores imposibles del orden de los millones y días sin reinicio del totalizador a medianoche.

### 1.2. Análisis temporal

- Mostrar la evolución del consumo de Frio (kW), la estacionalidad mensual y anual, y los patrones diarios y semanales.
- Medir el efecto del día de la semana. Si es despreciable, justificar el descarte de esas variables con la medición.

### 1.3. Correlaciones

- Calcular la matriz de correlación de variables numéricas e identificar las 15 variables más correlacionadas con el target.
- Analizar la multicolinealidad entre predictores y visualizar las correlaciones con mapas de calor.

### 1.4. Visualizaciones requeridas

- Serie temporal de Frio (kW).
- Boxplots mensuales.
- Histograma y QQ-plot del target.
- Gráficos de dispersión de las cinco variables más correlacionadas contra el target.
- Análisis de consumo por área de la planta.

**Entregable.** `notebooks/eda.ipynb` con análisis, conclusiones y un Pull Request de integración.

## Fase 2 — Preprocesamiento y selección de variables

Trabajar en `feature/preprocessing`.

### 2.1. Limpieza

Tratar días incompletos, totales imposibles y días sin reinicio del totalizador. Manejar los faltantes y eliminar o corregir outliers extremos. Documentar y justificar las decisiones en el notebook.

### 2.2. Ingeniería de variables

Se sugieren promedios móviles, rezagos, día de la semana, mes, indicador binario de fin de semana, ratios entre consumos de distintas áreas, medidas de eficiencia energética e interacciones. En la hoja `Consolidado KPI`, las columnas que empiezan con `Meta` expresan metas por hectolitro.

### 2.3. Selección de variables

Aplicar al menos dos métodos, como RFE, importancia de Random Forest o Lasso, y justificar el conjunto final.

### 2.4. Preparación y versionado de datos

- Documentar las transformaciones, el script y la fecha en un registro de linaje, por ejemplo `data/processed/data_lineage.json`.
- Guardar `data/processed/dataset_final.csv` y agregar su checksum a `data/checksums.json`.
- Dividir cronológicamente en Train (70 %), Validation (15 %) y Test (15 %, tramo más reciente).
- Elegir el modelo y sus hiperparámetros en Validation o mediante `TimeSeriesSplit` sobre Train. Reservar Test para reportar los resultados al final.
- Ajustar la normalización o estandarización dentro de un pipeline de scikit-learn con datos de entrenamiento. Guardar el CSV codificado y depurado en su escala original.
- Documentar el tamaño final de `X_train`, `X_val` y `X_test`.

**Entregables.** `notebooks/preprocesamiento.ipynb`, `src/preprocessing_pipeline.py` y un Pull Request de integración.

## Fase 3 — Entrenamiento y versionado de modelos

Trabajar en `feature/modeling`.

### 3.1. Modelos requeridos

Implementar primero los dos baselines obligatorios:

- **Persistencia.** Pronosticar para D+1 el consumo observado en D.
- **Media móvil de siete días.** Pronosticar para D+1 el promedio de los siete días anteriores.

Reportar el MAE de ambos baselines en el mismo conjunto y con el mismo preprocesamiento que el modelo final. Entrenar y comparar al menos XGBoost, Random Forest Regressor, LightGBM y una regresión Ridge o Lasso.

### 3.2. Optimización y seguimiento de experimentos

Registrar automáticamente los hiperparámetros, MAE, RMSE, R² y timestamp de cada ejecución de entrenamiento en `results/experiment_logs.csv`. Documentar el espacio de búsqueda y justificar el modelo elegido a partir de los experimentos.

### 3.3. Evaluación

Comparar resultados en Validation, analizar residuos, mostrar predicciones contra valores reales e incluir la importancia de variables del mejor modelo.

### 3.4. Modelo final y registro

- Elegir por MAE de Validation y reentrenar con Train + Validation.
- Serializar el modelo con una versión semántica, por ejemplo `models/modelo_v1.0.0.pkl`.
- Registrar versión, hiperparámetros finales, métricas de Validation, ruta del modelo y hash del commit de entrenamiento en `models/model_registry.json`.

**Entregables.** `notebooks/modelado.ipynb`, `src/train_model.py`, el archivo serializado del modelo, `models/model_registry.json` y un Pull Request de integración.

## Fase 4 — Pipeline de predicción

Trabajar en `feature/prediction-pipeline`. Crear `src/predict.py` para automatizar la predicción sobre un Excel nuevo, con la siguiente estructura:

- `load_model(version="latest")` carga el modelo y el preprocesamiento desde el registro. `latest` corresponde a la versión más alta.
- `load_and_preprocess(filepath)` carga el Excel y reutiliza `src/preprocessing_pipeline.py`. Devuelve las características y sus fechas.
- `predict_consumption(filepath)` combina ambos pasos y produce una fila por día con `fecha` y `prediccion_frio_kw`. `fecha` representa el día D, y la predicción corresponde a D+1.
- La ejecución por terminal toma el Excel de entrada y guarda `results/predicciones.csv`.

El script aplica el mismo preprocesamiento del entrenamiento, maneja errores y consulta hojas y columnas por nombre. El Excel de Test puede traer hojas o columnas adicionales y cambiar su orden. Si falta una hoja o columna necesaria, el error debe identificarla claramente.

**Entregable.** `src/predict.py` funcional y un Pull Request final.

## Conclusiones del informe

Responder las cinco preguntas con decisiones respaldadas por números medidos en el proyecto.

1. **¿Su modelo agrega valor?** Mostrar en una tabla el MAE de persistencia, media móvil y el modelo final sobre el mismo Test. Si el mejor baseline se mejora en menos del 10 %, explicar qué se gana con la complejidad adicional y si esa mejora la justifica.
2. **¿Qué variables se parecen demasiado al objetivo?** Revisar las variables cuya correlación con el Frío del mismo día supera 0,98. Decidir si entran al modelo y justificar cuándo se conoce cada dato. Explicar cómo se verificó que para pronosticar D+1 se usó información disponible en D.
3. **Un resultado que contradice la intuición.** Comparar el patrón semanal planteado en el enunciado con la medición de la Fase 1. Si difieren, indicar el número, qué pudo haber borrado el patrón y qué decisión tomaron.
4. **Defiendan un umbral.** Estimar cuántos kW de diferencia en MAE justifican reemplazar un modelo desplegado. Fundamentarlo, por ejemplo, con el desvío entre folds, la variabilidad del target o el costo de equivocarse.
5. **Lo que no usaron.** Elegir dos técnicas de los prácticos 1 a 5 que descartaron, como PCA, MICE, codificación cíclica, LOF o DVC, y justificar cada decisión.

## Modalidad y entregas

El trabajo se realiza en grupos de dos o tres integrantes. `git log --oneline` debe mostrar commits de todos los integrantes distribuidos en el tiempo.

| Entrega | Semana | Qué se entrega |
| --- | ---: | --- |
| 1 | 8 | Presentación de ideas |
| 2 | 9 | Propuesta formal |
| 3 | 12 | Presentación y defensa del código |
| 4 | 14 | Defensa final |

La defensa final es oral, ante la cátedra y con el proyecto funcionando. Cada integrante debe poder justificar cualquier parte. Se entrega además un informe con las decisiones, sus justificaciones y las limitaciones del modelo.

## Criterios de evaluación

Las cuatro condiciones son eliminatorias:

1. Reportar el MAE de ambos baselines sobre el mismo Test que el modelo final y superar a ambos.
2. Obtener **MAE < 3000 kW en el Test oculto**.
3. Asegurar que otra máquina pueda clonar, instalar `requirements.txt` con uv y ejecutar los scripts siguiendo el README.
4. Hacer que `predict.py` se ejecute sin errores y que todas las fases se reflejen en el historial de Git con sus ramas y Pull Requests.

| Componente | Puntos |
| --- | ---: |
| Fase 0 — Entorno, control de versiones y trazabilidad | 15 |
| Fase 1 — Análisis exploratorio y conclusiones | 20 |
| Fase 2 — Preprocesamiento, ingeniería y selección de variables | 25 |
| Fase 3 — Modelado, baselines, experimentos y registro | 25 |
| Fase 4 — Pipeline de predicción | 15 |
| **Total** | **100** |

Se aprueba con 60 puntos si se cumplen las cuatro condiciones eliminatorias.

| Penalización | Puntos |
| --- | ---: |
| Código no reproducible o entorno mal configurado | −20 |
| Falta de documentación, comentarios o justificaciones | −10 |
| Incumplimiento de la estrategia de control de versiones y ramas | −10 |
| Fallas de versionado de datos, tracking o registro de modelos | −10 |
| Uso del Test para seleccionar el modelo | −15 |

## Estructura de la entrega final

```text
tp_final/
├── data/
│   ├── Totalizadores Planta de Cerveza*.xlsx
│   ├── checksums.json
│   └── processed/
│       ├── dataset_final.csv
│       └── data_lineage.json
├── models/
│   ├── modelo_v1.0.0.pkl
│   └── model_registry.json
├── notebooks/
│   ├── eda.ipynb
│   ├── preprocesamiento.ipynb
│   └── modelado.ipynb
├── src/
│   ├── preprocessing_pipeline.py
│   ├── train_model.py
│   ├── predict.py
│   └── auxiliar_functions.py
├── results/
│   ├── predicciones.csv
│   └── experiment_logs.csv
├── .gitignore
├── README.md
├── requirements.txt
└── consigna.md
```

## Requisitos técnicos

- Usar Python 3.12, un entorno Conda con uv y `requirements.txt` congelado con `uv pip freeze`.
- Documentar en el README todos los pasos que requiere una instalación limpia.
- Declarar `SEMILLA = 42` y usarla en las particiones y estimadores que acepten `random_state`. Las ejecuciones sucesivas deben dar los mismos números.
- Ignorar los cuatro Excel con `.gitignore` y explicar en el README de dónde obtenerlos. Versionar `data/checksums.json` para verificar que los archivos coincidan.
- Versionar el modelo serializado con un nombre semántico y etiquetar su commit, por ejemplo con `git tag -a v1.0.0`. La etiqueta debe coincidir con la versión del registro.

## Recomendaciones

- Respetar el orden temporal y la información disponible al hacer el pronóstico del día siguiente.
- Considerar la temperatura ambiente en Tecate, Baja California, México (Open-Meteo), el volumen de producción, la etapa de fermentación y los patrones de producción por día de la semana.
- Justificar las transformaciones, la importancia de variables y las limitaciones del modelo.
