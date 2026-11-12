import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
import pandas as pd
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

def calcular_baselines(df, columna_fecha, columna_frio):
    # Ordenamos los datos cronológicamente
    df = df.sort_values(columna_fecha).copy()

    # Baseline 1: mañana consume lo mismo que hoy
    df["baseline_persistencia"] = df[columna_frio]

    # Baseline 2: promedio de los últimos 7 días
    df["baseline_media_7"] = (
        df[columna_frio]
        .rolling(window=7, min_periods=7)
        .mean()
    )

    return df


def crear_modelo(nombre, params):

    if nombre == "XGBoost":
        return XGBRegressor(
            **params,
            random_state=42,
            n_jobs=1,
            verbosity=0
        )

    elif nombre == "Random Forest":
        return RandomForestRegressor(
            **params,
            random_state=42,
            n_jobs=1
        )

    elif nombre == "LightGBM":
        return LGBMRegressor(
            **params,
            random_state=42,
            n_jobs=1,
            verbosity=-1
        )

    elif nombre == "Ridge":
        return make_pipeline(
            StandardScaler(),
            Ridge(alpha=params["alpha"])
        )

    else:
        raise ValueError(f"Modelo desconocido: {nombre}")


def optimizar_modelos(X_train, y_train, X_val, y_val, n_trials=20):

    nombres = ["XGBoost", "Random Forest", "LightGBM", "Ridge"]

    modelos = {}
    resultados = {}

    for nombre in nombres:

        print(f"\nOptimizando {nombre}...")

        # Función que Optuna intentará minimizar
        def objetivo(trial):

            # 1. Espacio de búsqueda de hiperparámetros
            if nombre == "XGBoost":
                params = {
                    "n_estimators": trial.suggest_int(
                        "n_estimators", 100, 500, step=100
                    ),
                    "max_depth": trial.suggest_int(
                        "max_depth", 2, 10
                    ),
                    "learning_rate": trial.suggest_float(
                        "learning_rate", 0.01, 0.2, log=True
                    ),
                    "subsample": trial.suggest_float(
                        "subsample", 0.6, 1.0
                    )
                }

            elif nombre == "Random Forest":
                params = {
                    "n_estimators": trial.suggest_int(
                        "n_estimators", 100, 400, step=50
                    ),
                    "max_depth": trial.suggest_int(
                        "max_depth", 3, 20
                    ),
                    "min_samples_split": trial.suggest_int(
                        "min_samples_split", 2, 10
                    ),
                    "min_samples_leaf": trial.suggest_int(
                        "min_samples_leaf", 1, 5
                    )
                }

            elif nombre == "LightGBM":
                params = {
                    "n_estimators": trial.suggest_int(
                        "n_estimators", 100, 500, step=100
                    ),
                    "num_leaves": trial.suggest_int(
                        "num_leaves", 15, 63
                    ),
                    "learning_rate": trial.suggest_float(
                        "learning_rate", 0.01, 0.2, log=True
                    ),
                    "max_depth": trial.suggest_int(
                        "max_depth", 3, 12
                    )
                }

            else:  # Ridge
                params = {
                    "alpha": trial.suggest_float(
                        "alpha", 0.001, 1000, log=True
                    )
                }

            # 2. Crear y entrenar modelo
            modelo = crear_modelo(nombre, params)
            modelo.fit(X_train, y_train)

            # 3. Evaluar en validación
            predicciones = modelo.predict(X_val)

            metricas = {
                "MAE": mean_absolute_error(y_val, predicciones),
                "RMSE": mean_squared_error(y_val, predicciones) ** 0.5,
                "R2": r2_score(y_val, predicciones)
            }

            # 4. Registrar cada experimento
            registrar_experimento(
                f"{nombre} - trial {trial.number}",
                modelo,
                metricas
            )

            return metricas["MAE"]

        # 5. Ejecutar la optimización
        estudio = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42)
        )

        estudio.optimize(objetivo, n_trials=n_trials)

        # 6. Recuperar los mejores hiperparámetros
        mejores_params = estudio.best_params

        print(f"Mejores parámetros de {nombre}:")
        print(mejores_params)

        # 7. Entrenar el mejor modelo con Train
        mejor_modelo = crear_modelo(nombre, mejores_params)
        mejor_modelo.fit(X_train, y_train)

        modelos[nombre] = mejor_modelo

        # 8. Evaluar en Validation
        predicciones = mejor_modelo.predict(X_val)

        resultados[nombre] = {
            "MAE": mean_absolute_error(y_val, predicciones),
            "RMSE": mean_squared_error(y_val, predicciones) ** 0.5,
            "R2": r2_score(y_val, predicciones)
        }

    # 9. Comparar modelos optimizados
    tabla = pd.DataFrame(resultados).T
    tabla = tabla.sort_values("MAE")

    return modelos, tabla


def entrenar_modelos(X_train, y_train, X_val, y_val):

    # 1. Definimos los cuatro modelos
    modelos = {
        "XGBoost": XGBRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(random_state=42),
        "LightGBM": LGBMRegressor(random_state=42, verbosity=-1),
        "Ridge": make_pipeline(StandardScaler(), Ridge())
    }

    resultados = {}

    # 2. Entrenamos y evaluamos cada modelo
    for nombre, modelo in modelos.items():

        modelo.fit(X_train, y_train)

        predicciones = modelo.predict(X_val)

        # Calculamos las métricas
        resultados[nombre] = {
            "MAE": mean_absolute_error(y_val, predicciones),
            "RMSE": mean_squared_error(y_val, predicciones) ** 0.5,
            "R2": r2_score(y_val, predicciones)
        }

        # 3. Registramos automáticamente el experimento
        registrar_experimento(
            nombre,
            modelo,
            resultados[nombre]
        )

    # 4. Comparamos los resultados
    tabla = pd.DataFrame(resultados).T

    tabla = tabla.sort_values("MAE")

    return modelos, tabla



def registrar_experimento(nombre, modelo, metricas):

    ruta = (
        Path(__file__).resolve().parent.parent
        / "results"
        / "experiment_logs.csv"
    )

    # Crear la carpeta results si no existe
    ruta.parent.mkdir(parents=True, exist_ok=True)

    # Comprobar si el CSV ya tiene contenido
    existe = ruta.exists() and ruta.stat().st_size > 0

    experimento = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "modelo": nombre,
        "hiperparametros": json.dumps(
            modelo.get_params(deep=True),
            default=str
        ),
        "MAE": metricas["MAE"],
        "RMSE": metricas["RMSE"],
        "R2": metricas["R2"]
    }

    # Agregar una fila sin borrar experimentos anteriores
    with open(ruta, "a", newline="", encoding="utf-8") as archivo:

        writer = csv.DictWriter(
            archivo,
            fieldnames=list(experimento.keys())
        )

        if not existe:
            writer.writeheader()

        writer.writerow(experimento)


def evaluar_mejor_modelo(modelos, tabla, X_val, y_val):

    # 1. Seleccionamos el modelo con menor MAE
    nombre = tabla.index[0]
    modelo = modelos[nombre]

    print(f"Mejor modelo: {nombre}")
    print(tabla.loc[nombre])

    # 2. Calculamos predicciones y residuos
    predicciones = modelo.predict(X_val)
    valores_reales = np.asarray(y_val).ravel()
    residuos = valores_reales - predicciones

    # 3. Gráfico: predicción vs realidad
    plt.figure(figsize=(7, 5))

    plt.scatter(valores_reales, predicciones, alpha=0.5)

    limites = [
        min(valores_reales.min(), predicciones.min()),
        max(valores_reales.max(), predicciones.max())
    ]

    plt.plot(limites, limites, "r--")

    plt.xlabel("Consumo real")
    plt.ylabel("Consumo predicho")
    plt.title(f"Predicción vs realidad - {nombre}")
    plt.show()

    # 4. Gráfico de residuos
    plt.figure(figsize=(7, 5))

    plt.scatter(predicciones, residuos, alpha=0.5)
    plt.axhline(0, color="red", linestyle="--")

    plt.xlabel("Consumo predicho")
    plt.ylabel("Residuo")
    plt.title(f"Análisis de residuos - {nombre}")
    plt.show()

    # 5. Importancia de variables
    importancia = permutation_importance(
        modelo,
        X_val,
        y_val,
        scoring="neg_mean_absolute_error",
        n_repeats=5,
        random_state=42
    )

    if hasattr(X_val, "columns"):
        nombres_variables = X_val.columns
    else:
        nombres_variables = [
            f"Variable {i}" for i in range(X_val.shape[1])
        ]

    ranking = pd.DataFrame({
        "Variable": nombres_variables,
        "Importancia": importancia.importances_mean
    })

    ranking = ranking.sort_values(
        "Importancia", ascending=False
    )

    print("\nImportancia de variables:")
    print(ranking.head(15))

    return ranking


modelos, tabla = optimizar_modelos(
    X_train, y_train,
    X_val, y_val,
    n_trials=20
)

print(tabla)

ranking = evaluar_mejor_modelo(
    modelos, tabla, X_val, y_val
)
