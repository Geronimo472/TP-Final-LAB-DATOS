"""Fase 3.1/3.3: métricas, baselines, gráficos y evaluación final del Test.

Uso ordinario: importado por train_model.py para validar.
Uso FINAL, una vez tomada la decisión del modelo:
    python src/evaluate_model.py --test

No se usa el conjunto Test para elegir algoritmos ni hiperparámetros.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calcular_metricas(y_real, predicciones):
    return {
        "MAE": float(mean_absolute_error(y_real, predicciones)),
        "RMSE": float(mean_squared_error(y_real, predicciones) ** 0.5),
        "R2": float(r2_score(y_real, predicciones)),
    }


def imprimir_diagnostico_errores(y_val, residuos, errores_absolutos):
    """Muestra las fechas difíciles y cuánto aportan los mayores errores al MAE."""
    cantidad = min(10, len(y_val))
    mediana = errores_absolutos.median(axis=1)
    fechas = mediana.nlargest(cantidad).index
    peores_fechas = residuos.loc[fechas].copy()
    peores_fechas.insert(0, "Frío real", y_val.loc[fechas])
    peores_fechas["Mediana |error|"] = mediana.loc[fechas]
    peores_fechas.index = (fechas + pd.DateOffset(days=1)).strftime("%Y-%m-%d")
    peores_fechas.index.name = "Día predicho"

    print(f"\n{cantidad} fechas de mayor error compartido en VALIDACIÓN")
    print("Ordenadas por la mediana del error absoluto de los modelos.")
    print("Error = real - predicción. Un valor positivo indica que el modelo predijo de menos.")
    print(peores_fechas.to_string(float_format="{:.2f}".format))

    suma_peores = errores_absolutos.apply(lambda errores: errores.nlargest(cantidad).sum())
    suma_total = errores_absolutos.sum()
    concentracion = pd.DataFrame({
        "MAE": errores_absolutos.mean(),
        "% del error total": suma_peores.div(suma_total.where(suma_total.ne(0))).mul(100),
        "Aporte al MAE": suma_peores.div(len(y_val)),
    })
    print(f"\nPeso de los {cantidad} peores días de cada modelo ({cantidad} de {len(y_val)} fechas)")
    print(concentracion.to_string(float_format="{:.2f}".format, na_rep="—"))


def predecir_baselines(y_historia_previa, y_evaluacion):
    """Predice con la última observación y la media de los datos disponibles en 7 días."""
    historia = pd.concat([y_historia_previa, y_evaluacion]).sort_index()
    if not isinstance(historia.index, pd.DatetimeIndex):
        raise ValueError("Se necesitan fechas para calcular los baselines.")
    if historia.index.has_duplicates:
        raise ValueError("Las fechas del histórico de baselines se repiten.")

    # Si el índice es D, el valor de y corresponde a D+1.
    # Las dos predicciones excluyen el valor de y de la fila evaluada.
    predicciones = {
        "Persistencia": historia.ffill().shift(1),
        "Media móvil 7 días": historia.rolling("7D", closed="left", min_periods=1).mean(),
    }
    return pd.DataFrame(predicciones).reindex(y_evaluacion.index)


def baselines_validacion(y_train, y_val):
    """Evalúa cada baseline en las fechas de Validation en que puede predecir."""
    resultados = {}
    for nombre, prediccion in predecir_baselines(y_train, y_val).items():
        disponible = prediccion.dropna()
        metricas = calcular_metricas(y_val.loc[disponible.index], disponible) if len(disponible) else {}
        resultados[nombre] = {**metricas, "Fechas": len(disponible)}
    return pd.DataFrame.from_dict(resultados, orient="index")


def comparar_baselines(modelos, y_historia_previa, X_evaluacion, y_evaluacion):
    """Compara el modelo y los baselines en las fechas con las tres predicciones."""
    baselines = predecir_baselines(y_historia_previa, y_evaluacion).dropna()
    fechas_comunes = X_evaluacion.index.intersection(baselines.index)
    if fechas_comunes.empty:
        print("No hay fechas con historial suficiente para los dos baselines.")
        return None

    reales = y_evaluacion.loc[fechas_comunes]
    resultados = {
        nombre: calcular_metricas(reales, prediccion)
        for nombre, prediccion in baselines.loc[fechas_comunes].items()
    }
    for nombre, modelo in modelos.items():
        resultados[nombre] = calcular_metricas(
            reales, modelo.predict(X_evaluacion.loc[fechas_comunes])
        )
    tabla = pd.DataFrame(resultados).T.assign(Fechas=len(fechas_comunes)).sort_values("MAE")
    print(f"\nComparación en {len(fechas_comunes)} de {len(X_evaluacion)} fechas comunes:")
    print(tabla)
    if len(fechas_comunes) != len(X_evaluacion):
        print("Se excluyeron fechas sin observaciones disponibles en los 7 días previos.")
    return tabla


def evaluar_mejor_modelo(modelos, tabla, X_val, y_val, graficos=True):
    """Muestra 3.3: residuos, predicción vs realidad, feature importance."""
    ganador = str(tabla.index[0])
    modelo = modelos[ganador]
    print(f"\nMejor modelo por MAE de validación: {ganador}")
    print(tabla.loc[ganador])
    pred = np.asarray(modelo.predict(X_val)).ravel()
    reales = np.asarray(y_val).ravel()
    residuos = reales - pred

    if graficos:
        limites = [min(reales.min(), pred.min()), max(reales.max(), pred.max())]
        plt.figure(figsize=(7, 5))
        plt.scatter(reales, pred, alpha=0.55)
        plt.plot(limites, limites, linestyle="--")
        plt.title(f"Predicción vs realidad — {ganador}")
        plt.xlabel("Consumo real")
        plt.ylabel("Consumo predicho")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(7, 5))
        plt.scatter(pred, residuos, alpha=0.55)
        plt.axhline(0, linestyle="--")
        plt.title(f"Residuos — {ganador}")
        plt.xlabel("Predicción")
        plt.ylabel("Residuo (real - predicción)")
        plt.tight_layout()
        plt.show()

    resultado = permutation_importance(
        modelo, X_val, y_val, scoring="neg_mean_absolute_error",
        n_repeats=5, random_state=42,
    )
    ranking = pd.DataFrame({
        "Variable": X_val.columns,
        "Importancia": resultado.importances_mean,
    }).sort_values("Importancia", ascending=False)
    print("\nImportancia de variables (variación estimada del MAE):")
    print(ranking.head(15).to_string(index=False))
    return ranking


def evaluar_test_final(version=None, ruta_csv=None, forzar=False):
    """Evalúa una vez el modelo guardado, sin usar Test para seleccionar."""
    from train_model import RAIZ, cargar_datos_preprocesados, resolver_ruta_csv, hash_archivo

    registro = RAIZ / "models" / "model_registry.json"
    if not registro.exists():
        raise FileNotFoundError("Falta models/model_registry.json. Ejecutá train_model.py --finalizar.")
    with registro.open(encoding="utf-8") as f:
        catalogo = json.load(f)
    version = version or catalogo["latest"]
    info = catalogo["models"].get(version)
    if info is None:
        raise ValueError(f"No existe la versión {version!r} en el registro.")
    ruta_resultado = RAIZ / "results" / f"evaluacion_test_{version}.json"
    if ruta_resultado.exists() and not forzar:
        raise FileExistsError(
            f"El Test de {version} ya fue evaluado ({ruta_resultado}). "
            "No se debe usar repetidamente para ajustar el modelo."
        )
    ruta_datos = resolver_ruta_csv(ruta_csv)
    if hash_archivo(ruta_datos) != info["dataset_sha256"]:
        raise ValueError("El dataset cambió desde el entrenamiento final. No se puede comparar este Test.")

    X_train, y_train, X_val, y_val, X_test, y_test = cargar_datos_preprocesados(ruta_datos)
    columnas = info["columnas_entrada"]
    if list(X_test.columns) != columnas:
        raise ValueError("Las columnas del Test no coinciden con las usadas en entrenamiento.")

    ruta_modelo = RAIZ / info["path"]
    # Solo cargar archivos joblib propios y confiables: la deserialización es ejecutable.
    modelo = joblib.load(ruta_modelo)
    metricas_test = calcular_metricas(y_test, modelo.predict(X_test))
    print(f"\nEvaluación final (Test) de {version}: {info['algoritmo']}")
    print(pd.Series(metricas_test))
    print("MAE menor a 3000:", metricas_test["MAE"] < 3000)

    comparacion = comparar_baselines(
        {info["algoritmo"]: modelo},
        pd.concat([y_train, y_val]).sort_index(),
        X_test, y_test,
    )
    ruta_resultado.parent.mkdir(parents=True, exist_ok=True)
    reporte = {
        "version": version,
        "algoritmo": info["algoritmo"],
        "fecha_utc": datetime.now(timezone.utc).isoformat(),
        "n_test": int(len(y_test)),
        "n_fechas_comunes": int(comparacion["Fechas"].iloc[0]) if comparacion is not None else 0,
        "metricas_test_completo": metricas_test,
        "comparacion_fechas_comunes": (
            comparacion.to_dict(orient="index") if comparacion is not None else None
        ),
    }
    with ruta_resultado.open("w", encoding="utf-8") as f:
        json.dump(reporte, f, indent=2, ensure_ascii=False)
    print(f"Resultado guardado: {ruta_resultado}")
    return reporte


def main():
    parser = argparse.ArgumentParser(description="Evaluación final, reservada para Test")
    parser.add_argument("--test", action="store_true", help="Autoriza la única evaluación final de Test")
    parser.add_argument("--version", help="Versión; por defecto la más reciente")
    parser.add_argument("--data", help="Ruta opcional al CSV procesado")
    parser.add_argument("--forzar-repeticion", action="store_true",
                        help="Solo en caso de repetición deliberada y documentada")
    args = parser.parse_args()
    if not args.test:
        parser.error("Para abrir el Test tenés que indicar explícitamente --test")
    evaluar_test_final(args.version, args.data, args.forzar_repeticion)


if __name__ == "__main__":
    main()
