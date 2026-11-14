"""Fase 3: entrenamiento, comparación y registro del modelo final.

Optuna se ejecuta únicamente desde notebooks/optuna_optimization.ipynb.
El notebook importa de este módulo:
    cargar_datos_preprocesados, crear_modelo, registrar_experimento

Uso:
    python src/train_model.py                 # evaluar 4 modelos sobre Validation
    python src/train_model.py --finalizar     # reentrenar ganador Train+Val y guardarlo
    python src/train_model.py --sin-graficos  # omite todos los gráficos

No se usa Test aquí. Para la evaluación final usar evaluate_model.py --test.
"""

import argparse
import csv
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from evaluate_model import (
    baselines_validacion,
    calcular_metricas,
    evaluar_mejor_modelo,
    imprimir_diagnostico_errores,
)

RAIZ = Path(__file__).resolve().parent.parent
TARGET = "Target_Frio_Dia_Siguiente"
NOMBRES_MODELOS = ("XGBoost", "Random Forest", "LightGBM", "Ridge")
SEMILLA = 42


def resolver_ruta_csv(ruta_csv=None):
    """Resuelve la ruta habitual o la alternativa del notebook."""
    if ruta_csv is not None:
        ruta = Path(ruta_csv).expanduser().resolve()
        if not ruta.is_file():
            raise FileNotFoundError(f"No existe el CSV: {ruta}")
        return ruta
    opciones = (
        RAIZ / "data" / "processed" / "dataset_final.csv",
        RAIZ / "notebooks" / "data" / "processed" / "dataset_final.csv",
    )
    for ruta in opciones:
        if ruta.is_file():
            return ruta
    raise FileNotFoundError(
        "No encontré data/processed/dataset_final.csv. "
        "Ejecutá la última celda de preprocesamiento.ipynb o usá --data RUTA."
    )


def hash_archivo(ruta):
    sha = hashlib.sha256()
    with Path(ruta).open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b""):
            sha.update(bloque)
    return sha.hexdigest()


def cargar_datos_preprocesados(ruta_csv=None):
    """Devuelve X_train, y_train, X_val, y_val, X_test, y_test.

    Si existe columna 'particion', conserva exactamente la división del notebook.
    Si no, reconstruye un reparto cronológico 70/15/15. En ese último caso
    comprobar que coincide con la partición original del preprocesamiento.
    """
    ruta = resolver_ruta_csv(ruta_csv)
    df = pd.read_csv(ruta)
    if TARGET not in df:
        raise ValueError(f"El CSV no tiene la columna objetivo {TARGET!r}.")

    fecha = next(
        (col for col in ("DIA", "fecha", "Fecha", "Unnamed: 0") if col in df),
        None,
    )
    if fecha is None:
        raise ValueError("El CSV debe tener columna de fecha: DIA, fecha o Unnamed: 0.")
    fechas = pd.to_datetime(df.pop(fecha), errors="coerce")
    if fechas.isna().any() or fechas.duplicated().any():
        raise ValueError("Fechas inválidas o repetidas en dataset_final.csv.")
    df.index = pd.DatetimeIndex(fechas, name="fecha")
    df = df.sort_index()

    y = pd.to_numeric(df.pop(TARGET), errors="coerce")
    if y.isna().any() or not np.isfinite(y.to_numpy()).all():
        raise ValueError("Hay valores faltantes o infinitos en el target.")

    # 'particion' y las columnas sin nombre nunca deben ser features.
    etiquetas = df.pop("particion").astype(str).str.lower() if "particion" in df else None
    columnas_aux = [c for c in df.columns if c.startswith("Unnamed:")]
    X = df.drop(columns=columnas_aux).copy()
    for columna in X.columns:
        try:
            X[columna] = pd.to_numeric(X[columna], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Feature no numérica: {columna!r}.") from exc
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    if X.shape[1] == 0:
        raise ValueError("El dataset no contiene variables predictoras.")

    if etiquetas is not None:
        validas = {"train", "val", "test"}
        if set(etiquetas.unique()) != validas:
            raise ValueError("La columna 'particion' debe contener train, val y test.")
        X_train, y_train = X.loc[etiquetas == "train"], y.loc[etiquetas == "train"]
        X_val, y_val = X.loc[etiquetas == "val"], y.loc[etiquetas == "val"]
        X_test, y_test = X.loc[etiquetas == "test"], y.loc[etiquetas == "test"]
        if not (X_train.index.max() < X_val.index.min() < X_test.index.min()):
            raise ValueError("Las particiones no respetan el orden de las fechas.")
    else:
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.15, shuffle=False
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.15 / 0.85, shuffle=False
        )
        print("AVISO: el CSV no tiene 'particion'; se reconstruyó un split 70/15/15.")
        print("       Verificá que coincida con la división original del notebook.")

    print(f"Datos: {ruta}")
    print(f"Train {X_train.shape} | Validation {X_val.shape} | Test reservado {X_test.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test


def crear_modelo(nombre, params=None):
    """Pipeline de modelo compartido con Optuna; fit() ajusta solo con Train."""
    params = dict(params or {})
    if nombre == "XGBoost":
        estimador = XGBRegressor(
            **params, random_state=SEMILLA, n_jobs=1, verbosity=0
        )
    elif nombre == "Random Forest":
        estimador = RandomForestRegressor(
            **params, random_state=SEMILLA, n_jobs=1
        )
    elif nombre == "LightGBM":
        estimador = LGBMRegressor(
            **params, random_state=SEMILLA, n_jobs=1, verbosity=-1
        )
    elif nombre == "Ridge":
        estimador = Ridge(**params)
        return make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            estimador,
        )
    else:
        raise ValueError(f"Modelo no admitido: {nombre}")
    return make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        estimador,
    )


def registrar_experimento(nombre, modelo, metricas):
    """Conserva todos los experimentos de Optuna y del entrenamiento habitual."""
    ruta = RAIZ / "results" / "experiment_logs.csv"
    ruta.parent.mkdir(parents=True, exist_ok=True)
    existe = ruta.is_file() and ruta.stat().st_size > 0
    registro = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "modelo": nombre,
        "hiperparametros": json.dumps(modelo.get_params(deep=True), default=str),
        "MAE": float(metricas["MAE"]),
        "RMSE": float(metricas["RMSE"]),
        "R2": float(metricas["R2"]),
    }
    with ruta.open("a", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=registro.keys())
        if not existe:
            escritor.writeheader()
        escritor.writerow(registro)


def cargar_mejores_parametros(ruta=None, ruta_datos=None):
    """Lee best_params.json; advierte si cambió el CSV tras ejecutar Optuna."""
    ruta = Path(ruta).expanduser().resolve() if ruta else RAIZ / "models" / "best_params.json"
    if not ruta.is_file():
        print("AVISO: no existe best_params.json; se usarán hiperparámetros por defecto.")
        return {}
    with ruta.open(encoding="utf-8") as archivo:
        params = json.load(archivo)
    if not isinstance(params, dict):
        raise ValueError("best_params.json debe ser un diccionario por modelo.")
    for nombre, valores in params.items():
        if nombre not in NOMBRES_MODELOS or not isinstance(valores, dict):
            raise ValueError(f"Parámetros inválidos para {nombre!r}.")

    metadata = RAIZ / "results" / "optuna_run_metadata.json"
    if metadata.is_file() and ruta_datos is not None:
        with metadata.open(encoding="utf-8") as f:
            info = json.load(f)
        original = info.get("dataset_sha256")
        if original and original != hash_archivo(ruta_datos):
            print("ATENCIÓN: el dataset cambió desde la última ejecución de Optuna.")
            print("          Recomendado: volver a optimizar antes de finalizar.")
    print(f"Parámetros leídos de {ruta}")
    return params


def entrenar_modelos(X_train, y_train, X_val, y_val, parametros=None):
    """Entrena los cuatro algoritmos y los compara únicamente en Validation."""
    parametros = parametros or {}
    modelos, resultados = {}, {}
    for nombre in NOMBRES_MODELOS:
        modelo = crear_modelo(nombre, parametros.get(nombre, {}))
        modelo.fit(X_train, y_train)
        metricas = calcular_metricas(y_val, modelo.predict(X_val))
        registrar_experimento(nombre, modelo, metricas)
        modelos[nombre], resultados[nombre] = modelo, metricas
    tabla = pd.DataFrame(resultados).T.sort_values("MAE")
    return modelos, tabla


def guardar_modelo_final(nombre, parametros, X_train, y_train, X_val, y_val,
                         metricas_validacion, ruta_datos, version="v1.0.0"):
    """Reentrena Train+Validation y registra un Pipeline versionado (NO Test)."""
    if not re.fullmatch(r"v\d+\.\d+\.\d+", version):
        raise ValueError("Versión inválida; ejemplo esperado: v1.0.0")
    carpeta = RAIZ / "models"
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta_modelo = carpeta / f"modelo_{version}.pkl"
    registro = carpeta / "model_registry.json"
    if ruta_modelo.exists():
        raise FileExistsError(
            f"Ya existe {ruta_modelo}. Elegí una versión nueva, por ejemplo v1.0.1."
        )

    X_total = pd.concat([X_train, X_val]).sort_index()
    y_total = pd.concat([y_train, y_val]).sort_index()
    modelo = crear_modelo(nombre, parametros)
    modelo.fit(X_total, y_total)
    joblib.dump(modelo, ruta_modelo)

    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=RAIZ,
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = None

    if registro.exists():
        with registro.open(encoding="utf-8") as archivo:
            catalogo = json.load(archivo)
        if "models" not in catalogo or not isinstance(catalogo["models"], dict):
            raise ValueError("model_registry.json tiene un formato inesperado.")
    else:
        catalogo = {"latest": None, "models": {}}

    catalogo["models"][version] = {
        "version": version,
        "algoritmo": nombre,
        "path": f"models/{ruta_modelo.name}",
        "hiperparametros": parametros,
        "metricas_validacion": {k: float(v) for k, v in metricas_validacion.items()},
        "dataset_sha256": hash_archivo(ruta_datos),
        "columnas_entrada": list(X_total.columns),
        "target": TARGET,
        "git_commit": git_commit,
        "entrenado_utc": datetime.now(timezone.utc).isoformat(),
    }
    catalogo["latest"] = version
    with registro.open("w", encoding="utf-8") as archivo:
        json.dump(catalogo, archivo, indent=2, ensure_ascii=False)

    print(f"\nModelo final: {nombre} -> {ruta_modelo}")
    print(f"Registro actualizado: {registro}")
    print("Falta etiquetar en Git la misma versión al cerrar la entrega.")
    return modelo


def main():
    parser = argparse.ArgumentParser(description="Fase 3: entrenar, evaluar y guardar")
    parser.add_argument("--data", help="Ruta a dataset_final.csv")
    parser.add_argument("--params", help="Ruta a best_params.json")
    parser.add_argument("--finalizar", action="store_true",
                        help="Reentrena con Train+Val y guarda el ganador; sin tocar Test")
    parser.add_argument("--version", default="v1.0.0", help="Versión para --finalizar")
    parser.add_argument("--sin-graficos", action="store_true",
                        help="Omite los gráficos")
    args = parser.parse_args()

    ruta_datos = resolver_ruta_csv(args.data)
    X_train, y_train, X_val, y_val, X_test, _y_test = cargar_datos_preprocesados(ruta_datos)
    parametros = cargar_mejores_parametros(args.params, ruta_datos)
    modelos, tabla = entrenar_modelos(X_train, y_train, X_val, y_val, parametros)
    baselines = baselines_validacion(y_train, y_val)
    tabla_completa = pd.concat([tabla.assign(Fechas=len(y_val)), baselines])
    print("\nModelos y baselines en VALIDACIÓN")
    print(tabla_completa.to_string(float_format="{:.2f}".format, na_rep="—"))
    print("Persistencia usa la última observación disponible.")
    print("La media móvil promedia las observaciones disponibles de los 7 días previos.")
    print("Fechas indica cuántas predicciones se pudieron evaluar por método.")
    evaluar_mejor_modelo(modelos, tabla, X_val, y_val,
                         graficos=not args.sin_graficos)

    residuos = pd.DataFrame({
        nombre: y_val - modelos[nombre].predict(X_val)
        for nombre in tabla.index
    })
    errores_por_fecha = residuos.abs()
    mae_por_anio = errores_por_fecha.groupby(y_val.index.year.to_numpy()).mean().T
    print("\nMAE de validación por año de la fecha de entrada")
    print(mae_por_anio.to_string(float_format="{:.2f}".format))

    filas_por_anio = pd.DataFrame({
        "Train": X_train.index.year.value_counts(),
        "Validation": X_val.index.year.value_counts(),
        "Test": X_test.index.year.value_counts(),
    }).fillna(0).astype(int).sort_index()
    print("\nCantidad de filas por año")
    print(filas_por_anio.rename_axis("Año").to_string())

    imprimir_diagnostico_errores(y_val, residuos, errores_por_fecha)

    if args.finalizar:
        ganador = tabla.index[0]
        guardar_modelo_final(
            ganador, parametros.get(ganador, {}), X_train, y_train, X_val,
            y_val, tabla.loc[ganador].to_dict(), ruta_datos, args.version
        )
    else:
        print("\nAún no se guardó el modelo final. Usá --finalizar cuando esté validado.")


if __name__ == "__main__":
    main()
