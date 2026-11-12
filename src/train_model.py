
import pandas as pd


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
