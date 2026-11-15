import argparse
import os
import json
import hashlib
import datetime
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.feature_selection import f_regression

warnings.filterwarnings('ignore', category=UserWarning)

# Cantidad de variables que conserva SelectKBest (ANOVA F). La comparación con RFE y Lasso
# que justifica el método y el valor de k está en notebooks/seleccion_variables.ipynb.
K_VARIABLES = 10
COLUMNA_FRIO = 'Frio (Kw) (Consolidado EE)'
TARGET = 'Target_Frio_Dia_Siguiente'

# Cierres sin escalar de las primeras 270 fechas de Train (2020-07-01 a 2021-04-18),
# el tramo de ajuste del primer fold de seleccion_variables.ipynb.
Q1_FRIO_TRAIN, Q3_FRIO_TRAIN = 17_859.5, 28_371.25
FACTOR_IQR_FRIO = 5
RANGO_IQR_FRIO = Q3_FRIO_TRAIN - Q1_FRIO_TRAIN
LIMITE_INFERIOR_FRIO = max(0.0, Q1_FRIO_TRAIN - FACTOR_IQR_FRIO * RANGO_IQR_FRIO)
LIMITE_SUPERIOR_FRIO = Q3_FRIO_TRAIN + FACTOR_IQR_FRIO * RANGO_IQR_FRIO

# ==============================================================================
# 1. DEFINICIÓN DEL PIPELINE DE SCIKIT-LEARN (ESCALADO Y ASIMETRÍA)
# ==============================================================================
def construir_pipeline_escalado(X_train_df: pd.DataFrame) -> ColumnTransformer:
    """
    Construye un Pipeline de scikit-learn ajustado únicamente con datos de Train:
    - Aplica Yeo-Johnson + StandardScaler a las variables con |skewness| > 1.0 en Train.
    - Aplica StandardScaler al resto de las variables numéricas.
    """
    cols_numericas = X_train_df.select_dtypes(include=[np.number]).columns.tolist()

    # Evaluamos asimetría exclusivamente sobre X_train
    skew_vals = X_train_df[cols_numericas].skew()
    cols_sesgadas = skew_vals[abs(skew_vals) > 1.0].index.tolist()
    cols_normales = [c for c in cols_numericas if c not in cols_sesgadas]

    pipe_sesgadas = Pipeline(steps=[
        ('yeo_johnson', PowerTransformer(method='yeo-johnson', standardize=False)),
        ('scaler', StandardScaler())
    ])

    pipe_normales = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    transformers = []
    if len(cols_sesgadas) > 0:
        transformers.append(('sesgadas', pipe_sesgadas, cols_sesgadas))
    if len(cols_normales) > 0:
        transformers.append(('normales', pipe_normales, cols_normales))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='passthrough',
        verbose_feature_names_out=False
    )

    # Devuelve un DataFrame de Pandas conservando los nombres de columnas y el índice DIA
    preprocessor.set_output(transform="pandas")
    return preprocessor


# ==============================================================================
# 2. FUNCIONES MODULARES DE LIMPIEZA Y PREPROCESAMIENTO
# ==============================================================================
def cargar_y_limpiar_registros(ruta_csv_entrada: Path) -> pd.DataFrame:
    """Filtra días incompletos/excedidos, días sin reinicio a las 00:00 y recorta a las 23:59."""
    print(f"[1/6] Cargando dataset desde {ruta_csv_entrada}...")
    df_final = pd.read_csv(ruta_csv_entrada, parse_dates=["DIA"], low_memory=False)
    return preparar_cierres(df_final)


def preparar_cierres(df_final: pd.DataFrame, *, con_objetivo=True, umbral_reinicio=None) -> pd.DataFrame:
    """Aplica las mismas reglas horarias a datos históricos y a un Excel nuevo."""
    # Filtrar días que no tengan exactamente 25 registros
    conteo_horas = df_final.groupby('DIA').size()
    dias_incompletos = conteo_horas[conteo_horas < 25].index
    dias_imposibles = conteo_horas[conteo_horas > 25].index

    df_final = df_final[~df_final['DIA'].isin(dias_incompletos)].copy()
    df_final = df_final[~df_final['DIA'].isin(dias_imposibles)].copy()

    # Filtrar días sin reinicio en el totalizador de frío a las 00:00:00
    df_final['HORA'] = df_final['HORA'].astype(str).str.strip()
    df_medianoche = df_final[df_final['HORA'].str.startswith('00:00:00')].copy()
    frio_medianoche = pd.to_numeric(df_medianoche[COLUMNA_FRIO], errors='coerce')

    if umbral_reinicio is None:
        df_cierre = df_final[df_final['HORA'].str.startswith('23:59:00')].copy()
        s_cierre = pd.to_numeric(df_cierre[COLUMNA_FRIO], errors='coerce')
        fechas_ordenadas = np.sort(df_final['DIA'].unique())
        fecha_corte_train = fechas_ordenadas[int(len(fechas_ordenadas) * 0.70)]
        mediana_cierre_train = s_cierre[df_cierre['DIA'] <= fecha_corte_train].median()
        umbral_reinicio = float(0.20 * mediana_cierre_train)

    dias_sin_reinicio = df_medianoche.loc[frio_medianoche > umbral_reinicio, 'DIA'].unique()
    df_final_limpio = df_final[~df_final['DIA'].isin(dias_sin_reinicio)].copy()

    # Quedarnos con el cierre diario (23:59:00) y forzar grilla diaria continua ('D')
    df_final_limpio['DIA'] = pd.to_datetime(df_final_limpio['DIA']).dt.normalize()
    df_final_limpio = df_final_limpio.sort_values(['DIA', 'HORA']).reset_index(drop=True)

    df_diario = df_final_limpio[df_final_limpio['HORA'] == '23:59:00'].copy()
    df_diario = df_diario.drop(columns=['HORA'], errors='ignore').set_index('DIA').sort_index()
    df_diario = df_diario.asfreq('D')

    # Reemplazar errores de Excel por NaN y asegurar tipo numérico
    df_diario = df_diario.replace(
        ['#VALUE!', '#DIV/0!', '#REF!', '#N/A', '#NAME?', '#NULL!', '#NUM!', ''],
        np.nan
    ).apply(pd.to_numeric, errors='coerce')

    # Validar el cierre antes de usarlo como variable, rezago o valor real de D+1.
    frio = df_diario[COLUMNA_FRIO]
    en_rango = frio.between(LIMITE_INFERIOR_FRIO, LIMITE_SUPERIOR_FRIO)
    fuera_de_rango = frio.notna() & ~en_rango
    if fuera_de_rango.any():
        print(f"Se marcaron {fuera_de_rango.sum()} cierres de Frío fuera del rango de Train como faltantes.")
    df_diario[COLUMNA_FRIO] = frio.where(en_rango)

    if con_objetivo:
        dia_siguiente = df_diario.index + pd.Timedelta(days=1)
        df_diario[TARGET] = df_diario[COLUMNA_FRIO].reindex(dia_siguiente).to_numpy()
    df_diario.attrs['umbral_reinicio'] = umbral_reinicio
    return df_diario


def fechas_modelables(df_diario: pd.DataFrame) -> pd.DatetimeIndex:
    """Devuelve las fechas con frío observado hoy y al día siguiente."""
    return df_diario.dropna(subset=[COLUMNA_FRIO, TARGET]).index


def auditar_sensores_muertos(df_diario: pd.DataFrame, *, hasta=None) -> pd.DataFrame:
    """Elimina sensores con más de 95% de nulos en el tramo final del ajuste."""
    print("[2/6] Auditando sensores muertos en Train...")
    umbral_reinicio = df_diario.attrs.get('umbral_reinicio')
    dias_validos = df_diario.loc[df_diario[COLUMNA_FRIO].notna()]
    if hasta is None:
        df_train_audit = dias_validos.iloc[:int(len(dias_validos) * 0.70)]
    else:
        df_train_audit = dias_validos.loc[:hasta]

    corte_70 = int(len(df_train_audit) * 0.70)
    nulos_fin = df_train_audit.iloc[corte_70:].isnull().mean()
    sensores_muertos = nulos_fin.drop(labels=[COLUMNA_FRIO, TARGET]).loc[lambda s: s > 0.95].index
    print(f"      -> Sensores muertos eliminados: {len(sensores_muertos)}")
    resultado = df_diario.drop(columns=sensores_muertos)
    resultado.attrs['umbral_reinicio'] = umbral_reinicio
    return resultado


PARES_EFICIENCIA = [
    ('EE Planta / Hl (Consolidado KPI)', 'Meta EE Planta (Consolidado KPI)'),
    ('Agua Planta / Hl (Consolidado KPI)', 'Meta Agua Planta (Consolidado KPI)'),
    ('ET Planta / Hl (Consolidado KPI)', 'Meta ET Planta (Consolidado KPI)'),
    ('Aire Planta / Hl (Consolidado KPI)', 'Meta Aire Planta (Consolidado KPI)'),
    ('EE Elaboracion / Hl (Consolidado KPI)', 'Meta EE Elaboracion (Consolidado KPI)'),
    ('EE Bodega / Hl (Consolidado KPI)', 'Meta EE Bodega (Consolidado KPI)'),
    ('EE Cocina / Hl (Consolidado KPI)', 'Meta EE Cocina (Consolidado KPI)'),
    ('EE Envasado / Hl (Consolidado KPI)', 'Meta EE Envasado (Consolidado KPI)'),
    ('EE Frio / Hl (Consolidado KPI)', 'Meta EE Frio (Consolidado KPI)'),
    ('Agua Elab / Hl (Consolidado KPI)', 'Meta Agua Elab (Consolidado KPI)'),
    ('Agua Bodega / Hl (Consolidado KPI)', 'Meta Agua Bodega (Consolidado KPI)'),
    ('Agua Cocina / Hl (Consolidado KPI)', 'Meta Agua Cocina (Consolidado KPI)'),
    ('Agua Envas / Hl (Consolidado KPI)', 'Meta Agua Envas (Consolidado KPI)')
]

EFICIENCIAS = {
    f"Eficiencia_{real.split('/')[0].strip()}": (real, meta)
    for real, meta in PARES_EFICIENCIA
}
COLUMNA_EE_FRIO = 'EE Frio / Hl (Consolidado KPI)'
DERIVADAS_FRIO = {
    'Ratio_Frio_vs_ProdBodega': ('Hl Producido Bodega (Consolidado Produccion)', 'ratio'),
    'Ratio_Frio_vs_Mosto': ('Hl de Mosto (Consolidado Produccion)', 'ratio'),
    'Interaccion_Frio_CervezaFiltrada': ('Hl Cerveza Filtrada (Consolidado Produccion)', 'producto'),
    'Interaccion_Frio_Cocimientos': ('Cocimientos Diarios (Consolidado Produccion)', 'producto'),
}
CALENDARIO = ('Anio', 'Mes', 'Dia_Semana', 'Es_Fin_De_Semana')
COLUMNAS_CLIMA = ('Temp_Media_Tecate', 'Temp_Max_Tecate', 'Temp_Min_Tecate')
SUFIJOS_TEMPORALES = ('_Lag_1', '_Lag_7', '_MediaMov_7d')


def crear_features(df_diario: pd.DataFrame, *, clima=None, con_objetivo=True) -> pd.DataFrame:
    """Genera variables de calendario, lags, medias móviles, ratios, interacciones y clima."""
    print("[3/6] Ejecutando Feature Engineering...")
    umbral_reinicio = df_diario.attrs.get('umbral_reinicio')

    df_diario = df_diario.copy()
    if not isinstance(df_diario.index, pd.DatetimeIndex):
        df_diario.index = pd.to_datetime(df_diario.index)

    # Calendario
    df_diario['Anio'] = df_diario.index.year
    df_diario['Mes'] = df_diario.index.month
    df_diario['Dia_Semana'] = df_diario.index.dayofweek
    df_diario['Es_Fin_De_Semana'] = df_diario['Dia_Semana'].isin([5, 6]).astype(int)

    # Lags y Medias Móviles
    cols_para_features = df_diario.select_dtypes(include=[np.number]).columns.drop(
        [TARGET, *CALENDARIO], errors='ignore'
    )

    nuevas_cols_lag = {}
    for col in cols_para_features:
        nuevas_cols_lag[f'{col}_Lag_1'] = df_diario[col].shift(1)
        nuevas_cols_lag[f'{col}_Lag_7'] = df_diario[col].shift(7)
        nuevas_cols_lag[f'{col}_MediaMov_7d'] = (
            df_diario[col].rolling(window=7, min_periods=1, closed='left').mean()
        )

    df_diario = pd.concat([df_diario, pd.DataFrame(nuevas_cols_lag, index=df_diario.index)], axis=1)
    columnas_validas = [COLUMNA_FRIO, TARGET] if con_objetivo else [COLUMNA_FRIO]
    df_diario = df_diario.dropna(subset=columnas_validas).copy()
    if df_diario.empty:
        raise ValueError("No hay días válidos con cierre de Frío a las 23:59.")

    epsilon = 1e-5
    for nombre, (real, meta) in EFICIENCIAS.items():
        if real in df_diario and meta in df_diario:
            df_diario[nombre] = df_diario[real] / (df_diario[meta] + epsilon)
    for nombre, (otra, operacion) in DERIVADAS_FRIO.items():
        if COLUMNA_EE_FRIO in df_diario and otra in df_diario:
            frio = df_diario[COLUMNA_EE_FRIO]
            df_diario[nombre] = frio / (df_diario[otra] + epsilon) if operacion == 'ratio' else frio * df_diario[otra]

    if clima is None:
        clima = cargar_clima(df_diario.index)
    if not clima.empty:
        df_diario = df_diario.join(clima, how='left')
        df_diario[clima.columns] = df_diario[clima.columns].ffill()
    df_diario.attrs['umbral_reinicio'] = umbral_reinicio
    return df_diario


def cargar_clima(fechas: pd.DatetimeIndex) -> pd.DataFrame:
    """Obtiene el clima de Tecate para el intervalo de fechas solicitado."""
    lat, lon = 32.5714, -116.6275
    fecha_inicio = fechas.min().strftime('%Y-%m-%d')
    fecha_fin = fechas.max().strftime('%Y-%m-%d')
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
        f"&start_date={fecha_inicio}&end_date={fecha_fin}"
        f"&daily=temperature_2m_mean,temperature_2m_max,temperature_2m_min&timezone=America/Los_Angeles"
    )
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()
        if "daily" in data:
            return pd.DataFrame({
                'Fecha': pd.to_datetime(data['daily']['time']),
                'Temp_Media_Tecate': data['daily']['temperature_2m_mean'],
                'Temp_Max_Tecate': data['daily']['temperature_2m_max'],
                'Temp_Min_Tecate': data['daily']['temperature_2m_min']
            }).set_index('Fecha')
    except Exception as exc:
        raise RuntimeError(
            "No se pudieron obtener los datos de Open-Meteo. "
            "El pipeline se detiene para no generar un dataset diferente sin avisar."
        ) from exc

    return pd.DataFrame()


def ajustar_limites_outliers(X_train):
    """Aprende los límites de Z modificada y winsorización."""
    limites = {}
    for columna, valores in X_train.items():
        mediana = valores.median()
        mad = np.median(np.abs(valores - mediana))
        if mad > 0:
            amplitud = 3.5 * mad / 0.6745
            valores = valores.clip(lower=mediana - amplitud, upper=mediana + amplitud)

        # Los cuantiles ya están dentro de los límites de Z modificada.
        limites[columna] = (valores.quantile(0.01), valores.quantile(0.99))
    return limites


def aplicar_limites_outliers(datos, limites):
    for columna in datos.columns:
        inferior, superior = limites[columna]
        datos[columna] = datos[columna].clip(lower=inferior, upper=superior)
    return datos


def dividir_temporalmente(datos):
    """Divide datos ordenados cronológicamente en Train, Validation y Test (70/15/15)."""
    desarrollo, test = train_test_split(datos, test_size=0.15, shuffle=False)
    train, val = train_test_split(desarrollo, test_size=0.15 / 0.85, shuffle=False)
    return train, val, test


def ajustar_limpieza(df_train):
    """Ajusta los filtros, imputadores y límites con las filas de entrenamiento."""
    # 1. Filtro de Redundancia (|r| >= 0.90) en Train
    X_features_train = df_train.drop(columns=[TARGET]).select_dtypes(include=[np.number])
    matriz_corr = X_features_train.corr().abs()
    upper_tri = matriz_corr.where(np.triu(np.ones(matriz_corr.shape), k=1).astype(bool))

    cols_a_eliminar_red = set()
    for col in upper_tri.columns:
        correlacionadas = upper_tri.index[upper_tri[col] >= 0.90].tolist()
        if correlacionadas:
            grupo = correlacionadas + [col]
            mejor_var = df_train[grupo].corrwith(df_train[TARGET]).abs().idxmax()
            for c in grupo:
                if c != mejor_var:
                    cols_a_eliminar_red.add(c)

    # 2. Filtro de P-Value (p < 0.05) en Train
    df_train_sin_red = df_train.drop(columns=list(cols_a_eliminar_red))
    X_final_candidatas = df_train_sin_red.drop(columns=[TARGET]).select_dtypes(include=[np.number])
    variables_significativas = []

    for col in X_final_candidatas.columns:
        datos_validos = df_train_sin_red[[col, TARGET]].dropna()
        if len(datos_validos) > 2 and datos_validos[col].nunique() > 1:
            _, p_value = pearsonr(datos_validos[col], datos_validos[TARGET])
            if p_value < 0.05:
                variables_significativas.append(col)

    # 3. Eliminar columnas auxiliares y patrones administrativos.
    patrones_a_eliminar = ['Meta', 'Id ', 'Auxiliar', 'Ultimo Dato']
    columnas = [
        col for col in variables_significativas
        if 'unnamed' not in col.lower()
        and (col in ['DIA', 'HORA'] or not any(p.lower() in col.lower() for p in patrones_a_eliminar))
    ]
    X_train = df_train[columnas].copy()

    # 4. Proteger los sensores incorporados tarde durante el tramo de ajuste.
    inicio_corte = int(len(X_train) * 0.30)
    fin_corte = int(len(X_train) * 0.70)
    sensores_protegidos = (
        (X_train.iloc[:inicio_corte].isna().mean() > 0.50)
        & (X_train.iloc[fin_corte:].isna().mean() < 0.20)
    )

    # 5. Elegir la imputación a partir del porcentaje de nulos en Train.
    porcentaje_nulos = X_train.loc[:, ~sensores_protegidos].isna().mean() * 100
    cols_media = porcentaje_nulos[(porcentaje_nulos > 0) & (porcentaje_nulos < 1)].index
    cols_knn = porcentaje_nulos[porcentaje_nulos.between(1, 45)].index
    cols_descartadas = porcentaje_nulos[porcentaje_nulos > 45].index
    columnas_finales = X_train.columns.drop(cols_descartadas)
    # Mantener el orden de salida del pipeline original.
    protegidas = sensores_protegidos[sensores_protegidos].index
    columnas_finales = columnas_finales.drop(protegidas).append(protegidas)
    X_train = X_train[columnas_finales].copy()
    imputadores = []
    for cols, imputador in (
        (cols_media, SimpleImputer(strategy='mean')),
        (cols_knn, KNNImputer(n_neighbors=5)),
    ):
        if len(cols):
            X_train[cols] = imputador.fit_transform(X_train[cols])
            imputadores.append((cols.tolist(), imputador))

    limites = ajustar_limites_outliers(X_train)
    estado = {
        'columnas': columnas_finales.tolist(),
        'imputadores': imputadores,
        'limites': limites,
        'umbral_reinicio': df_train.attrs.get('umbral_reinicio'),
    }
    return aplicar_limites_outliers(X_train, limites), estado


def columnas_para_transformar(estado, columnas):
    """Incluye las variables auxiliares que determinan las distancias de KNN."""
    necesarias = list(columnas)
    for grupo, imputador in estado['imputadores']:
        if isinstance(imputador, KNNImputer) and set(columnas).intersection(grupo):
            necesarias.extend(grupo)
    return list(dict.fromkeys(necesarias))


def transformar_limpieza(df, estado, columnas=None):
    """Aplica la limpieza aprendida y devuelve las columnas en el orden solicitado."""
    columnas = estado['columnas'] if columnas is None else list(columnas)
    X = df[columnas].copy()
    for grupo, imputador in estado['imputadores']:
        elegidas = [col for col in grupo if col in columnas]
        if elegidas:
            # KNN calcula la distancia con las mediciones disponibles de su grupo.
            valores = imputador.transform(df.reindex(columns=grupo))
            X[elegidas] = pd.DataFrame(valores, index=X.index, columns=grupo)[elegidas]

    return aplicar_limites_outliers(X, estado['limites'])


def columnas_de_origen(columnas):
    """Resuelve las columnas de Excel a partir de las variables del modelo."""
    origen = {COLUMNA_FRIO}
    for columna in columnas:
        for sufijo in SUFIJOS_TEMPORALES:
            if columna.endswith(sufijo):
                columna = columna.removesuffix(sufijo)
                break
        if columna in EFICIENCIAS:
            origen.update(EFICIENCIAS[columna])
        elif columna in DERIVADAS_FRIO:
            origen.update((COLUMNA_EE_FRIO, DERIVADAS_FRIO[columna][0]))
        elif columna not in CALENDARIO and columna not in COLUMNAS_CLIMA:
            origen.add(columna)
    return sorted(origen)


def cargar_excel(ruta, columnas_requeridas, columnas_auxiliares=()):
    """Unifica las hojas por nombre y alinea los auxiliares con el histórico requerido."""
    requeridas = set(columnas_requeridas)
    hojas = {}
    for columna in sorted(requeridas.union(columnas_auxiliares)):
        nombre, separador, hoja = columna.rpartition(' (')
        if not separador or not hoja.endswith(')'):
            raise ValueError(f'No se puede identificar la hoja de la variable {columna!r}.')
        hojas.setdefault(hoja[:-1], {})[nombre] = columna in requeridas

    tablas_requeridas, tablas_auxiliares = [], []
    faltantes = []
    claves = ['DIA', 'HORA']
    with pd.ExcelFile(ruta, engine='openpyxl') as libro:
        for hoja, mediciones in hojas.items():
            obligatorias = [col for col, requerida in mediciones.items() if requerida]
            if hoja not in libro.sheet_names:
                if obligatorias:
                    faltantes.append(f'Hoja {hoja!r}')
                continue
            tabla = libro.parse(sheet_name=hoja, usecols=lambda col: col in claves or col in mediciones)
            ausentes = [col for col in claves + obligatorias if col not in tabla]
            if ausentes:
                if obligatorias:
                    faltantes.append(f'Hoja {hoja!r}, columnas: {", ".join(ausentes)}')
                continue
            nombres = [col for col in mediciones if col in tabla]
            if not nombres:
                continue

            tabla['DIA'] = pd.to_datetime(tabla['DIA'], errors='coerce', format='mixed').dt.normalize()
            horas = tabla['HORA'].astype('string').str.strip().str.extract(
                r'^([01]?\d|2[0-3]):([0-5]\d)(?::([0-5]\d))?$', expand=True,
            )
            tabla['HORA'] = horas[0].str.zfill(2) + ':' + horas[1] + ':' + horas[2].fillna('00')
            cierres = tabla['HORA'].str.startswith('23:59:', na=False)
            tabla.loc[cierres, 'HORA'] = '23:59:00'
            tabla[nombres] = tabla[nombres].apply(pd.to_numeric, errors='coerce')
            registros = tabla.dropna(subset=claves).groupby(claves, sort=False)[nombres]
            unificada = registros.last()
            cierre = unificada.index.get_level_values('HORA') == '23:59:00'
            if cierre.any():
                unificada.loc[cierre] = unificada.loc[cierre].mask(registros.nunique().loc[cierre] > 1)
            unificada.columns = [f'{col} ({hoja})' for col in nombres]
            tablas = tablas_requeridas if obligatorias else tablas_auxiliares
            tablas.append(unificada)

    if faltantes:
        raise ValueError('Faltan datos requeridos en el Excel:\n- ' + '\n- '.join(faltantes))
    registros = pd.concat(tablas_requeridas, axis=1)
    if tablas_auxiliares:
        registros = registros.join(pd.concat(tablas_auxiliares, axis=1))
    return registros.sort_index().reset_index()


def preparar_excel_prediccion(ruta, preprocesador):
    """Transforma un Excel nuevo con el estado ajustado durante Train."""
    variables = preprocesador['variables']
    estado = preprocesador['limpieza']
    umbral_reinicio = preprocesador.get('umbral_reinicio')
    variables_con_auxiliares = columnas_para_transformar(estado, variables)
    registros = cargar_excel(ruta, columnas_de_origen(variables), columnas_de_origen(variables_con_auxiliares))
    diario = preparar_cierres(registros, con_objetivo=False, umbral_reinicio=umbral_reinicio)
    usa_clima = any(col in COLUMNAS_CLIMA for col in variables_con_auxiliares)
    features = crear_features(diario, clima=None if usa_clima else pd.DataFrame(), con_objetivo=False)
    faltantes = sorted(set(variables) - set(features.columns))
    if faltantes:
        raise ValueError(f'No se pudieron calcular las variables: {", ".join(faltantes)}.')
    X = transformar_limpieza(features, estado, variables).replace([np.inf, -np.inf], np.nan)
    return X, X.index


def filtrar_imputar_y_limpiar(df_train, *evaluaciones):
    """Ajusta la limpieza en el tramo de entrenamiento de cada fold."""
    X_train, estado = ajustar_limpieza(df_train)
    return X_train, *(transformar_limpieza(df, estado) for df in evaluaciones)


def filtrar_imputar_y_dividir(df_diario: pd.DataFrame, *, guardar_estado=False):
    """Divide el dataset y ajusta la limpieza con Train completo."""
    print("[4/6] Filtrando, imputando y tratando outliers (Train 70%, Validation y Test 15%).")
    train, val, test = dividir_temporalmente(df_diario)
    train.attrs['umbral_reinicio'] = df_diario.attrs.get('umbral_reinicio')
    X_train, estado = ajustar_limpieza(train)
    X_val = transformar_limpieza(val, estado)
    X_test = transformar_limpieza(test, estado)
    datos = X_train, X_val, X_test, train[TARGET], val[TARGET], test[TARGET]
    return (*datos, estado) if guardar_estado else datos


def seleccionar_variables_kbest(X_train_scaled: pd.DataFrame, y_train: pd.Series, k: int = K_VARIABLES):
    """Ordena las variables por el estadístico F del ANOVA (f_regression) calculado solo con Train.

    Antes del test se imputan los faltantes de sensores tardíos con la mediana de Train y se acotan
    los valores escalados a ±8 desvíos, igual que en la comparación de notebooks/seleccion_variables.ipynb.
    Devuelve las k mejores variables (de mayor a menor F) y la tabla completa de puntajes.
    """
    X_sel = X_train_scaled.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    X_sel = X_sel.fillna(X_sel.median()).fillna(0).clip(-8, 8)
    puntaje_f, p_valor = f_regression(X_sel, y_train.loc[X_sel.index])
    puntajes = (
        pd.DataFrame({'F': np.nan_to_num(puntaje_f), 'p_valor': p_valor}, index=X_sel.columns)
        .sort_values('F', ascending=False)
    )
    return puntajes.index[:k].tolist(), puntajes


def escalar_y_seleccionar_variables(X_train_limpio, X_val_limpio, X_test_limpio, y_train):
    """Ajusta el Pipeline de scikit-learn en Train y selecciona variables con SelectKBest (ANOVA F)."""
    print("[5/6] Ajustando Pipeline de scikit-learn (Yeo-Johnson + StandardScaler) y seleccionando variables...")

    # Ajustamos el Pipeline de scikit-learn ÚNICAMENTE con los datos de entrenamiento
    pipeline_escalado = construir_pipeline_escalado(X_train_limpio)
    X_train_scaled = pipeline_escalado.fit_transform(X_train_limpio)
    X_val_scaled   = pipeline_escalado.transform(X_val_limpio)
    X_test_scaled  = pipeline_escalado.transform(X_test_limpio)

    # SelectKBest con ANOVA F: elegido frente a RFE y Lasso por menor MAE en la validación
    # cruzada temporal y mayor estabilidad (ver notebooks/seleccion_variables.ipynb)
    features_finales, puntajes = seleccionar_variables_kbest(X_train_scaled, y_train)

    X_train_final = X_train_scaled[features_finales]
    X_val_final   = X_val_scaled[features_finales]
    X_test_final  = X_test_scaled[features_finales]

    print(f"      -> Candidatas: {len(puntajes)} | Seleccionadas por SelectKBest (ANOVA F): {len(features_finales)}")
    for posicion, variable in enumerate(features_finales, start=1):
        print(f"         {posicion:2d}. {variable} (F = {puntajes.loc[variable, 'F']:.0f})")
    print(f"      -> Shape final X_train: {X_train_final.shape}")
    print(f"      -> Shape final X_val:   {X_val_final.shape}")
    print(f"      -> Shape final X_test:  {X_test_final.shape}")

    return features_finales, X_train_final, X_val_final, X_test_final, pipeline_escalado


def guardar_artefactos(
    X_train_limpio, X_val_limpio, X_test_limpio,
    y_train, y_val, y_test, features_finales, dir_raiz: Path, *, estado
):
    """Guarda dataset_final.csv (SIN ESCALAR), checksums.json y data_lineage.json."""
    print("[6/6] Exportando artefactos procesados y registros de versionado...")
    processed_dir = dir_raiz / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Unir particiones depuradas pero SIN ESCALAR (como exige el punto 2.4)
    train_sin_escalar = X_train_limpio[features_finales].copy()
    train_sin_escalar['Target_Frio_Dia_Siguiente'] = y_train
    train_sin_escalar['particion'] = 'train'

    val_sin_escalar = X_val_limpio[features_finales].copy()
    val_sin_escalar['Target_Frio_Dia_Siguiente'] = y_val
    val_sin_escalar['particion'] = 'val'

    test_sin_escalar = X_test_limpio[features_finales].copy()
    test_sin_escalar['Target_Frio_Dia_Siguiente'] = y_test
    test_sin_escalar['particion'] = 'test'

    df_dataset_final = pd.concat([train_sin_escalar, val_sin_escalar, test_sin_escalar]).sort_index()

    ruta_csv = processed_dir / "dataset_final.csv"
    df_dataset_final.to_csv(ruta_csv, index=True, lineterminator='\n')
    print(f"      -> Guardado: {ruta_csv} (Shape: {df_dataset_final.shape})")

    # Calcular SHA-256 y actualizar checksums.json
    sha256_hash = hashlib.sha256()
    with open(ruta_csv, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    checksum_val = sha256_hash.hexdigest()
    variables = set(features_finales)
    joblib.dump({
        'dataset_sha256': checksum_val,
        'variables': list(features_finales),
        'umbral_reinicio': estado.get('umbral_reinicio'),
        'limpieza': {
            'columnas': list(features_finales),
            'imputadores': [(grupo, imputador) for grupo, imputador in estado['imputadores']
                            if variables.intersection(grupo)],
            'limites': {col: estado['limites'][col] for col in features_finales},
        },
    }, processed_dir / 'preprocesador.joblib')

    ruta_checksums = dir_raiz / "data" / "checksums.json"
    checksums_data = {}
    if ruta_checksums.exists():
        try:
            with open(ruta_checksums, 'r', encoding='utf-8') as f:
                checksums_data = json.load(f)
        except Exception:
            checksums_data = {}

    checksums_data["data/processed/dataset_final.csv"] = checksum_val
    with open(ruta_checksums, 'w', encoding='utf-8') as f:
        json.dump(checksums_data, f, indent=4, ensure_ascii=False)
    print(f"      -> Checksum SHA-256 actualizado en {ruta_checksums}")

    # Generar data_lineage.json
    linaje_datos = {
        "proyecto": "Modelo Predictivo de Refrigeración - Planta Cerveza Tecate",
        "fecha_generacion": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "script_origen": "src/preprocessing_pipeline.py",
        "transformaciones_aplicadas": [
            "Filtrado de días incompletos (<25 registros), excedidos (>25) y sin reinicio a las 00:00 (umbral adaptativo: 20% de la mediana de cierre en Train)",
            "Agregación temporal de datos a nivel diario (cierre 23:59) con frecuencia continua ('D')",
            f"Cierres de Frío fuera de [{LIMITE_INFERIOR_FRIO:g}, {LIMITE_SUPERIOR_FRIO:g}] kW tratados como faltantes "
            f"(cuartiles del primer tramo de ajuste temporal de Train, con margen de {FACTOR_IQR_FRIO} IQR y piso en cero)",
            "Target de Frío alineado con el cierre válido del día calendario D + 1",
            "Auditoría y eliminación de sensores muertos en Train (>95% nulos al final)",
            "Extracción de variables de calendario (Anio, Mes, Dia_Semana, Es_Fin_De_Semana)",
            "Generación de rezagos (Lag_1, Lag_7) y medias móviles de 7 días (closed='left')",
            "Creación de indicadores de eficiencia energética (Real vs Meta) y ratios/interacciones de frío",
            "Integración de temperatura ambiente (media, max, min) de Tecate vía API Open-Meteo",
            "Filtro anti-leakage de redundancia (|r| >= 0.90) y significancia estadística (p < 0.05) en Train",
            "División temporal cronológica: Train (70%), Validation (15%), Test (15%)",
            "Imputación MCAR (SimpleImputer media) y MAR (KNNImputer) protegiendo sensores tardíos",
            "Tratamiento de outliers con Z-Modificada (MAD) y Winsorización (1%-99%) ajustado en Train",
            "Transformación Yeo-Johnson y Estandarización dentro de un Pipeline de scikit-learn ajustado en Train",
            f"Selección de características con SelectKBest (ANOVA F, f_regression) ajustado en Train: k = {len(features_finales)}",
            "Método elegido frente a RFE (Random Forest) y LASSO por MAE en TimeSeriesSplit, Validation y estabilidad "
            "(notebooks/seleccion_variables.ipynb)"
        ],
        "shapes_particiones": {
            "X_train": list(X_train_limpio[features_finales].shape),
            "X_val": list(X_val_limpio[features_finales].shape),
            "X_test": list(X_test_limpio[features_finales].shape)
        },
        "dimensiones_dataset_final": {
            "filas": int(df_dataset_final.shape[0]),
            "columnas_totales": int(df_dataset_final.shape[1]),
            "features_seleccionadas": len(features_finales)
        },
        "variables_seleccionadas": list(features_finales)
    }

    ruta_linaje = processed_dir / "data_lineage.json"
    with open(ruta_linaje, 'w', encoding='utf-8') as f:
        json.dump(linaje_datos, f, indent=4, ensure_ascii=False)
    print(f"      -> Linaje de datos guardado en {ruta_linaje}")


def ejecutar_pipeline_completo(ruta_entrada=None):
    """Punto de entrada principal para correr todo el pipeline desde consola."""
    # Detecta automáticamente la raíz del repositorio (ya sea que se corra desde raíz o desde src/)
    dir_actual = Path(__file__).resolve().parent
    dir_raiz = dir_actual.parent if dir_actual.name == "src" else dir_actual

    ruta_entrada = (
        Path(ruta_entrada).expanduser().resolve()
        if ruta_entrada
        else dir_raiz / "data" / "dataset_unificado_inicial.csv"
    )
    if not ruta_entrada.is_file():
        raise FileNotFoundError(
            f"Falta {ruta_entrada}. Ejecutá el EDA hasta el paso 5.1 "
            "para generar data/dataset_unificado_inicial.csv."
        )

    df_diario = cargar_y_limpiar_registros(ruta_entrada)
    df_diario = auditar_sensores_muertos(df_diario)
    df_diario = crear_features(df_diario)

    datos = filtrar_imputar_y_dividir(df_diario, guardar_estado=True)
    X_train_limpio, X_val_limpio, X_test_limpio, y_train, y_val, y_test, estado = datos

    features_finales, X_train_final, X_val_final, X_test_final, _ = escalar_y_seleccionar_variables(
        X_train_limpio, X_val_limpio, X_test_limpio, y_train
    )

    guardar_artefactos(
        X_train_limpio, X_val_limpio, X_test_limpio,
        y_train, y_val, y_test, features_finales, dir_raiz, estado=estado
    )
    print("\n¡Pipeline de preprocesamiento ejecutado de punta a punta con éxito!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline reproducible de preprocesamiento")
    parser.add_argument(
        "--data",
        help="Ruta opcional al CSV unificado generado por el EDA",
    )
    args = parser.parse_args()
    ejecutar_pipeline_completo(args.data)