import argparse
import os
import json
import hashlib
import datetime
import warnings
from pathlib import Path

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

    # Filtrar días que no tengan exactamente 25 registros
    conteo_horas = df_final.groupby('DIA').size()
    dias_incompletos = conteo_horas[conteo_horas < 25].index
    dias_imposibles = conteo_horas[conteo_horas > 25].index

    df_final = df_final[~df_final['DIA'].isin(dias_incompletos)].copy()
    df_final = df_final[~df_final['DIA'].isin(dias_imposibles)].copy()

    # Filtrar días sin reinicio en el totalizador de frío a las 00:00:00 (umbral = 100 kW)
    df_final['HORA'] = df_final['HORA'].astype(str).str.strip()
    df_medianoche = df_final[df_final['HORA'].str.startswith('00:00:00')].copy()
    col_frio = [c for c in df_final.columns if 'Frio' in c and 'Kw' in c][0]

    umbral_reinicio = 100
    dias_sin_reinicio = df_medianoche[df_medianoche[col_frio] > umbral_reinicio]['DIA'].unique()
    df_final_limpio = df_final[~df_final['DIA'].isin(dias_sin_reinicio)].copy()

    # Quedarnos con el cierre diario (23:59:00) y forzar grilla diaria continua ('D')
    df_final_limpio['DIA'] = pd.to_datetime(df_final_limpio['DIA']).dt.normalize()
    df_final_limpio = df_final_limpio.sort_values(['DIA', 'HORA']).reset_index(drop=True)

    df_diario = df_final_limpio[df_final_limpio['HORA'] == '23:59:00'].copy()
    df_diario = df_diario.drop(columns=['HORA'], errors='ignore').set_index('DIA').sort_index()
    df_diario = df_diario.asfreq('D')

    # Crear variable objetivo usando exactamente el día calendario siguiente.
    col_objetivo = 'Frio (Kw) (Consolidado EE)'
    dia_siguiente = df_diario.index + pd.Timedelta(days=1)
    df_diario['Target_Frio_Dia_Siguiente'] = (
        df_diario[col_objetivo].reindex(dia_siguiente).to_numpy()
    )

    # Reemplazar errores de Excel por NaN y asegurar tipo numérico
    df_diario = df_diario.replace(
        ['#VALUE!', '#DIV/0!', '#REF!', '#N/A', '#NAME?', '#NULL!', '#NUM!', ''],
        np.nan
    )
    for col in df_diario.columns:
        df_diario[col] = pd.to_numeric(df_diario[col], errors='coerce')

    return df_diario


def auditar_sensores_muertos(df_diario: pd.DataFrame) -> pd.DataFrame:
    """Detecta y elimina sensores muertos evaluando exclusivamente el 70% inicial (Train)."""
    print("[2/6] Auditando sensores muertos en Train...")
    df_temp = df_diario.copy()
    dias_validos_mask = df_temp['Frio (Kw) (Consolidado EE)'].notna()
    df_temp_validos = df_temp[dias_validos_mask]

    n_train_validos = int(len(df_temp_validos) * 0.70)
    df_train_audit = df_temp_validos.iloc[:n_train_validos]

    n = len(df_train_audit)
    corte_70 = int(n * 0.70)
    nulos_fin = df_train_audit.iloc[corte_70:].isnull().mean()

    mask_muertos = nulos_fin > 0.95
    vector_sensores_muertos = df_train_audit.columns[mask_muertos].tolist()

    df_diario = df_temp.drop(columns=vector_sensores_muertos)
    print(f"      -> Sensores muertos eliminados: {len(vector_sensores_muertos)}")
    return df_diario


def crear_features(df_diario: pd.DataFrame) -> pd.DataFrame:
    """Genera variables de calendario, lags, medias móviles, ratios, interacciones y clima."""
    print("[3/6] Ejecutando Feature Engineering...")
    target = 'Target_Frio_Dia_Siguiente'
    col_objetivo = 'Frio (Kw) (Consolidado EE)'

    if not isinstance(df_diario.index, pd.DatetimeIndex):
        df_diario.index = pd.to_datetime(df_diario.index)

    # Calendario
    df_diario['Anio'] = df_diario.index.year
    df_diario['Mes'] = df_diario.index.month
    df_diario['Dia_Semana'] = df_diario.index.dayofweek
    df_diario['Es_Fin_De_Semana'] = df_diario['Dia_Semana'].isin([5, 6]).astype(int)

    # Lags y Medias Móviles
    cols_para_features = df_diario.select_dtypes(include=[np.number]).columns.drop(
        [target, 'Anio', 'Mes', 'Dia_Semana', 'Es_Fin_De_Semana'], errors='ignore'
    )

    nuevas_cols_lag = {}
    for col in cols_para_features:
        nuevas_cols_lag[f'{col}_Lag_1'] = df_diario[col].shift(1)
        nuevas_cols_lag[f'{col}_Lag_7'] = df_diario[col].shift(7)
        nuevas_cols_lag[f'{col}_MediaMov_7d'] = (
            df_diario[col].rolling(window=7, min_periods=1, closed='left').mean()
        )

    df_diario = pd.concat([df_diario, pd.DataFrame(nuevas_cols_lag, index=df_diario.index)], axis=1)
    df_diario = df_diario.dropna(subset=[col_objetivo, target]).copy()

    # Eficiencia energética (Real vs Meta)
    epsilon = 1e-5
    pares_eficiencia = [
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
    for real_col, meta_col in pares_eficiencia:
        if real_col in df_diario.columns and meta_col in df_diario.columns:
            nombre_eficiencia = f"Eficiencia_{real_col.split('/')[0].strip()}"
            df_diario[nombre_eficiencia] = df_diario[real_col] / (df_diario[meta_col] + epsilon)

    # Ratios e Interacciones térmicas de frío
    if 'EE Frio / Hl (Consolidado KPI)' in df_diario.columns:
        if 'Hl Producido Bodega (Consolidado Produccion)' in df_diario.columns:
            df_diario['Ratio_Frio_vs_ProdBodega'] = (
                df_diario['EE Frio / Hl (Consolidado KPI)'] /
                (df_diario['Hl Producido Bodega (Consolidado Produccion)'] + epsilon)
            )
        if 'Hl de Mosto (Consolidado Produccion)' in df_diario.columns:
            df_diario['Ratio_Frio_vs_Mosto'] = (
                df_diario['EE Frio / Hl (Consolidado KPI)'] /
                (df_diario['Hl de Mosto (Consolidado Produccion)'] + epsilon)
            )
        if 'Hl Cerveza Filtrada (Consolidado Produccion)' in df_diario.columns:
            df_diario['Interaccion_Frio_CervezaFiltrada'] = (
                df_diario['EE Frio / Hl (Consolidado KPI)'] *
                df_diario['Hl Cerveza Filtrada (Consolidado Produccion)']
            )
        if 'Cocimientos Diarios (Consolidado Produccion)' in df_diario.columns:
            df_diario['Interaccion_Frio_Cocimientos'] = (
                df_diario['EE Frio / Hl (Consolidado KPI)'] *
                df_diario['Cocimientos Diarios (Consolidado Produccion)']
            )

    # Integración climática de Tecate vía API Open-Meteo
    lat, lon = 32.5714, -116.6275
    fecha_inicio = df_diario.index.min().strftime('%Y-%m-%d')
    fecha_fin = df_diario.index.max().strftime('%Y-%m-%d')
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
            df_clima = pd.DataFrame({
                'Fecha': pd.to_datetime(data['daily']['time']),
                'Temp_Media_Tecate': data['daily']['temperature_2m_mean'],
                'Temp_Max_Tecate': data['daily']['temperature_2m_max'],
                'Temp_Min_Tecate': data['daily']['temperature_2m_min']
            }).set_index('Fecha')
            df_diario = df_diario.join(df_clima, how='left')
            for col_c in ['Temp_Media_Tecate', 'Temp_Max_Tecate', 'Temp_Min_Tecate']:
                df_diario[col_c] = df_diario[col_c].ffill()
    except Exception as exc:
        raise RuntimeError(
            "No se pudieron obtener los datos de Open-Meteo. "
            "El pipeline se detiene para no generar un dataset diferente sin avisar."
        ) from exc

    return df_diario


def aplicar_limpieza_outliers(df_train, df_val, df_test, umbral_z=3.5, lower_q=0.01, upper_q=0.99):
    """Acota outliers mediante Z-Modificada (MAD) y Winsorización ajustados SOLO en Train."""
    tr, va, te = df_train.copy(), df_val.copy(), df_test.copy()
    cols_numericas = [
        col for col in tr.select_dtypes(include=[np.number]).columns
        if col != 'Target_Frio_Dia_Siguiente'
    ]

    for col in cols_numericas:
        mediana = tr[col].median()
        mad = np.median(np.abs(tr[col] - mediana))
        if mad > 0:
            lim_inf_z = mediana - (umbral_z * mad / 0.6745)
            lim_sup_z = mediana + (umbral_z * mad / 0.6745)
            tr[col] = tr[col].clip(lower=lim_inf_z, upper=lim_sup_z)
            va[col] = va[col].clip(lower=lim_inf_z, upper=lim_sup_z)
            te[col] = te[col].clip(lower=lim_inf_z, upper=lim_sup_z)

        p_inf = tr[col].quantile(lower_q)
        p_sup = tr[col].quantile(upper_q)
        tr[col] = tr[col].clip(lower=p_inf, upper=p_sup)
        va[col] = va[col].clip(lower=p_inf, upper=p_sup)
        te[col] = te[col].clip(lower=p_inf, upper=p_sup)

    return tr, va, te


def filtrar_imputar_y_dividir(df_diario: pd.DataFrame):
    """Filtra redundancia/significancia en Train, divide 70/15/15, imputa nulos y limpia outliers."""
    print("[4/6] Filtrando redundancia, dividiendo (70/15/15), imputando y tratando outliers...")
    target = 'Target_Frio_Dia_Siguiente'
    n_train = int(len(df_diario) * 0.70)
    df_train_only = df_diario.iloc[:n_train]

    # 1. Filtro de Redundancia (|r| >= 0.90) en Train
    X_features_train = df_train_only.drop(columns=[target]).select_dtypes(include=[np.number])
    matriz_corr = X_features_train.corr().abs()
    upper_tri = matriz_corr.where(np.triu(np.ones(matriz_corr.shape), k=1).astype(bool))

    cols_a_eliminar_red = set()
    for col in upper_tri.columns:
        correlacionadas = upper_tri.index[upper_tri[col] >= 0.90].tolist()
        if correlacionadas:
            grupo = correlacionadas + [col]
            mejor_var = df_train_only[grupo].corrwith(df_train_only[target]).abs().idxmax()
            for c in grupo:
                if c != mejor_var:
                    cols_a_eliminar_red.add(c)

    df_sin_redundancia = df_diario.drop(columns=list(cols_a_eliminar_red))

    # 2. Filtro de P-Value (p < 0.05) en Train
    df_train_sin_red = df_sin_redundancia.iloc[:n_train]
    X_final_candidatas = df_train_sin_red.drop(columns=[target]).select_dtypes(include=[np.number])
    variables_significativas = []

    for col in X_final_candidatas.columns:
        datos_validos = df_train_sin_red[[col, target]].dropna()
        if len(datos_validos) > 2 and datos_validos[col].nunique() > 1:
            _, p_value = pearsonr(datos_validos[col], datos_validos[target])
            if p_value < 0.05:
                variables_significativas.append(col)

    df_diario = df_sin_redundancia[variables_significativas + [target]].copy()

    # 3. Eliminar columnas vacías, 'Unnamed' y patrones administrativos ('Meta', 'Id ', etc.)
    df_diario = df_diario.dropna(axis=1, how='all')
    df_diario = df_diario.loc[:, ~df_diario.columns.str.contains('Unnamed', case=False, na=False)]

    patrones_a_eliminar = ['Meta', 'Id ', 'Auxiliar', 'Ultimo Dato']
    columnas_a_conservar = [
        col for col in df_diario.columns
        if not any(p.lower() in str(col).lower() for p in patrones_a_eliminar) or col in ['DIA', 'HORA']
    ]
    df_diario = df_diario[columnas_a_conservar]

    # 4. Identificar sensores incorporados tarde en Train
    df_train_check = df_diario.iloc[:n_train]
    inicio_corte = int(n_train * 0.30)
    fin_corte = int(n_train * 0.70)
    sensores_protegidos = []

    for col in df_train_check.columns:
        if col == target:
            continue
        serie = df_train_check[col]
        if serie.iloc[:inicio_corte].isnull().mean() > 0.50 and serie.iloc[fin_corte:].isnull().mean() < 0.20:
            sensores_protegidos.append(col)

    # 5. División Temporal (Train 70% / Val 15% / Test 15%)
    X = df_diario.drop(columns=[target])
    y = df_diario[target]

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, shuffle=False)
    val_prop = 0.15 / 0.85
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_prop, shuffle=False)

    # 6. Imputación MCAR (<1%) y MAR (1%-45%) protegiendo sensores nuevos
    cols_normales = [col for col in X_train.columns if col not in sensores_protegidos]
    X_train_norm, X_train_prot = X_train[cols_normales].copy(), X_train[sensores_protegidos].copy()
    X_val_norm, X_val_prot     = X_val[cols_normales].copy(), X_val[sensores_protegidos].copy()
    X_test_norm, X_test_prot   = X_test[cols_normales].copy(), X_test[sensores_protegidos].copy()

    porcentaje_nulos = (X_train_norm.isnull().sum() / len(X_train_norm)) * 100
    cols_mcar = porcentaje_nulos[(porcentaje_nulos > 0) & (porcentaje_nulos < 1.0)].index.tolist()
    cols_mar  = porcentaje_nulos[(porcentaje_nulos >= 1.0) & (porcentaje_nulos <= 45.0)].index.tolist()
    cols_mnar = porcentaje_nulos[porcentaje_nulos > 45.0].index.tolist()

    X_train_norm = X_train_norm.drop(columns=cols_mnar, errors='ignore')
    X_val_norm   = X_val_norm.drop(columns=cols_mnar, errors='ignore')
    X_test_norm  = X_test_norm.drop(columns=cols_mnar, errors='ignore')

    cols_mcar_activas = [c for c in cols_mcar if c in X_train_norm.columns]
    if len(cols_mcar_activas) > 0:
        imputer_mcar = SimpleImputer(strategy='mean')
        X_train_norm[cols_mcar_activas] = imputer_mcar.fit_transform(X_train_norm[cols_mcar_activas])
        X_val_norm[cols_mcar_activas]   = imputer_mcar.transform(X_val_norm[cols_mcar_activas])
        X_test_norm[cols_mcar_activas]  = imputer_mcar.transform(X_test_norm[cols_mcar_activas])

    cols_mar_activas = [c for c in cols_mar if c in X_train_norm.columns]
    if len(cols_mar_activas) > 0:
        imputer_mar = KNNImputer(n_neighbors=5)
        X_train_norm[cols_mar_activas] = imputer_mar.fit_transform(X_train_norm[cols_mar_activas])
        X_val_norm[cols_mar_activas]   = imputer_mar.transform(X_val_norm[cols_mar_activas])
        X_test_norm[cols_mar_activas]  = imputer_mar.transform(X_test_norm[cols_mar_activas])

    X_train = pd.concat([X_train_norm, X_train_prot], axis=1)
    X_val   = pd.concat([X_val_norm, X_val_prot], axis=1)
    X_test  = pd.concat([X_test_norm, X_test_prot], axis=1)

    # 7. Limpieza de Outliers (Z-Modificada + Winsorización)
    X_train_limpio, X_val_limpio, X_test_limpio = aplicar_limpieza_outliers(X_train, X_val, X_test)

    return X_train_limpio, X_val_limpio, X_test_limpio, y_train, y_val, y_test


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
    y_train, y_val, y_test, features_finales, dir_raiz: Path
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
    df_dataset_final.to_csv(ruta_csv, index=True)
    print(f"      -> Guardado: {ruta_csv} (Shape: {df_dataset_final.shape})")

    # Calcular SHA-256 y actualizar checksums.json
    sha256_hash = hashlib.sha256()
    with open(ruta_csv, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    checksum_val = sha256_hash.hexdigest()

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
            "Filtrado de días incompletos (<25 registros), excedidos (>25) y sin reinicio (>100 kW a las 00:00)",
            "Agregación temporal de datos a nivel diario (cierre 23:59) con frecuencia continua ('D')",
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

    X_train_limpio, X_val_limpio, X_test_limpio, y_train, y_val, y_test = filtrar_imputar_y_dividir(df_diario)

    features_finales, X_train_final, X_val_final, X_test_final, _ = escalar_y_seleccionar_variables(
        X_train_limpio, X_val_limpio, X_test_limpio, y_train
    )

    guardar_artefactos(
        X_train_limpio, X_val_limpio, X_test_limpio,
        y_train, y_val, y_test, features_finales, dir_raiz
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
