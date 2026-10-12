import pandas as pd
from pathlib import Path
from functools import reduce

# 1. Definir la ruta y los archivos a leer (ordenados del más antiguo al más nuevo)
carpeta = Path(__file__).resolve().parent
archivos_excel = [
    "Totalizadores Planta 2020_2022.xlsx",
    "Totalizadores Planta - 2021_2023.xlsx",
    "Totalizadores Planta 2022_2023.xlsx"
]

# Diccionarios para agrupar los datos
datos_por_hoja = {}          # Para el consolidado de las 23:59
datos_por_hoja_ceros = {}    # Para el análisis de reinicios a las 00:00

# 2. Leer todos los archivos y agrupar sus hojas
for nombre_archivo in archivos_excel:
    ruta_excel = carpeta / nombre_archivo
    print(f"Leyendo archivo: {nombre_archivo}...")
    
    diccionario_hojas = pd.read_excel(ruta_excel, sheet_name=None)
    
    for nombre_hoja, df in diccionario_hojas.items():
        if 'DIA' not in df.columns or 'HORA' not in df.columns:
            continue

        # 1. Normalizar la fecha
        df['DIA'] = pd.to_datetime(df['DIA'], errors='coerce').dt.normalize()

        # 2. VERIFICACIÓN DE JORNADAS APILADAS (> 25 registros)
        conteo_dias = df.groupby('DIA').size()
        dias_corruptos = conteo_dias[conteo_dias > 25]

        if not dias_corruptos.empty:
            print(f"\n🚨 ALERTA en '{nombre_archivo}' - Hoja '{nombre_hoja}':")
            print("Se encontraron fechas con más de 25 registros (jornadas apiladas):")
            for fecha, cantidad in dias_corruptos.items():
                print(f"   - {fecha.strftime('%Y-%m-%d')}: {cantidad} registros")
            
            df = df[~df['DIA'].isin(dias_corruptos.index)].copy()
            print("   -> 🗑️ Estos días fueron descartados para proteger la integridad del modelo.\n")

        # Limpiar la columna HORA para evitar espacios raros
        df['HORA'] = df['HORA'].astype(str).str.strip()
        
        # 3. Filtrar las 23:59 para el dataset final
        # Usamos startswith para ser exactos y evitar errores con "123:59"
        df_filtrado = df[df['HORA'].str.startswith('23:59', na=False)].copy()
        if not df_filtrado.empty:
            if nombre_hoja not in datos_por_hoja:
                datos_por_hoja[nombre_hoja] = []
            datos_por_hoja[nombre_hoja].append(df_filtrado)

        # 4. Filtrar las 00:00 para el análisis de reinicios
        # Usamos startswith para evitar que agarre 10:00:00 o 20:00:00
        df_ceros = df[df['HORA'].str.startswith('00:00', na=False)].copy()
        if not df_ceros.empty:
            if nombre_hoja not in datos_por_hoja_ceros:
                datos_por_hoja_ceros[nombre_hoja] = []
            datos_por_hoja_ceros[nombre_hoja].append(df_ceros)

# =============================================================================
# NUEVO BLOQUE: VERIFICACIÓN DE REINICIO A LAS 00:00
# =============================================================================
ruta_log_ceros = carpeta / "Reporte_Fallas_Reinicio_0000.txt"
print("\nVerificando si los totalizadores se reiniciaron correctamente a las 00:00...")
hubo_fallas_reinicio = False

with open(ruta_log_ceros, 'w', encoding='utf-8') as archivo_ceros:
    archivo_ceros.write("=== REPORTE DE TOTALIZADORES QUE NO SE REINICIARON A LAS 00:00 ===\n")
    archivo_ceros.write("Se listan los días y las columnas que mantuvieron valores mayores a 0 en la medianoche.\n\n")

    for nombre_hoja, lista_dfs in datos_por_hoja_ceros.items():
        # Juntamos y eliminamos duplicados igual que con los datos principales
        df_hoja_ceros = pd.concat(lista_dfs, ignore_index=True)
        df_hoja_ceros = df_hoja_ceros.drop_duplicates(subset=['DIA', 'HORA'], keep='last')
        
        columnas_valores = [col for col in df_hoja_ceros.columns if col not in ['DIA', 'HORA']]
        fallas_en_esta_hoja = False
        
        # Escanear cada registro de 00:00
        for idx, row in df_hoja_ceros.iterrows():
            dia_str = row['DIA'].strftime('%Y-%m-%d')
            columnas_no_reiniciadas = []
            
            for col in columnas_valores:
                val = row[col]
                try:
                    val_num = float(val)
                    # Si el número es mayor a cero, significa que no se reinició
                    if pd.notna(val_num) and val_num > 0:
                        columnas_no_reiniciadas.append(f"{col}: {val_num}")
                except (ValueError, TypeError):
                    pass # Ignorar celdas de texto si las hubiera
                    
            if columnas_no_reiniciadas:
                if not fallas_en_esta_hoja:
                    archivo_ceros.write(f"--- Hoja: {nombre_hoja} ---\n")
                    fallas_en_esta_hoja = True
                    hubo_fallas_reinicio = True
                
                archivo_ceros.write(f"Fecha: {dia_str}\n")
                for falla in columnas_no_reiniciadas:
                    archivo_ceros.write(f"  - {falla}\n")
        
        if fallas_en_esta_hoja:
            archivo_ceros.write("\n")

    if not hubo_fallas_reinicio:
        archivo_ceros.write("¡Excelente! Todos los totalizadores registrados a las 00:00 estaban en 0.\n")


# =============================================================================
# BLOQUE ORIGINAL: APILAR AÑOS Y BUSCAR DISCREPANCIAS
# =============================================================================
dfs_listos_para_cruzar = []
print("\nProcesando y buscando discrepancias en fechas solapadas...")

ruta_log_discrepancias = carpeta / "Reporte_Discrepancias.txt"
hubo_discrepancias = False 

with open(ruta_log_discrepancias, 'w', encoding='utf-8') as archivo_log:
    archivo_log.write("=== REPORTE DE DISCREPANCIAS ===\n")
    archivo_log.write("Muestra las fechas superpuestas que tienen valores diferentes entre archivos.\n")
    archivo_log.write("CRITERIO DE DESEMPATE APLICADO: Se conservó el dato del archivo más reciente.\n\n")

    for nombre_hoja, lista_dfs in datos_por_hoja.items():
        df_hoja_completa = pd.concat(lista_dfs, ignore_index=True)
        
        # Verificación de discrepancias
        duplicados_fecha = df_hoja_completa[df_hoja_completa.duplicated(subset=['DIA', 'HORA'], keep=False)]
        
        if not duplicados_fecha.empty:
            duplicados_unicos = duplicados_fecha.drop_duplicates()
            discrepancias = duplicados_unicos[duplicados_unicos.duplicated(subset=['DIA', 'HORA'], keep=False)]
            
            if not discrepancias.empty:
                hubo_discrepancias = True
                print(f"  -> ⚠️ Discrepancias detectadas en la hoja '{nombre_hoja}' (Agregado al reporte txt)")
                archivo_log.write(f"⚠️ ATENCIÓN: Discrepancias en la hoja '{nombre_hoja}'\n")
                archivo_log.write(discrepancias.sort_values(['DIA']).to_string())
                archivo_log.write("\n" + "-" * 80 + "\n\n")
        
        # ELIMINAR SOLAPAMIENTO
        df_hoja_completa = df_hoja_completa.drop_duplicates(subset=['DIA', 'HORA'], keep='last')
        
        # Renombrar columnas
        columnas_valores = [col for col in df_hoja_completa.columns if col not in ['DIA', 'HORA']]
        renombres = {col: f"{col} ({nombre_hoja})" for col in columnas_valores}
        df_hoja_completa.rename(columns=renombres, inplace=True)
        
        dfs_listos_para_cruzar.append(df_hoja_completa)
        
    if not hubo_discrepancias:
        archivo_log.write("¡Todo perfecto! No se encontraron valores discrepantes entre los archivos solapados.\n")

# 4. Cruzar todas las hojas procesadas
if dfs_listos_para_cruzar:
    print("\nCruzando todas las tablas en una sola...")
    df_final = reduce(lambda left, right: pd.merge(left, right, on=['DIA', 'HORA'], how='outer'), dfs_listos_para_cruzar)
    
    df_final = df_final.drop(columns=['HORA'])
    df_final = df_final.sort_values('DIA').reset_index(drop=True)
    
    ruta_salida_csv = carpeta / "Totalizadores_2020_2023.csv"
    df_final.to_csv(ruta_salida_csv, index=False, sep=';', decimal=',', encoding='utf-8-sig')
    
    print(f"\n¡Éxito! Procesamiento terminado.")
    print(f"📄 Archivo consolidado guardado en: {ruta_salida_csv}")
    print(f"📝 Reporte de discrepancias en: {ruta_log_discrepancias}")
    print(f"⏱️  Reporte de fallas de reinicio a las 00:00 en: {ruta_log_ceros}")

else:
    print("\nNo se encontraron registros.")