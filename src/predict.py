"""Predice el consumo de frío de D+1 a partir del Excel de operaciones."""

import argparse
import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import BadZipFile

import joblib
import numpy as np
import pandas as pd
from openpyxl.utils.exceptions import InvalidFileException

from preprocessing_pipeline import preparar_excel_prediccion


RAIZ = Path(__file__).resolve().parent.parent


def load_model(version='latest'):
    """Carga una versión del registro junto con su preprocesamiento ajustado."""
    registro = RAIZ / 'models' / 'model_registry.json'
    if not registro.is_file():
        raise FileNotFoundError(f'Falta {registro}. Ejecutá train_model.py --finalizar primero.')
    catalogo = json.loads(registro.read_text(encoding='utf-8'))
    modelos = catalogo.get('models')
    if not isinstance(modelos, dict) or not modelos:
        raise ValueError('El registro no contiene modelos.')

    if version == 'latest':
        disponibles = [v for v in modelos if re.fullmatch(r'v\d+\.\d+\.\d+', v)]
        if not disponibles:
            raise ValueError('El registro no contiene versiones semánticas, por ejemplo v1.0.0.')
        version = max(disponibles, key=lambda v: tuple(map(int, v[1:].split('.'))))
    if version not in modelos:
        raise ValueError(f'No existe la versión {version!r}. Disponibles: {", ".join(modelos)}.')

    info = modelos[version]
    ruta_modelo = info.get('path')
    if not ruta_modelo:
        raise ValueError(f'Falta la ruta del modelo {version} en el registro.')
    modelo = joblib.load(RAIZ / ruta_modelo)
    preprocesador = getattr(modelo, 'preprocesador_datos_', None)
    if preprocesador is None:
        raise ValueError(
            f'El modelo {version} no incluye un preprocesador. Corré el preprocesamiento y '
            'volvé a finalizar el modelo con una versión nueva.'
        )
    if (preprocesador['variables'] != info.get('columnas_entrada')
            or preprocesador['dataset_sha256'] != info.get('dataset_sha256')):
        raise ValueError(f'El preprocesador de {version} no coincide con el registro.')
    return modelo


def load_and_preprocess(filepath, preprocesador=None):
    ruta = Path(filepath).expanduser()
    if not ruta.is_file():
        raise FileNotFoundError(f'No existe el Excel de entrada: {ruta}')
    if ruta.suffix.lower() != '.xlsx':
        raise ValueError(f'Se esperaba un archivo .xlsx: {ruta}')
    if preprocesador is None:
        preprocesador = load_model().preprocesador_datos_
    return preparar_excel_prediccion(ruta, preprocesador)


def predict_consumption(filepath, version='latest'):
    modelo = load_model(version)
    X, fechas = load_and_preprocess(filepath, modelo.preprocesador_datos_)
    predicciones = modelo.predict(X)
    if not np.isfinite(predicciones).all():
        raise ValueError('El modelo produjo predicciones no finitas.')
    return pd.DataFrame({'fecha': fechas, 'prediccion_frio_kw': predicciones})


def main():
    parser = argparse.ArgumentParser(description='Predice el Frío de D+1 desde un Excel de totalizadores.')
    parser.add_argument('excel', type=Path, help='Archivo .xlsx de entrada')
    parser.add_argument('--version', default='latest', help='Versión del modelo (ej. v1.0.0)')
    parser.add_argument('--salida', type=Path, default=RAIZ / 'results' / 'predicciones.csv')
    args = parser.parse_args()

    try:
        resultados = predict_consumption(args.excel, args.version)
        args.salida.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=args.salida.parent) as carpeta:
            temporal = Path(carpeta) / 'predicciones.csv'
            resultados.to_csv(temporal, index=False, date_format='%Y-%m-%d', lineterminator='\n')
            temporal.replace(args.salida)
    except (BadZipFile, InvalidFileException) as exc:
        parser.exit(1, f'Error: no se pudo abrir el Excel {args.excel}: {exc}\n')
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f'Error: {exc}\n')

    print(f'{len(resultados)} predicciones generadas en {args.salida}')


if __name__ == '__main__':
    main()
