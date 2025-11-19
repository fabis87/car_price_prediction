"""
Módulo de carga de datos para el proyecto de predicción de precios de vehículos usados.
"""

import pandas as pd


def cargar_datos(path):
    """
    Carga el dataset desde un archivo CSV.
    
    Args:
        path (str): Ruta al archivo CSV
        
    Returns:
        pd.DataFrame: DataFrame con los datos cargados, o None si hay error
    """
    print(f"Cargando datos desde: {path}")
    
    try:
        df = pd.read_csv(path)
        print(f"✓ Datos cargados exitosamente")
        print(f"  Filas: {len(df):,}")
        print(f"  Columnas: {len(df.columns)}")
        return df
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo en {path}")
        return None
    except Exception as e:
        print(f"Error inesperado: {e}")
        return None
    
