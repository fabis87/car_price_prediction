"""
Módulo de guardado de datos para el proyecto de predicción de precios de vehículos usados.
"""

import pandas as pd
import os


def guardar_datos_limpios(df, path): 
    """
    Guarda el DataFrame procesado en un archivo CSV.
    Crea el directorio si no existe.
    
    Args:
        df (pd.DataFrame): DataFrame a guardar
        path (str): Ruta donde guardar el archivo
        
    Returns:
        bool: True si se guardó exitosamente, False en caso contrario
    """
    try: 
        print(f"Guardando datos en: {path}")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        df.to_csv(path, index=False)
        
        file_size = os.path.getsize(path) / (1024 * 1024)
        print(f"✓ Archivo guardado exitosamente ({file_size:.2f} MB)")
        
        return True
    
    except Exception as e:
        print(f"Error al guardar los datos: {e}")
        return False