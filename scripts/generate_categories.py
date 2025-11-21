"""
Script para generar categories.json desde el dataset procesado.
Este archivo se usa en la API para obtener las categorías sin leer el CSV completo.
"""

import pandas as pd
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_categories_json(csv_path: str, output_path: str):
    """
    Genera un archivo JSON con todas las categorías y rangos del dataset.
    
    Args:
        csv_path: Ruta al CSV procesado
        output_path: Ruta donde guardar el JSON
    """
    logger.info(f"Leyendo dataset desde: {csv_path}")
    df = pd.read_csv(csv_path)
    
    logger.info(f"Dataset cargado: {len(df):,} filas")
    
    # Extraer categorías
    categories = {
        "manufacturers": sorted(df['manufacturer'].dropna().unique().tolist()),
        "conditions": ['salvage', 'fair', 'good', 'excellent', 'like new', 'new'],
        "fuels": sorted(df['fuel'].dropna().unique().tolist()),
        "transmissions": sorted(df['transmission'].dropna().unique().tolist()),
        "drives": ['fwd', 'rwd', '4wd'],
        "types": sorted(df['type'].dropna().unique().tolist()),
        "states": sorted(df['state'].dropna().unique().tolist()),
        "cylinders_range": {
            "min": int(df['cylinders'].min()),
            "max": int(df['cylinders'].max())
        },
        "age_range": {
            "min": int(df['age'].min()),
            "max": int(df['age'].max())
        },
        "odometer_range": {
            "min": int(df['odometer'].min()),
            "max": int(df['odometer'].max())
        }
    }
    
    # Guardar JSON
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(categories, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✓ Categories JSON guardado en: {output_path}")
    logger.info(f"  - Manufacturers: {len(categories['manufacturers'])}")
    logger.info(f"  - Fuels: {len(categories['fuels'])}")
    logger.info(f"  - Types: {len(categories['types'])}")
    logger.info(f"  - States: {len(categories['states'])}")
    logger.info(f"  - Cylinders: {categories['cylinders_range']['min']}-{categories['cylinders_range']['max']}")
    logger.info(f"  - Age: {categories['age_range']['min']}-{categories['age_range']['max']}")
    logger.info(f"  - Odometer: {categories['odometer_range']['min']}-{categories['odometer_range']['max']}")


if __name__ == "__main__":
    # Rutas
    csv_path = "data/processed/vehicles_with_features.csv"
    output_path = "src/models/categories.json"
    
    if not Path(csv_path).exists():
        logger.error(f"❌ No se encontró el archivo: {csv_path}")
        logger.error("   Ejecuta primero el pipeline de datos.")
        exit(1)
    
    generate_categories_json(csv_path, output_path)
    logger.info("✅ Proceso completado")

