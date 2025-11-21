"""
Script para generar datos agregados de EDA como JSON.
Estos datos se usan en el dashboard para visualizaciones sin cargar el CSV completo.
"""

import pandas as pd
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_eda_data_json(csv_path: str, output_path: str):
    """
    Genera un archivo JSON con datos agregados para EDA.
    
    Args:
        csv_path: Ruta al CSV procesado
        output_path: Ruta donde guardar el JSON
    """
    logger.info(f"Leyendo dataset desde: {csv_path}")
    df = pd.read_csv(csv_path)
    
    logger.info(f"Dataset cargado: {len(df):,} filas")
    logger.info("Generando datos agregados...")
    
    # 1. Estadísticas generales
    general_stats = {
        "total_vehicles": int(len(df)),
        "price_mean": float(df['price'].mean()),
        "price_median": float(df['price'].median()),
        "price_std": float(df['price'].std()),
        "price_min": float(df['price'].min()),
        "price_max": float(df['price'].max()),
        "price_q25": float(df['price'].quantile(0.25)),
        "price_q75": float(df['price'].quantile(0.75)),
        "unique_manufacturers": int(df['manufacturer'].nunique()),
        "unique_types": int(df['type'].nunique()),
        "unique_states": int(df['state'].nunique())
    }
    
    # 2. Precio por edad (depreciación)
    price_by_age = df.groupby('age')['price'].agg(['mean', 'median', 'count']).reset_index()
    price_by_age = price_by_age[price_by_age['count'] > 100]  # Filtrar edades con pocos datos
    depreciation_data = {
        "age": price_by_age['age'].astype(int).tolist(),
        "price_mean": price_by_age['mean'].round(2).astype(float).tolist(),
        "price_median": price_by_age['median'].round(2).astype(float).tolist(),
        "count": price_by_age['count'].astype(int).tolist()
    }
    
    # 3. Precio promedio por marca (top 20)
    top_manufacturers = df.groupby('manufacturer')['price'].mean().sort_values(ascending=False).head(20)
    manufacturers_data = {
        "manufacturers": top_manufacturers.index.tolist(),
        "prices": top_manufacturers.round(2).astype(float).tolist()
    }
    
    # 4. Distribución de precios (histograma)
    # Crear bins para histograma
    bins = [0, 5000, 10000, 15000, 20000, 25000, 30000, 40000, 50000, 75000, 100000, float('inf')]
    labels = ['0-5k', '5k-10k', '10k-15k', '15k-20k', '20k-25k', '25k-30k', 
              '30k-40k', '40k-50k', '50k-75k', '75k-100k', '100k+']
    df['price_bin'] = pd.cut(df['price'], bins=bins, labels=labels, include_lowest=True)
    price_distribution = df['price_bin'].value_counts().sort_index()
    distribution_data = {
        "bins": price_distribution.index.tolist(),
        "counts": price_distribution.astype(int).tolist()
    }
    
    # 5. Precio por condición
    price_by_condition = df.groupby('condition')['price'].agg(['mean', 'median', 'count']).reset_index()
    condition_order = ['salvage', 'fair', 'good', 'excellent', 'like new', 'new']
    price_by_condition['condition'] = pd.Categorical(price_by_condition['condition'], categories=condition_order, ordered=True)
    price_by_condition = price_by_condition.sort_values('condition')
    condition_data = {
        "conditions": price_by_condition['condition'].tolist(),
        "price_mean": price_by_condition['mean'].round(2).astype(float).tolist(),
        "price_median": price_by_condition['median'].round(2).astype(float).tolist(),
        "count": price_by_condition['count'].astype(int).tolist()
    }
    
    # Compilar todos los datos
    eda_data = {
        "general_stats": general_stats,
        "depreciation": depreciation_data,
        "manufacturers": manufacturers_data,
        "price_distribution": distribution_data,
        "condition": condition_data
    }
    
    # Guardar JSON
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(eda_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✓ EDA data JSON guardado en: {output_path}")
    logger.info(f"  Tamaño del archivo: {Path(output_path).stat().st_size / 1024:.2f} KB")
    logger.info(f"  - Estadísticas generales: ✓")
    logger.info(f"  - Datos de depreciación: {len(depreciation_data['age'])} puntos")
    logger.info(f"  - Top marcas: {len(manufacturers_data['manufacturers'])}")
    logger.info(f"  - Distribución de precios: {len(distribution_data['bins'])} bins")
    logger.info(f"  - Precio por condición: {len(condition_data['conditions'])} condiciones")


if __name__ == "__main__":
    # Rutas
    csv_path = "data/processed/vehicles_with_features.csv"
    output_path = "src/models/eda_data.json"
    
    if not Path(csv_path).exists():
        logger.error(f"❌ No se encontró el archivo: {csv_path}")
        logger.error("   Ejecuta primero el pipeline de datos.")
        exit(1)
    
    generate_eda_data_json(csv_path, output_path)
    logger.info("✅ Proceso completado")

