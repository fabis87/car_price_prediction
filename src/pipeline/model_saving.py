"""
Módulo de guardado de modelos y artefactos de Machine Learning.
Persiste modelos, encoders y metadata.
"""

import os
import json
import joblib
from datetime import datetime


def crear_directorio_modelos(modelo_dir='src/models'):
    """
    Crea el directorio para guardar modelos si no existe.
    
    Args:
        modelo_dir (str): Ruta del directorio
        
    Returns:
        str: Ruta absoluta del directorio creado
    """
    os.makedirs(modelo_dir, exist_ok=True)
    print(f"Directorio de modelos: {modelo_dir}")
    return modelo_dir


def guardar_modelo(model, path):
    """
    Guarda el modelo entrenado en disco.
    
    Args:
        model: Modelo a guardar (XGBoost, LightGBM, etc.)
        path (str): Ruta donde guardar el modelo
        
    Returns:
        bool: True si se guardó exitosamente
    """
    try:
        joblib.dump(model, path)
        file_size = os.path.getsize(path) / (1024 * 1024)
        print(f"✓ Modelo guardado: {path} ({file_size:.2f} MB)")
        return True
    except Exception as e:
        print(f"Error al guardar modelo: {e}")
        return False


def guardar_encoders(encoders_dict, path):
    """
    Guarda todos los LabelEncoders en un archivo.
    
    Args:
        encoders_dict (dict): Diccionario de encoders
        path (str): Ruta donde guardar los encoders
        
    Returns:
        bool: True si se guardó exitosamente
    """
    try:
        joblib.dump(encoders_dict, path)
        print(f"✓ Encoders guardados: {path}")
        print(f"  Contiene {len(encoders_dict)} encoders:")
        for col in encoders_dict.keys():
            print(f"    - {col}")
        return True
    except Exception as e:
        print(f"Error al guardar encoders: {e}")
        return False


def guardar_scaler(scaler, path):
    """
    Guarda el StandardScaler en un archivo.
    Necesario para normalizar nuevos datos antes de predicción.
    
    Args:
        scaler: StandardScaler objeto
        path (str): Ruta donde guardar el scaler
        
    Returns:
        bool: True si se guardó exitosamente
    """
    try:
        if scaler is None:
            print("Scaler es None, no se guarda")
            return True
        
        joblib.dump(scaler, path)
        print(f"✓ Scaler guardado: {path}")
        return True
    except Exception as e:
        print(f"Error al guardar scaler: {e}")
        return False


def guardar_metadata(model_info, path):
    """
    Guarda metadata del modelo y entrenamiento en JSON.
    
    Args:
        model_info (dict): Información del modelo
        path (str): Ruta donde guardar el JSON
        
    Returns:
        bool: True si se guardó exitosamente
    """
    try:
        metadata = {
            'training_date': datetime.now().isoformat(),
            'model_type': model_info.get('model_type', 'unknown'),
            'metrics': {
                'mae_train': float(model_info.get('mae_train', 0)),
                'rmse_train': float(model_info.get('rmse_train', 0)),
                'r2_train': float(model_info.get('r2_train', 0)),
                'mae_test': float(model_info.get('mae_test', 0)),
                'rmse_test': float(model_info.get('rmse_test', 0)),
                'r2_test': float(model_info.get('r2_test', 0)),
            },
            'categorical_features': model_info.get('categorical_cols', []),
            'numerical_features': model_info.get('numerical_cols', []),
            'total_features': model_info.get('total_features', 0),
            'training_samples': model_info.get('training_samples', 0),
            'test_samples': model_info.get('test_samples', 0),
        }
        
        with open(path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Metadata guardado: {path}")
        return True
    except Exception as e:
        print(f"Error al guardar metadata: {e}")
        return False


def pipeline_model_saving(model_final, encoders_dict, scaler, model_info, 
                          modelo_dir='src/models'):
    """
    Ejecuta el pipeline completo de guardado del modelo y artefactos.
    
    Guarda:
    1. Modelo entrenado (xgboost_model.joblib)
    2. Encoders (encoders.joblib)
    3. Scaler (scaler.joblib)
    4. Metadata (metadata.json)
    
    Args:
        model_final: Modelo entrenado
        encoders_dict (dict): Diccionario de encoders
        scaler: StandardScaler objeto
        model_info (dict): Información del modelo
        modelo_dir (str): Directorio donde guardar archivos
        
    Returns:
        dict: Rutas de los archivos guardados
    """
    print("=" * 80)
    print("GUARDANDO MODELO Y ARTEFACTOS")
    print("=" * 80)
    
    # Crear directorio
    modelo_dir = crear_directorio_modelos(modelo_dir)
    
    # Rutas
    model_path = os.path.join(modelo_dir, "xgboost_model.joblib")
    encoders_path = os.path.join(modelo_dir, "encoders.joblib")
    scaler_path = os.path.join(modelo_dir, "scaler.joblib")
    metadata_path = os.path.join(modelo_dir, "metadata.json")
    
    # Guardar artefactos
    guardar_modelo(model_final, model_path)
    guardar_encoders(encoders_dict, encoders_path)
    guardar_scaler(scaler, scaler_path)
    guardar_metadata(model_info, metadata_path)
    
    print("=" * 80)
    
    # Retornar rutas
    rutas = {
        'model': model_path,
        'encoders': encoders_path,
        'scaler': scaler_path,
        'metadata': metadata_path
    }
    
    print("\nArtefactos guardados exitosamente:")
    for clave, ruta in rutas.items():
        print(f"  - {clave}: {ruta}")
    
    return rutas