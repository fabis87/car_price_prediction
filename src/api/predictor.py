"""
Clase para cargar el modelo y realizar predicciones.
Maneja la carga de artefactos (modelo, encoders, scaler) y el preprocesamiento.
"""

import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class VehiclePricePredictor:
    """
    Predictor de precios de vehículos.
    Carga el modelo entrenado y todos los artefactos necesarios.
    """
    
    def __init__(self, models_dir: str = "src/models"):
        """
        Inicializa el predictor cargando todos los artefactos.
        
        Args:
            models_dir: Directorio donde están guardados los modelos
        """
        self.models_dir = Path(models_dir)
        self.model = None
        self.encoders = None
        self.scaler = None
        self.metadata = None
        
        self._load_artifacts()
    
    def _load_artifacts(self):
        """Carga el modelo, encoders, scaler y metadata."""
        try:
            # Cargar modelo
            model_path = self.models_dir / "xgboost_model.joblib"
            self.model = joblib.load(model_path)
            logger.info(f"✓ Modelo cargado: {model_path}")
            
            # Cargar encoders
            encoders_path = self.models_dir / "encoders.joblib"
            self.encoders = joblib.load(encoders_path)
            logger.info(f"✓ Encoders cargados: {encoders_path}")
            
            # Cargar scaler
            scaler_path = self.models_dir / "scaler.joblib"
            self.scaler = joblib.load(scaler_path)
            logger.info(f"✓ Scaler cargado: {scaler_path}")
            
            # Cargar metadata
            metadata_path = self.models_dir / "metadata.json"
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            logger.info(f"✓ Metadata cargada: {metadata_path}")
            
            logger.info("=" * 80)
            logger.info("PREDICTOR LISTO")
            logger.info(f"Modelo: {self.metadata.get('model_type', 'Unknown')}")
            logger.info(f"R² Test: {self.metadata['metrics']['r2_test']:.4f}")
            logger.info(f"MAE Test: ${self.metadata['metrics']['mae_test']:,.2f}")
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"Error al cargar artefactos: {e}")
            raise
    
    def _create_features(self, data: dict) -> pd.DataFrame:
        """
        Crea todas las features necesarias a partir de los datos de entrada.
        Replica el proceso de feature engineering del pipeline.
        
        Args:
            data: Diccionario con las características del vehículo
            
        Returns:
            DataFrame con todas las features procesadas
        """
        # Crear DataFrame base
        df = pd.DataFrame([data])
        
        # 1. CONDITION_ENCODED (ordinal)
        condition_mapping = {
            'salvage': 0,
            'fair': 1,
            'good': 2,
            'excellent': 3,
            'like new': 4,
            'new': 5
        }
        df['condition_encoded'] = df['condition'].map(condition_mapping)
        
        # 2. LOG_ODOMETER
        df['log_odometer'] = np.log1p(df['odometer'])
        
        # 3. MILEAGE_PER_YEAR
        df['mileage_per_year'] = df['odometer'] / (df['age'] + 1)
        
        # 4. CYLINDERS_SQUARED
        df['cylinders_squared'] = df['cylinders'] ** 2
        
        # 5. INTERACCIONES
        df['age_x_condition'] = df['age'] * df['condition_encoded']
        df['age_x_log_odometer'] = df['age'] * df['log_odometer']
        df['cylinders_x_condition'] = df['cylinders'] * df['condition_encoded']
        
        # 6. FEATURES BINARIAS
        df['is_automatic'] = (df['transmission'] == 'automatic').astype(int)
        df['is_4wd'] = (df['drive'] == '4wd').astype(int)
        df['is_rwd'] = (df['drive'] == 'rwd').astype(int)
        
        return df
    
    def _preprocess_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Preprocesa las features: encoding y normalización.
        
        Args:
            df: DataFrame con todas las features
            
        Returns:
            Array NumPy listo para predicción
        """
        df_processed = df.copy()
        
        # Obtener columnas categóricas y numéricas del metadata
        categorical_cols = self.metadata['categorical_features']
        numerical_cols = [col for col in self.metadata['numerical_features'] if col != 'price']
        
        # 1. ENCODEAR VARIABLES CATEGÓRICAS
        for col in categorical_cols:
            if col in df_processed.columns:
                try:
                    # Usar el encoder entrenado
                    encoder = self.encoders[col]
                    df_processed[col] = encoder.transform(df_processed[col].astype(str))
                except ValueError as e:
                    # Si la categoría no fue vista en el entrenamiento, usar la más común (0)
                    logger.warning(f"Categoría desconocida en '{col}': {df_processed[col].iloc[0]}. Usando valor por defecto.")
                    df_processed[col] = 0
        
        # 2. NORMALIZAR FEATURES NUMÉRICAS
        if self.scaler is not None:
            df_processed[numerical_cols] = self.scaler.transform(df_processed[numerical_cols])
        
        # 3. ORDENAR COLUMNAS EN EL ORDEN CORRECTO
        feature_order = categorical_cols + numerical_cols
        df_processed = df_processed[feature_order]
        
        return df_processed.values
    
    def predict(self, vehicle_data: dict) -> dict:
        """
        Realiza una predicción de precio.
        
        Args:
            vehicle_data: Diccionario con las características del vehículo
            
        Returns:
            Diccionario con la predicción y metadatos
        """
        try:
            # 1. Crear features
            df_features = self._create_features(vehicle_data)
            
            # 2. Preprocesar
            X = self._preprocess_features(df_features)
            
            # 3. Predecir
            prediction = self.model.predict(X)[0]
            
            # 4. Calcular intervalo de confianza aproximado (±15% basado en MAE)
            mae = self.metadata['metrics']['mae_test']
            confidence_margin = mae * 1.5  # Margen de confianza
            
            result = {
                'predicted_price': float(prediction),
                'predicted_price_formatted': f"${prediction:,.2f}",
                'confidence_interval_lower': float(max(0, prediction - confidence_margin)),
                'confidence_interval_upper': float(prediction + confidence_margin),
                'model_version': f"{self.metadata['model_type']}-v1.0"
            }
            
            logger.info(f"Predicción realizada: ${prediction:,.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Error en predicción: {e}")
            raise
    
    def get_model_info(self) -> dict:
        """
        Retorna información sobre el modelo cargado.
        
        Returns:
            Diccionario con metadata del modelo
        """
        return {
            'model_type': self.metadata.get('model_type', 'Unknown'),
            'training_date': self.metadata.get('training_date', 'Unknown'),
            'metrics': self.metadata.get('metrics', {}),
            'total_features': self.metadata.get('total_features', 0),
            'categorical_features': self.metadata.get('categorical_features', []),
            'numerical_features': self.metadata.get('numerical_features', [])
        }
