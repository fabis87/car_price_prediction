"""
Modelos Pydantic para validación de requests y responses de la API.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Literal, Optional


class VehicleFeatures(BaseModel):
    """
    Modelo de entrada para predicción de precios.
    Representa las características de un vehículo.
    """
    manufacturer: str = Field(..., description="Marca del vehículo (ej: toyota, ford, honda)")
    condition: Literal['salvage', 'fair', 'good', 'excellent', 'like new', 'new'] = Field(
        ..., 
        description="Condición del vehículo"
    )
    cylinders: int = Field(..., ge=3, le=12, description="Número de cilindros (3-12)")
    fuel: str = Field(..., description="Tipo de combustible (gas, diesel, electric, hybrid, other)")
    odometer: float = Field(..., ge=0, le=300000, description="Kilometraje en millas (0-300,000)")
    transmission: str = Field(..., description="Tipo de transmisión (automatic, manual, other)")
    drive: Literal['fwd', 'rwd', '4wd'] = Field(..., description="Tipo de tracción")
    vehicle_type: str = Field(..., alias='type', description="Tipo de vehículo (sedan, suv, truck, etc)")
    state: str = Field(..., min_length=2, max_length=2, description="Estado (código de 2 letras)")
    age: int = Field(..., ge=0, le=45, description="Edad del vehículo en años (0-45)")
    
    @field_validator('manufacturer', 'fuel', 'transmission', 'vehicle_type', 'state')
    @classmethod
    def lowercase_strings(cls, v: str) -> str:
        """Convierte strings a minúsculas para consistencia."""
        return v.lower().strip()
    
    @field_validator('state')
    @classmethod
    def validate_state(cls, v: str) -> str:
        """Valida que el estado sea de 2 caracteres."""
        v = v.lower().strip()
        if len(v) != 2:
            raise ValueError('El estado debe ser un código de 2 letras (ej: ca, tx, ny)')
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "manufacturer": "toyota",
                "condition": "good",
                "cylinders": 6,
                "fuel": "gas",
                "odometer": 45000,
                "transmission": "automatic",
                "drive": "4wd",
                "type": "pickup",
                "state": "tx",
                "age": 5
            }
        }


class PricePrediction(BaseModel):
    """
    Modelo de respuesta para predicción de precios.
    """
    predicted_price: float = Field(..., description="Precio predicho en USD")
    predicted_price_formatted: str = Field(..., description="Precio formateado (ej: $25,430.50)")
    confidence_interval_lower: Optional[float] = Field(None, description="Límite inferior del intervalo de confianza")
    confidence_interval_upper: Optional[float] = Field(None, description="Límite superior del intervalo de confianza")
    model_version: str = Field(..., description="Versión del modelo usado")
    
    class Config:
        json_schema_extra = {
            "example": {
                "predicted_price": 25430.50,
                "predicted_price_formatted": "$25,430.50",
                "confidence_interval_lower": 22000.00,
                "confidence_interval_upper": 29000.00,
                "model_version": "XGBoost-v1.0"
            }
        }


class CategoryOptions(BaseModel):
    """
    Modelo para devolver las opciones disponibles de categorías.
    """
    manufacturers: list[str]
    conditions: list[str]
    fuels: list[str]
    transmissions: list[str]
    drives: list[str]
    types: list[str]
    states: list[str]
    cylinders_range: dict[str, int]
    age_range: dict[str, int]
    odometer_range: dict[str, int]


class HealthCheck(BaseModel):
    """
    Modelo para health check de la API.
    """
    status: str
    model_loaded: bool
    model_type: str
    training_date: str
    api_version: str
