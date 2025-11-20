"""
Script para probar la API localmente.
Realiza una predicción de ejemplo.
"""

import requests
import json

# URL de la API (ajustar según donde corra)
API_URL = "http://localhost:8000"


def test_health():
    """Prueba el endpoint de health check."""
    print("=" * 80)
    print("TEST: Health Check")
    print("=" * 80)
    
    response = requests.get(f"{API_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()


def test_categories():
    """Prueba el endpoint de categorías."""
    print("=" * 80)
    print("TEST: Get Categories")
    print("=" * 80)
    
    response = requests.get(f"{API_URL}/vehicles/categories")
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Manufacturers: {len(data['manufacturers'])} opciones")
        print(f"  Ejemplos: {data['manufacturers'][:5]}")
        print(f"Fuels: {data['fuels']}")
        print(f"Transmissions: {data['transmissions']}")
        print(f"Types: {len(data['types'])} opciones")
        print(f"Cylinders range: {data['cylinders_range']}")
        print(f"Age range: {data['age_range']}")
        print(f"Odometer range: {data['odometer_range']}")
    else:
        print(f"Error: {response.text}")
    print()


def test_prediction():
    """Prueba el endpoint de predicción."""
    print("=" * 80)
    print("TEST: Predict Price")
    print("=" * 80)
    
    # Vehículo de ejemplo
    vehicle_data = {
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
    
    print("Input:")
    print(json.dumps(vehicle_data, indent=2))
    print()
    
    response = requests.post(
        f"{API_URL}/vehicles/predict_price",
        json=vehicle_data
    )
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        prediction = response.json()
        print("Prediction:")
        print(json.dumps(prediction, indent=2))
    else:
        print(f"Error: {response.text}")
    print()


def test_model_info():
    """Prueba el endpoint de información del modelo."""
    print("=" * 80)
    print("TEST: Model Info")
    print("=" * 80)
    
    response = requests.get(f"{API_URL}/model/info")
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        info = response.json()
        print(f"Model Type: {info['model_type']}")
        print(f"Training Date: {info['training_date']}")
        print(f"Total Features: {info['total_features']}")
        print(f"Metrics:")
        for metric, value in info['metrics'].items():
            print(f"  - {metric}: {value}")
    else:
        print(f"Error: {response.text}")
    print()


if __name__ == "__main__":
    print("\n")
    print("🚗 PRUEBAS DE LA API - VEHICLE PRICE PREDICTION 🚗")
    print("\n")
    
    try:
        test_health()
        test_categories()
        test_model_info()
        test_prediction()
        
        print("=" * 80)
        print("✓ TODAS LAS PRUEBAS COMPLETADAS")
        print("=" * 80)
        
    except requests.exceptions.ConnectionError:
        print("❌ ERROR: No se pudo conectar a la API")
        print(f"   Asegúrate de que la API esté corriendo en {API_URL}")
        print("   Ejecuta: uvicorn src.api.main:app --reload")
    except Exception as e:
        print(f"❌ ERROR: {e}")
