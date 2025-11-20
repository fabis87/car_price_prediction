"""
Dashboard Streamlit - Herramienta de Tasación de Vehículos Usados
Consume la API FastAPI para realizar predicciones de precios.
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

st.set_page_config(
    page_title="Tasación de Vehículos",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# URL de la API (prioridad: variable de entorno > secrets > localhost)
import os

API_URL = os.getenv("API_URL")
if not API_URL:
    try:
        API_URL = st.secrets["API_URL"]
    except (FileNotFoundError, KeyError):
        API_URL = "http://localhost:8000"

# ============================================================================
# FUNCIONES DE API
# ============================================================================

@st.cache_data(ttl=3600)
def get_categories():
    """Obtiene las categorías disponibles de la API."""
    try:
        response = requests.get(f"{API_URL}/vehicles/categories", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Error al conectar con la API: {e}")
        return None


@st.cache_data(ttl=3600)
def get_model_info():
    """Obtiene información del modelo."""
    try:
        response = requests.get(f"{API_URL}/model/info", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


def check_api_health():
    """Verifica que la API esté disponible."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200 and response.json().get('status') == 'healthy'
    except:
        return False


def predict_price(vehicle_data: dict) -> Optional[dict]:
    """Envía datos a la API y obtiene predicción."""
    try:
        response = requests.post(
            f"{API_URL}/vehicles/predict_price",
            json=vehicle_data,
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error: {response.json().get('detail', 'Error desconocido')}")
            return None
    except Exception as e:
        st.error(f"Error de conexión: {e}")
        return None


# ============================================================================
# FUNCIONES DE VISUALIZACIÓN EDA
# ============================================================================

@st.cache_data
def load_data_for_eda():
    """Carga datos procesados para EDA."""
    try:
        df = pd.read_csv('data/processed/vehicles_with_features.csv')
        return df
    except:
        return None


def plot_price_by_manufacturer(df, top_n=15):
    """Gráfico: Precio promedio por marca."""
    top_manufacturers = df.groupby('manufacturer')['price'].mean().sort_values(ascending=False).head(top_n)
    
    fig = px.bar(
        x=top_manufacturers.values,
        y=top_manufacturers.index,
        orientation='h',
        labels={'x': 'Precio Promedio (USD)', 'y': 'Marca'},
        title=f'Top {top_n} Marcas por Precio Promedio',
        color=top_manufacturers.values,
        color_continuous_scale='Viridis'
    )
    fig.update_layout(height=500, showlegend=False)
    return fig


def plot_price_by_age(df):
    """Gráfico: Curva de depreciación (Precio vs Edad)."""
    price_by_age = df.groupby('age')['price'].agg(['mean', 'median', 'count']).reset_index()
    price_by_age = price_by_age[price_by_age['count'] > 100]  # Filtrar edades con pocos datos
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=price_by_age['age'],
        y=price_by_age['mean'],
        mode='lines+markers',
        name='Precio Promedio',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=price_by_age['age'],
        y=price_by_age['median'],
        mode='lines+markers',
        name='Precio Mediano',
        line=dict(color='#ff7f0e', width=2, dash='dash'),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title='Curva de Depreciación: Precio vs Edad del Vehículo',
        xaxis_title='Edad (años)',
        yaxis_title='Precio (USD)',
        height=400,
        hovermode='x unified'
    )
    
    return fig


def plot_price_distribution(df):
    """Gráfico: Distribución de precios."""
    fig = px.histogram(
        df,
        x='price',
        nbins=50,
        title='Distribución de Precios de Vehículos',
        labels={'price': 'Precio (USD)', 'count': 'Frecuencia'},
        color_discrete_sequence=['#2ca02c']
    )
    fig.update_layout(height=400)
    return fig


def plot_price_by_condition(df):
    """Gráfico: Precio por condición."""
    condition_order = ['salvage', 'fair', 'good', 'excellent', 'like new', 'new']
    df_filtered = df[df['condition'].isin(condition_order)]
    
    fig = px.box(
        df_filtered,
        x='condition',
        y='price',
        category_orders={'condition': condition_order},
        title='Distribución de Precios por Condición del Vehículo',
        labels={'condition': 'Condición', 'price': 'Precio (USD)'},
        color='condition',
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig.update_layout(height=400, showlegend=False)
    return fig


# ============================================================================
# INTERFAZ PRINCIPAL
# ============================================================================

def main():
    # Header
    st.title("🚗 Sistema de Tasación de Vehículos Usados")
    st.markdown("### Predicción de Precios con Machine Learning")
    
    # Verificar API
    if not check_api_health():
        st.error("❌ No se pudo conectar con la API")
        st.info(f"Asegúrate de que la API esté corriendo en: `{API_URL}`")
        st.code("uvicorn src.api.main:app --reload --port 8000")
        st.stop()
    
    # Obtener categorías y modelo info
    categories = get_categories()
    if not categories:
        st.error("No se pudieron obtener las categorías de la API")
        st.stop()
    
    model_info = get_model_info()
    if model_info:
        st.success(f"✅ API Conectada | Modelo: {model_info['model_type']} | R²: {model_info['metrics']['r2_test']:.4f}")
    
    # Sidebar con información
    with st.sidebar:
        st.header("📊 Información del Modelo")
        
        if model_info:
            metrics = model_info['metrics']
            st.metric("R² Score", f"{metrics['r2_test']:.4f}")
            st.metric("MAE", f"${metrics['mae_test']:,.2f}")
            st.metric("RMSE", f"${metrics['rmse_test']:,.2f}")
            
            st.markdown("---")
            st.markdown("**Features usadas:**")
            st.caption(f"• {model_info['total_features']} features totales")
            st.caption(f"• {len(model_info['categorical_features'])} categóricas")
            st.caption(f"• {len(model_info['numerical_features']) - 1} numéricas")
        
        st.markdown("---")
        st.markdown("### 🎯 Navegación")
        page = st.radio(
            "Selecciona una página:",
            ["🔍 Predicción", "📈 Análisis Exploratorio"],
            label_visibility="collapsed"
        )
    
    # ========================================================================
    # PÁGINA: PREDICCIÓN
    # ========================================================================
    
    if page == "🔍 Predicción":
        st.header("📝 Ingresa las Características del Vehículo")
        
        # Formulario en columnas
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Información Básica")
            manufacturer = st.selectbox(
                "🏭 Marca",
                options=categories["manufacturers"],
                index=categories["manufacturers"].index("toyota") if "toyota" in categories["manufacturers"] else 0
            )
            
            condition = st.selectbox(
                "⭐ Condición",
                options=categories["conditions"],
                index=2  # "good"
            )
            
            vehicle_type = st.selectbox(
                "🚙 Tipo de Vehículo",
                options=categories["types"],
                index=categories["types"].index("pickup") if "pickup" in categories["types"] else 0
            )
            
            state = st.selectbox(
                "📍 Estado",
                options=categories["states"],
                index=categories["states"].index("tx") if "tx" in categories["states"] else 0,
                help="Estado de Estados Unidos (código de 2 letras)"
            )
        
        with col2:
            st.subheader("Motor y Transmisión")
            cylinders = st.slider(
                "🔧 Cilindros",
                min_value=categories["cylinders_range"]["min"],
                max_value=categories["cylinders_range"]["max"],
                value=6
            )
            
            fuel = st.selectbox(
                "⛽ Combustible",
                options=categories["fuels"],
                index=categories["fuels"].index("gas") if "gas" in categories["fuels"] else 0
            )
            
            transmission = st.selectbox(
                "⚙️ Transmisión",
                options=categories["transmissions"],
                index=categories["transmissions"].index("automatic") if "automatic" in categories["transmissions"] else 0
            )
            
            drive = st.selectbox(
                "🛞 Tracción",
                options=categories["drives"],
                index=categories["drives"].index("4wd") if "4wd" in categories["drives"] else 0
            )
        
        with col3:
            st.subheader("Uso y Antigüedad")
            age = st.slider(
                "📅 Edad (años)",
                min_value=categories["age_range"]["min"],
                max_value=categories["age_range"]["max"],
                value=5,
                help="Antigüedad del vehículo en años"
            )
            
            odometer = st.number_input(
                "🛣️ Kilometraje (millas)",
                min_value=categories["odometer_range"]["min"],
                max_value=categories["odometer_range"]["max"],
                value=45000,
                step=1000,
                help="Kilometraje total del vehículo en millas"
            )
        
        st.markdown("---")
        
        # Botón de predicción centrado
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            predict_button = st.button(
                "🔍 PREDECIR PRECIO",
                type="primary",
                use_container_width=True
            )
        
        if predict_button:
            # Preparar datos
            vehicle_data = {
                "manufacturer": manufacturer,
                "condition": condition,
                "cylinders": cylinders,
                "fuel": fuel,
                "odometer": float(odometer),
                "transmission": transmission,
                "drive": drive,
                "type": vehicle_type,
                "state": state,
                "age": age
            }
            
            # Hacer predicción
            with st.spinner("🤖 Calculando precio..."):
                prediction = predict_price(vehicle_data)
            
            if prediction:
                st.success("✅ Predicción completada")
                
                # Resultado principal
                st.markdown("## 💰 Precio Estimado")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Precio Predicho",
                        prediction["predicted_price_formatted"],
                        help="Valor estimado por el modelo ML"
                    )
                
                with col2:
                    st.metric(
                        "Rango Inferior",
                        f"${prediction['confidence_interval_lower']:,.2f}",
                        help="Límite inferior del intervalo de confianza"
                    )
                
                with col3:
                    st.metric(
                        "Rango Superior",
                        f"${prediction['confidence_interval_upper']:,.2f}",
                        help="Límite superior del intervalo de confianza"
                    )
                
                # Gráfico de barras del intervalo
                st.markdown("### 📊 Intervalo de Confianza")
                
                fig = go.Figure()
                
                # Barra del intervalo completo
                fig.add_trace(go.Bar(
                    x=['Precio'],
                    y=[prediction['predicted_price']],
                    marker_color='#2ecc71',
                    name='Precio Predicho',
                    text=[prediction["predicted_price_formatted"]],
                    textposition='outside',
                    width=0.4
                ))
                
                # Error bars para el intervalo
                fig.add_trace(go.Scatter(
                    x=['Precio'],
                    y=[prediction['predicted_price']],
                    error_y=dict(
                        type='data',
                        symmetric=False,
                        array=[prediction['confidence_interval_upper'] - prediction['predicted_price']],
                        arrayminus=[prediction['predicted_price'] - prediction['confidence_interval_lower']],
                        color='rgba(255, 127, 14, 0.8)',
                        thickness=3,
                        width=15
                    ),
                    mode='markers',
                    marker=dict(size=12, color='#2ecc71'),
                    showlegend=False
                ))
                
                fig.update_layout(
                    height=350,
                    showlegend=False,
                    yaxis_title="Precio (USD)",
                    xaxis=dict(showticklabels=False)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Información adicional
                st.info(f"🤖 {prediction['model_version']} | Error promedio del modelo: $3,642")
                
                # Resumen del vehículo
                with st.expander("📋 Resumen del Vehículo"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Marca:** {manufacturer.title()}")
                        st.write(f"**Tipo:** {vehicle_type.title()}")
                        st.write(f"**Condición:** {condition.title()}")
                        st.write(f"**Cilindros:** {cylinders}")
                        st.write(f"**Combustible:** {fuel.title()}")
                    with col2:
                        st.write(f"**Transmisión:** {transmission.title()}")
                        st.write(f"**Tracción:** {drive.upper()}")
                        st.write(f"**Estado:** {state.upper()}")
                        st.write(f"**Edad:** {age} años")
                        st.write(f"**Kilometraje:** {odometer:,} millas")
    
    # ========================================================================
    # PÁGINA: ANÁLISIS EXPLORATORIO
    # ========================================================================
    
    elif page == "📈 Análisis Exploratorio":
        st.header("📊 Análisis Exploratorio de Datos (EDA)")
        st.markdown("Visualizaciones basadas en el dataset de ~250,000 vehículos usados")
        
        # Cargar datos
        df = load_data_for_eda()
        
        if df is None:
            st.warning("⚠️ No se pudieron cargar los datos para EDA")
            st.info("Asegúrate de que existe: `data/processed/vehicles_with_features.csv`")
            return
        
        # Métricas generales
        st.subheader("📈 Estadísticas Generales")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Vehículos", f"{len(df):,}")
        with col2:
            st.metric("Precio Promedio", f"${df['price'].mean():,.2f}")
        with col3:
            st.metric("Precio Mediano", f"${df['price'].median():,.2f}")
        with col4:
            st.metric("Marcas Únicas", f"{df['manufacturer'].nunique()}")
        
        st.markdown("---")
        
        # Gráficos en tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📉 Depreciación",
            "🏭 Precio por Marca",
            "📊 Distribución",
            "⭐ Condición"
        ])
        
        with tab1:
            st.markdown("### Curva de Depreciación")
            st.markdown("Cómo el precio disminuye con la edad del vehículo")
            fig = plot_price_by_age(df)
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("💡 **Insight:** Los vehículos pierden valor rápidamente en los primeros 5 años.")
        
        with tab2:
            st.markdown("### Precio Promedio por Marca")
            top_n = st.slider("Número de marcas a mostrar", 5, 25, 15, key="top_n_brands")
            fig = plot_price_by_manufacturer(df, top_n)
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("💡 **Insight:** Marcas de lujo como Ferrari, Tesla y Porsche tienen los precios más altos.")
        
        with tab3:
            st.markdown("### Distribución de Precios")
            fig = plot_price_distribution(df)
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Precio Mínimo", f"${df['price'].min():,.2f}")
                st.metric("Percentil 25", f"${df['price'].quantile(0.25):,.2f}")
            with col2:
                st.metric("Percentil 75", f"${df['price'].quantile(0.75):,.2f}")
                st.metric("Precio Máximo", f"${df['price'].max():,.2f}")
        
        with tab4:
            st.markdown("### Precio por Condición del Vehículo")
            fig = plot_price_by_condition(df)
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("💡 **Insight:** La condición del vehículo es un factor determinante en el precio.")


# ============================================================================
# EJECUTAR
# ============================================================================

if __name__ == "__main__":
    main()
