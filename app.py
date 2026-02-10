import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
from datetime import datetime

# Añadir src al path para poder importar módulos
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.append(src_dir)

try:
    from src.data_loader import generate_synthetic_data
    from src.preprocessing import preprocess_data
    from src.train_model import train_model
except ImportError:
    # Si falla la importación directa, intentamos añadir root al path
    sys.path.append(current_dir)
    from src.data_loader import generate_synthetic_data
    from src.preprocessing import preprocess_data
    from src.train_model import train_model

# Configuración de la página
st.set_page_config(page_title="Predicción de Lluvia IDEAM", layout="wide")

# Función para cargar o entrenar el modelo (Robustez para Cloud)
@st.cache_resource
def load_or_train_model():
    # Rutas relativas a la raiz
    model_dir = os.path.join(current_dir, 'models')
    data_dir = os.path.join(current_dir, 'data')
    
    model_path = os.path.join(model_dir, 'rain_model.pkl')
    
    # Asegurar directorios
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    model = None
    
    try:
        if os.path.exists(model_path):
            with st.spinner('Cargando modelo...'):
                model = joblib.load(model_path)
                # Test simple
                model.predict([[1]*18])
                return model
    except Exception as e:
        st.warning(f"Modelo no compatible o corrupto: {e}. Re-entrenando...")
        model = None

    if model is None:
        st.info("Entrenando modelo inicial en el servidor...")
        try:
            # Full pipeline
            raw_data = os.path.join(data_dir, 'rain_data.csv')
            processed_data = os.path.join(data_dir, 'processed_rain_data.csv')
            
            generate_synthetic_data(output_file=raw_data)
            preprocess_data(input_file=raw_data, output_file=processed_data)
            train_model(input_file=processed_data, model_file=model_path)
            
            model = joblib.load(model_path)
            st.success("Modelo entrenado exitosamente.")
        except Exception as e:
            st.error(f"Error entrenando modelo: {e}")
            return None
            
    return model

# Cargar modelo
model = load_or_train_model()

def main():
    st.title("🌧️ Predicción de Precipitación (Tipo IDEAM)")
    
    if model is None:
        st.error("No se pudo cargar el modelo. Por favor revisa los logs.")
        return

    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Parámetros")
        mes = st.slider("Mes", 1, 12, datetime.now().month)
        dia_anio = st.slider("Día del Año", 1, 365, datetime.now().timetuple().tm_yday)
        
        temperatura = st.number_input("Temperatura (°C)", value=25.0)
        humedad = st.number_input("Humedad (%)", value=70.0)
        presion = st.number_input("Presión (hPa)", value=1013.0)
        
        st.markdown("---")
        st.write("Datos históricos (ayer, antier...)")
        p_lag1 = st.number_input("Lluvia Ayer (mm)", value=5.0)
        
    with col2:
        st.subheader("Predicción")
        if st.button("Calcular Probabilidad", type="primary"):
            # Simplificacion para demo: rellenar lags con valores coherentes
            # Input vector must match training features exactly (18 features)
            # Orden asumido: mes, dia_anio, temp, hum, pres, 
            # p_lag1, p_lag2, p_lag3, 
            # t_lag1, t_lag2, t_lag3, 
            # h_lag1, h_lag2, h_lag3, 
            # pr_lag1, pr_lag2, pr_lag3, 
            # roll_mean
            
            # Construimos vector
            input_features = [
                mes, dia_anio, temperatura, humedad, presion,
                p_lag1, 0, 0, # lags lluvia
                temperatura, temperatura, temperatura, # lags temp (asumimos constancia para simplificar UX)
                humedad, humedad, humedad,
                presion, presion, presion,
                p_lag1 # roll mean aprox
            ]
            
            try:
                pred = model.predict([input_features])[0]
                st.metric("Lluvia Esperada", f"{pred:.2f} mm")
                
                if pred > 1.0:
                    st.warning("🌧️ Probabilidad de lluvia alta")
                else:
                    st.success("☀️ Tiempo mayormente seco")
            except Exception as e:
                st.error(f"Error en predicción: {e}")
                st.write("Verifica que las características coincidan con el entrenamiento.")

if __name__ == "__main__":
    main()
