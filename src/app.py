import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
from datetime import datetime

# Asegurar que Python encuentre los módulos en src/
# Añadimos el directorio actual al path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Importar funciones de los otros scripts para auto-entrenamiento
try:
    from src.data_loader import generate_synthetic_data
    from src.preprocessing import preprocess_data
    from src.train_model import train_model
except ImportError:
    # Fallback por si se ejecuta desde dentro de src/
    from data_loader import generate_synthetic_data
    from preprocessing import preprocess_data
    from train_model import train_model

# Configuración de la página
st.set_page_config(page_title="Predicción de Lluvia IDEAM", layout="wide")

# Función para cargar o entrenar el modelo
@st.cache_resource
def load_or_train_model():
    model_path = os.path.join(parent_dir, 'models', 'rain_model.pkl')
    data_path = os.path.join(parent_dir, 'data', 'rain_data.csv')
    
    model = None
    
    # Intentar cargar modelo existente
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            # Prueba rápida de predicción para asegurar compatibilidad
            # Si falla (ej. sklearn version mismatch), saltará al except
            model.predict([[1, 1, 25, 70, 1013, 0, 0, 0, 25, 25, 25, 70, 70, 70, 1013, 1013, 1013, 0]])
            return model
        except Exception as e:
            st.warning(f"⚠️ Modelo existente incompatible o corrupto ({e}). Re-entrenando...")
            model = None

    # Si no existe o falló la carga, entrenamos
    if model is None:
        st.info("🔄 Iniciando entrenamiento del modelo en el entorno nube...")
        
        try:
            with st.spinner('Generando datos sintéticos...'):
                generate_synthetic_data(output_file=os.path.join(parent_dir, 'data', 'rain_data.csv'))
                
            with st.spinner('Preprocesando datos...'):
                preprocess_data(input_file=os.path.join(parent_dir, 'data', 'rain_data.csv'),
                              output_file=os.path.join(parent_dir, 'data', 'processed_rain_data.csv'))
                
            with st.spinner('Entrenando modelo Random Forest...'):
                train_model(input_file=os.path.join(parent_dir, 'data', 'processed_rain_data.csv'),
                          model_file=model_path)
                
            model = joblib.load(model_path)
            st.success("✅ Modelo entrenado y cargado correctamente.")
        except Exception as e:
            st.error(f"❌ Error fatal entrenando el modelo: {e}")
            return None
            
    return model

try:
    model = load_or_train_model()
except Exception as e:
    st.error(f"Error crítico en validación inicial: {e}")
    model = None

def main():
    st.title("🌧️ Predicción de Precipitación (Tipo IDEAM)")
    st.markdown("""
    Esta aplicación utiliza un modelo de Machine Learning (Random Forest) entrenado con datos históricos 
    para estimar la cantidad de lluvia esperada en base a condiciones climáticas.
    """)
    
    # Checkbox para traducción automática warning
    st.info("ℹ️ Si ves un error 'removeChild', por favor DESACTIVA la traducción automática de tu navegador para esta página.")

    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Datos del Día")
        mes = st.slider("Mes", 1, 12, datetime.now().month)
        dia_anio = st.slider("Día del Año", 1, 365, datetime.now().timetuple().tm_yday)
        
        st.subheader("Condiciones Actuales")
        temperatura = st.number_input("Temperatura (°C)", value=25.0, step=0.1)
        humedad = st.number_input("Humedad (%)", value=70.0, min_value=0.0, max_value=100.0, step=0.1)
        presion = st.number_input("Presión (hPa)", value=1013.0, step=0.1)
        
        st.subheader("Histórico Reciente (Lags)")
        st.caption("Datos de los últimos 3 días")
        
        precip_lags = []
        temp_lags = []
        hum_lags = []
        pres_lags = []
        
        for i in range(1, 4):
            with st.expander(f"Día Anterior -{i}"):
                p = st.number_input(f"Lluvia (mm) día -{i}", value=5.0 if i==1 else 0.0, key=f"p_{i}")
                t = st.number_input(f"Temp (°C) día -{i}", value=25.0, key=f"t_{i}")
                h = st.number_input(f"Hum (%) día -{i}", value=70.0, key=f"h_{i}")
                pr = st.number_input(f"Pres (hPa) día -{i}", value=1013.0, key=f"pr_{i}")
                
                precip_lags.append(p)
                temp_lags.append(t)
                hum_lags.append(h)
                pres_lags.append(pr)

    with col2:
        if model is None:
            st.error("⚠️ El modelo no está disponible.")
        else:
            if st.button("☂️ Predecir Lluvia", type="primary"):
                # Calcular roll mean
                roll_mean_7 = np.mean(precip_lags + [0]*4) 
                
                input_data = {
                    'mes': mes,
                    'dia_anio': dia_anio,
                    'temperatura': temperatura,
                    'humedad': humedad,
                    'presion': presion,
                    
                    'precipitacion_lag1': precip_lags[0],
                    'precipitacion_lag2': precip_lags[1],
                    'precipitacion_lag3': precip_lags[2],
                    
                    'temperatura_lag1': temp_lags[0],
                    'temperatura_lag2': temp_lags[1],
                    'temperatura_lag3': temp_lags[2],
                    
                    'humedad_lag1': hum_lags[0],
                    'humedad_lag2': hum_lags[1],
                    'humedad_lag3': hum_lags[2],
                    
                    'presion_lag1': pres_lags[0],
                    'presion_lag2': pres_lags[1],
                    'presion_lag3': pres_lags[2],
                    
                    'precipitacion_roll_mean_7': roll_mean_7
                }
                
                try:
                    features_df = pd.DataFrame([input_data])
                    # Reordenar columnas asumiendo estructura de entrenamiento
                    # Cargamos una muestra dummy para ver columnas si es posible, sino usamos el orden hardcodeado
                    # Para robustez, usamos las columnas esperadas hardcodeadas si falla la lectura
                    expected_cols = [
                        'mes', 'dia_anio', 'temperatura', 'humedad', 'presion',
                        'precipitacion_lag1', 'precipitacion_lag2', 'precipitacion_lag3',
                        'temperatura_lag1', 'temperatura_lag2', 'temperatura_lag3',
                        'humedad_lag1', 'humedad_lag2', 'humedad_lag3',
                        'presion_lag1', 'presion_lag2', 'presion_lag3',
                        'precipitacion_roll_mean_7'
                    ]
                    
                    # Asegurar que todas esten
                    for col in expected_cols:
                        if col not in features_df.columns:
                            features_df[col] = 0
                            
                    features_df = features_df[expected_cols]
                    
                    prediction = model.predict(features_df)[0]
                    
                    st.success(f"Predicción Completada")
                    st.metric(label="Precipitación Estimada", value=f"{prediction:.2f} mm")
                    
                    if prediction < 0.5:
                        st.info("☀️ Probabilidad de lluvia baja (Tiempo Seco)")
                    elif prediction < 10:
                        st.warning("🌧️ Lluvia Ligera a Moderada")
                    else:
                        st.error("⛈️ Lluvia Fuerte Esperada")
                        
                    # Visualización
                    st.subheader("Análisis de Entrada")
                    chart_data = pd.DataFrame({
                        'Variable': ['Humedad', 'Temperatura'],
                        'Valor': [humedad, temperatura]
                    })
                    st.bar_chart(chart_data.set_index('Variable'))
                    
                except Exception as e:
                    st.error(f"Error en la predicción: {e}")

if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

# Configuración de la página
st.set_page_config(page_title="Predicción de Lluvia IDEAM", layout="wide")

# Función para cargar el modelo
@st.cache_resource
def load_model():
    model_path = 'models/rain_model.pkl'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_model()

def main():
    st.title("🌧️ Predicción de Precipitación (Tipo IDEAM)")
    st.markdown("""
    Esta aplicación utiliza un modelo de Machine Learning (Random Forest) entrenado con datos históricos 
    para estimar la cantidad de lluvia esperada en base a condiciones climáticas.
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Datos del Día")
        mes = st.slider("Mes", 1, 12, datetime.now().month)
        dia_anio = st.slider("Día del Año", 1, 365, datetime.now().timetuple().tm_yday)
        
        st.subheader("Condiciones Actuales")
        temperatura = st.number_input("Temperatura (°C)", value=25.0, step=0.1)
        humedad = st.number_input("Humedad (%)", value=70.0, min_value=0.0, max_value=100.0, step=0.1)
        presion = st.number_input("Presión (hPa)", value=1013.0, step=0.1)
        
        st.subheader("Histórico Reciente (Lags)")
        st.caption("Datos de los últimos 3 días")
        
        precip_lags = []
        temp_lags = []
        hum_lags = []
        pres_lags = []
        
        for i in range(1, 4):
            with st.expander(f"Día Anterior -{i}"):
                p = st.number_input(f"Lluvia (mm) día -{i}", value=5.0 if i==1 else 0.0, key=f"p_{i}")
                t = st.number_input(f"Temp (°C) día -{i}", value=25.0, key=f"t_{i}")
                h = st.number_input(f"Hum (%) día -{i}", value=70.0, key=f"h_{i}")
                pr = st.number_input(f"Pres (hPa) día -{i}", value=1013.0, key=f"pr_{i}")
                
                precip_lags.append(p)
                temp_lags.append(t)
                hum_lags.append(h)
                pres_lags.append(pr)

    with col2:
        if model is None:
            st.error("⚠️ No se encontró el modelo `models/rain_model.pkl`. Por favor entrena el modelo primero ejecutando `python src/train_model.py`.")
        else:
            if st.button("☂️ Predecir Lluvia", type="primary"):
                # Preparar datos para el modelo
                # Necesitamos construir un DF con las mismas columnas que usó el modelo
                # Orden: mes, dia_anio, precip_lag1..3, temp_lag1..3, hum_lag1..3, pres_lag1..3, roll_mean
                
                # Calcular roll mean de los lags de precipitacion ingresados
                roll_mean_7 = np.mean(precip_lags + [0]*4) # Aproximacion simple usando los 3 dias conocidos
                
                input_data = {
                    'mes': mes,
                    'dia_anio': dia_anio,
                    'temperatura': temperatura,
                    'humedad': humedad,
                    'presion': presion, # Asumiendo que train_model usó features actuales (revisar abajo)
                    # NOTA: En train_model.py, usamos df[features]. 
                    # Features eran todas MENOS fecha y precipitacion target.
                    # El df original tenía 'temperatura', 'humedad', 'presion' actuales.
                    # Así que SÍ se usan.
                    
                    'precipitacion_lag1': precip_lags[0],
                    'precipitacion_lag2': precip_lags[1],
                    'precipitacion_lag3': precip_lags[2],
                    
                    'temperatura_lag1': temp_lags[0],
                    'temperatura_lag2': temp_lags[1],
                    'temperatura_lag3': temp_lags[2],
                    
                    'humedad_lag1': hum_lags[0],
                    'humedad_lag2': hum_lags[1],
                    'humedad_lag3': hum_lags[2],
                    
                    'presion_lag1': pres_lags[0],
                    'presion_lag2': pres_lags[1],
                    'presion_lag3': pres_lags[2],
                    
                    'precipitacion_roll_mean_7': roll_mean_7
                }
                
                # Crear DF
                # Intentar cargar columnas de entrenamiento para asegurar orden
                try:
                    features_df = pd.DataFrame([input_data])
                    # Reordenar columnas si podemos leer el csv procesado
                    if os.path.exists('data/processed_rain_data.csv'):
                        sample = pd.read_csv('data/processed_rain_data.csv', nrows=1)
                        train_cols = [c for c in sample.columns if c not in ['fecha', 'precipitacion']]
                        # Rellenar faltantes
                        for c in train_cols:
                            if c not in features_df.columns:
                                features_df[c] = 0
                        features_df = features_df[train_cols]
                    
                    prediction = model.predict(features_df)[0]
                    
                    st.success(f"Predicción Completada")
                    st.metric(label="Precipitación Estimada", value=f"{prediction:.2f} mm")
                    
                    if prediction < 0.5:
                        st.info("☀️ Probabilidad de lluvia baja (Tiempo Seco)")
                    elif prediction < 10:
                        st.warning("🌧️ Lluvia Ligera a Moderada")
                    else:
                        st.error("⛈️ Lluvia Fuerte Esperada")
                        
                    # Visualización simple de los datos de entrada
                    st.subheader("Análisis de Entrada")
                    chart_data = pd.DataFrame({
                        'Variable': ['Humedad', 'Temperatura'],
                        'Valor': [humedad, temperatura]
                    })
                    st.bar_chart(chart_data.set_index('Variable'))
                    
                except Exception as e:
                    st.error(f"Error en la predicción: {e}")

if __name__ == "__main__":
    main()
