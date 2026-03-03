# app.py
import streamlit as st
import mlflow
import pickle  # si tu modelo está guardado como .pkl
import os

# --------------------------
# Configuración de MLflow
# --------------------------
MLRUNS_DIR = "./mlruns"
mlflow.set_tracking_uri(MLRUNS_DIR)
mlflow.set_experiment("prediccion_lluvia")

# --------------------------
# Cargar modelo
# --------------------------
# Ajusta la ruta según donde tengas tu modelo
MODELO_PATH = "modelo_lluvia.pkl"

if os.path.exists(MODELO_PATH):
    with open(MODELO_PATH, "rb") as f:
        modelo = pickle.load(f)
else:
    st.warning("Modelo no encontrado. Sube tu modelo .pkl")
    modelo = None

# --------------------------
# Función para registrar predicciones
# --------------------------
def registrar_prediccion(input_data: dict, pred: float):
    """
    Guarda la predicción en MLflow
    """
    with mlflow.start_run():
        mlflow.log_params(input_data)  # registra las variables de entrada
        mlflow.log_metric("prediccion_lluvia_mm", pred)  # registra la predicción

# --------------------------
# Interfaz de Streamlit
# --------------------------
st.title("Predicción de lluvia 🌧️")

# Inputs de usuario
temp = st.number_input("Temperatura (°C)", min_value=-50.0, max_value=50.0, value=25.0)
hum = st.number_input("Humedad (%)", min_value=0.0, max_value=100.0, value=80.0)
viento = st.number_input("Velocidad del viento (km/h)", min_value=0.0, max_value=200.0, value=10.0)

if st.button("Predecir lluvia"):

    if modelo:
        # Preparar datos
        input_data = {"temp": temp, "hum": hum, "viento": viento}
        # Hacer predicción
        pred = modelo.predict([list(input_data.values())])[0]
        st.success(f"Predicción de lluvia: {pred:.2f} mm")
        
        # Guardar en MLflow
        registrar_prediccion(input_data, pred)
        st.info("Predicción registrada en MLflow ✅")
    else:
        st.error("No hay modelo cargado para hacer predicción")