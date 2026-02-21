import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Predicción IDEAM", layout="wide")

MODEL_PATH = "models/rain_model.pkl"

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

model = load_model()

st.title("🌧️ Predicción de Precipitación")

if model is None:
    st.error("❌ No se encontró el modelo. Ejecuta primero: python src/train_model.py")
else:

    lag1 = st.number_input("Lluvia día -1", value=0.0)
    lag2 = st.number_input("Lluvia día -2", value=0.0)
    lag3 = st.number_input("Lluvia día -3", value=0.0)
    mm3 = st.number_input("Promedio 3 días", value=0.0)
    mm7 = st.number_input("Promedio 7 días", value=0.0)
    mes = st.slider("Mes", 1, 12, 1)
    extremo = st.selectbox("Evento extremo previo", [0, 1])

    if st.button("Predecir"):

        input_data = pd.DataFrame([{
            "lag1": lag1,
            "lag2": lag2,
            "lag3": lag3,
            "mm3": mm3,
            "mm7": mm7,
            "mes": mes,
            "extremo": extremo
        }])

        prediction = model.predict(input_data)[0]

        st.success(f"🌧️ Precipitación estimada: {prediction:.2f} mm")