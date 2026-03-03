import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
import streamlit as st

st.set_page_config(page_title="Predicción de Lluvia IDEAM 🌧️", layout="centered")
st.title("Predicción de Lluvia IDEAM 🌧️")

# -------------------------------
# 1️⃣ Cargar modelo entrenado
# -------------------------------
try:
    with open("models/rain_model.pkl", "rb") as f:
        modelo_cargado = pickle.load(f)
    st.success("✅ Modelo cargado correctamente")
except FileNotFoundError:
    st.error("❌ No se encontró el archivo 'rain_model.pkl'. Asegúrate de que esté en la carpeta 'models'.")
    st.stop()
except Exception as e:
    st.error(f"❌ Error cargando el modelo: {e}")
    st.stop()

# -------------------------------
# 2️⃣ Cargar dataset para predicción
# -------------------------------
try:
    df = pd.read_csv("data/dataset_modelo_estacion_52045020.csv")
    st.write("Dataset cargado correctamente. Primeras filas:")
    st.dataframe(df.head())
except FileNotFoundError:
    st.error("❌ No se encontró el archivo CSV. Asegúrate de que esté en 'data/dataset_modelo_estacion_52045020.csv'.")
    st.stop()
except Exception as e:
    st.error(f"❌ Error cargando el dataset: {e}")
    st.stop()

# -------------------------------
# 3️⃣ Seleccionar columnas correctas
# -------------------------------
columnas_modelo = ["lag1", "lag2", "lag3", "mm3", "mm7", "mes", "extremo"]
if not all(col in df.columns for col in columnas_modelo):
    st.error(f"❌ Las columnas esperadas {columnas_modelo} no están en el dataset")
    st.stop()

X_nuevo = df[columnas_modelo].iloc[:5]  # primeras 5 filas como ejemplo

# -------------------------------
# 4️⃣ Escalar datos
# -------------------------------
scaler = StandardScaler()
X_nuevo_scaled = scaler.fit_transform(X_nuevo)

# -------------------------------
# 5️⃣ Hacer predicciones automáticas
# -------------------------------
try:
    predicciones = modelo_cargado.predict(X_nuevo_scaled)
    st.subheader("Predicciones automáticas del dataset:")
    for i, pred in enumerate(predicciones):
        st.write(f"Fila {i+1}: Predicción lluvia = {pred}")
except Exception as e:
    st.error(f"❌ Error al hacer la predicción: {e}")

# -------------------------------
# 6️⃣ Predicción manual por usuario
# -------------------------------
st.subheader("Predicción manual")
st.write("Ingresa los valores para las 7 características:")

lag1 = st.number_input("lag1", value=float(df['lag1'].mean()))
lag2 = st.number_input("lag2", value=float(df['lag2'].mean()))
lag3 = st.number_input("lag3", value=float(df['lag3'].mean()))
mm3 = st.number_input("mm3", value=float(df['mm3'].mean()))
mm7 = st.number_input("mm7", value=float(df['mm7'].mean()))
mes = st.number_input("mes", value=int(df['mes'].mode()[0]), min_value=1, max_value=12)
extremo = st.selectbox("extremo (0=No, 1=Sí)", options=[0, 1], index=int(df['extremo'].mode()[0]))

if st.button("Predecir lluvia"):
    try:
        X_manual = [[lag1, lag2, lag3, mm3, mm7, mes, extremo]]
        X_manual_scaled = scaler.fit_transform(X_manual)  # ⚠️ si guardaste scaler, usarlo
        pred_manual = modelo_cargado.predict(X_manual_scaled)[0]
        st.success(f"🌧️ Predicción de lluvia: {pred_manual}")
    except Exception as e:
        st.error(f"❌ Error al predecir manualmente: {e}")