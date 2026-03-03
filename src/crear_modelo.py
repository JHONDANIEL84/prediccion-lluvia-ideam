# crear_modelo.py
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
import pickle
import os

# Ruta donde guardar el modelo
MODEL_PATH = os.path.join("models", "rain_model.pkl")
os.makedirs("models", exist_ok=True)

# Cargar dataset real
df = pd.read_csv("data/dataset_modelo_estacion_52045020.csv")

# Preparar datos
df = df.drop(columns=["fecha", "estacion"], errors="ignore")
df["lluvia_binaria"] = (df["precip"] > 0).astype(int)
y = df["lluvia_binaria"]
X = df.drop(columns=["precip", "lluvia_binaria"])

# Escalar datos
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Entrenar modelo
modelo = SGDClassifier(loss="log_loss", max_iter=1000, random_state=42)
modelo.fit(X, y)

# Guardar con pickle
with open(MODEL_PATH, "wb") as f:
    pickle.dump(modelo, f)

print("✅ Modelo entrenado y guardado correctamente en 'models/rain_model.pkl'")