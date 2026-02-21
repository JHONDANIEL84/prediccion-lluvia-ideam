import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# -------------------------
# Rutas dinámicas correctas
# -------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "dataset_modelo_estacion_52045020.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "rain_model.pkl")

os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)

print("📥 Cargando dataset...")
df = pd.read_csv(DATA_PATH)

# Eliminar columnas que no sirven
df = df.drop(columns=["fecha", "estacion"], errors="ignore")

# Target
y = df["precip"]

# Features
X = df.drop(columns=["precip"])

# División
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("🌲 Entrenando modelo...")
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluación
y_pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))

# Guardar modelo
joblib.dump(model, MODEL_PATH)
print("✅ Modelo guardado en:", MODEL_PATH)