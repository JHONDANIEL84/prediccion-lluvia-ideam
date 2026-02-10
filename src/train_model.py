import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import matplotlib.pyplot as plt

def train_model(input_file='data/processed_rain_data.csv', model_file='models/rain_model.pkl'):
    """
    Entrena un modelo Random Forest para predecir precipitación.
    """
    print(f"Cargando datos preprocesados desde {input_file}...")
    df = pd.read_csv(input_file)
    
    # Definir características (X) y objetivo (y)
    # Excluir fecha y precipitacion actual de X geteando las columnas features
    features = [col for col in df.columns if col not in ['fecha', 'precipitacion']]
    target = 'precipitacion'
    
    X = df[features]
    y = df[target]
    
    # División Train/Test (Cronológica para series de tiempo)
    train_size = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    
    print(f"Datos de entrenamiento: {X_train.shape}")
    print(f"Datos de prueba: {X_test.shape}")
    
    # Entrenar modelo
    print("Entrenando Random Forest Regressor...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predecir
    y_pred = model.predict(X_test)
    
    # Evaluar
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")
    
    # Guardar modelo
    os.makedirs(os.path.dirname(model_file), exist_ok=True)
    joblib.dump(model, model_file)
    print(f"Modelo guardado en {model_file}")
    
    # Verificar importancia de características
    importances = model.feature_importances_
    feature_imp_df = pd.DataFrame({'feature': features, 'importance': importances}).sort_values('importance', ascending=False)
    print("\nImportancia de Características:")
    print(feature_imp_df)

if __name__ == "__main__":
    train_model()
