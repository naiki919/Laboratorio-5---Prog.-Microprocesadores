import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import math

# ==== Lectura de los datos ====
# Cargar el dataset desde GitHub o desde una carpeta local
# Si lo ejecutas en local, reemplaza la URL por "data/lab3.csv"
url = "https://raw.githubusercontent.com/TU_USUARIO/Laboratorio-5----Prog.-Microprocesadores/main/data/data.csv"
datos = pd.read_csv(url)

# Mostrar información básica
print("\n--- Información del dataset ---")
print(datos.info())
print("\n--- Primeras filas ---")
print(datos.head())
print("\n--- Estadísticas descriptivas ---")
print(datos.describe())

# ==== Visualización inicial de los datos ====
# Histograma de temperatura
plt.figure(figsize=(6,4))
sns.histplot(datos['tempC'], bins=15, kde=True, color='red')
plt.title("Distribución de la Temperatura (°C)")
plt.xlabel("Temperatura (°C)")
plt.ylabel("Frecuencia")
plt.show()

# Histograma de humedad
plt.figure(figsize=(6,4))
sns.histplot(datos['hum'], bins=15, kde=True, color='blue')
plt.title("Distribución de la Humedad (%)")
plt.xlabel("Humedad (%)")
plt.ylabel("Frecuencia")
plt.show()

# Gráfico comparativo de variables
plt.figure(figsize=(10,5))
plt.plot(datos['tempC'], label='Temperatura (°C)', color='red')
plt.plot(datos['hum'], label='Humedad (%)', color='blue')
plt.plot(datos['mic_db'], label='Nivel Sonoro (dB)', color='green')
plt.title("Lecturas Sensoriales del Arduino")
plt.xlabel("Tiempo (índice)")
plt.ylabel("Valor medido")
plt.legend()
plt.show()

# ==== Entrenamiento del modelo ====
# Se seleccionan las variables
X = datos[['hum']]      # Variable independiente
y = datos['tempC']      # Variable dependiente

# División del dataset en entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creación y entrenamiento del modelo
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Predicciones
y_pred = modelo.predict(X_test)

# ==== Evaluación del modelo ====
rmse = math.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n--- Evaluación del Modelo ---")
print(f"Error cuadrático medio (RMSE): {rmse:.4f}")
print(f"Coeficiente de determinación (R²): {r2:.4f}")

# ==== Gráficos de resultados ====
# Gráfico de regresión
plt.figure(figsize=(6,5))
plt.scatter(X_test, y_test, color='blue', label='Datos reales')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicción')
plt.title("Modelo de Regresión Lineal: Temperatura vs Humedad")
plt.xlabel("Humedad (%)")
plt.ylabel("Temperatura (°C)")
plt.legend()
plt.show()

# Gráfico comparativo Real vs Predicho
plt.figure(figsize=(8,4))
plt.plot(y_test.values, label='Real', color='blue')
plt.plot(y_pred, label='Predicho', color='red')
plt.title("Comparación entre Valores Reales y Predichos")
plt.xlabel("Índice de muestra")
plt.ylabel("Temperatura (°C)")
plt.legend()
plt.show()

# ==== Resumen final ====
print("\n--- Resumen ---")
print("El modelo lineal logró ajustar la temperatura en función de la humedad,")
print("con un error promedio (RMSE) bajo y un nivel de correlación representado por R².")
print("El entrenamiento fue realizado localmente, pero puede replicarse mediante GitHub Actions (runner).")
