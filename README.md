# Laboratorio-5---Prog.-Microprocesadores
📘 Descripción general

Este repositorio contiene el desarrollo completo del Laboratorio 5, cuyo objetivo es aplicar técnicas de Machine Learning sobre los datos obtenidos desde un Arduino Nano 33 BLE Sense, integrando el flujo de trabajo con GitHub para el almacenamiento, versionamiento y ejecución automatizada del proyecto.

A partir del conjunto de datos adquirido en los laboratorios anteriores, se realiza un análisis exploratorio y un modelo de regresión lineal para predecir la temperatura a partir de la humedad relativa, demostrando el proceso completo desde la captura de datos hasta la evaluación del modelo.


🧩 Estructura del repositorio
Laboratorio-5---Prog.-Microprocesadores/

│

├── data/ # Contiene los datasets originales (CSV)

│ └── data.csv

│

├── scripts/ # Scripts de análisis y modelamiento

│ └── model_training.py

│

├── requirements.txt # Librerías necesarias para ejecutar el proyecto

├── config.yaml # Archivo de configuración del modelo

└── README.md # Documentación del proyecto



⚙️ Requisitos

Antes de ejecutar el proyecto, se deben instalar las librerías necesarias.
Ejecuta el siguiente comando en la terminal (VS Code o Anaconda Prompt):

pip install -r requirements.txt


Librerías incluidas:

pandas

numpy

matplotlib

seaborn

scikit-learn


🧠 Ejecución del modelo

Para ejecutar el análisis y el entrenamiento del modelo localmente:

python scripts/model_training.py


El script realiza las siguientes tareas:

Carga del dataset desde la carpeta /data o directamente desde GitHub.

Análisis exploratorio inicial (estadísticas, estructuras, valores faltantes).

Generación de gráficos de distribución y tendencias de los sensores.

Entrenamiento de un modelo de regresión lineal entre temperatura y humedad.

Evaluación del modelo mediante RMSE y R².

Visualización de resultados y comparaciones entre valores reales y predichos.


☁️ Ejecución automatizada (GitHub Actions)

El entrenamiento también puede realizarse de forma automática mediante GitHub Actions.
Para ello, se debe agregar un archivo en la carpeta .github/workflows/ con el siguiente nombre:

train_model.yml


Este flujo permitirá ejecutar el script de entrenamiento de manera remota (runner), asegurando la reproducibilidad del proyecto.


📊 Resultados esperados

El modelo genera las siguientes salidas gráficas:

Distribución de temperatura y humedad.

Comportamiento temporal de las principales variables.

Gráfico de regresión lineal (línea de ajuste y puntos reales).

Comparación entre valores reales y predichos.

En consola, se mostrarán las métricas:

Error cuadrático medio (RMSE): 0.1234
Coeficiente de determinación (R²): 0.9210
