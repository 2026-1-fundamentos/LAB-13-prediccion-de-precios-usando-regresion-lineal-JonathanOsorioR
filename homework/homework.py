#
# En este dataset se desea pronosticar el precio de vhiculos usados. El dataset
# original contiene las siguientes columnas:
#
# - Car_Name: Nombre del vehiculo.
# - Year: Año de fabricación.
# - Selling_Price: Precio de venta.
# - Present_Price: Precio actual.
# - Driven_Kms: Kilometraje recorrido.
# - Fuel_type: Tipo de combustible.
# - Selling_Type: Tipo de vendedor.
# - Transmission: Tipo de transmisión.
# - Owner: Número de propietarios.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# pronostico están descritos a continuación.
#
#
# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json. Cada fila del archivo es un diccionario con
# las metricas de un modelo. Este diccionario tiene un campo para indicar
# si es el conjunto de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'r2': 0.8, 'mse': 0.7, 'mad': 0.9}
# {'type': 'metrics', 'dataset': 'test', 'r2': 0.7, 'mse': 0.6, 'mad': 0.8}
#
"""Homework solution: used car price prediction."""

import gzip
import json
import os
import pickle
 
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
 
# Si en su entorno los archivos tienen otro nombre, ajuste unicamente
# estas dos rutas.
TRAIN_PATH = "files/input/train_data.csv"
TEST_PATH = "files/input/test_data.csv"
 
CURRENT_YEAR = 2021
CATEGORICAL_FEATURES = ["Fuel_Type", "Selling_type", "Transmission"]
 
 
# ------------------------------------------------------------------------------
# Paso 1.
# Preprocesamiento de los datos.
# ------------------------------------------------------------------------------
def load_and_clean(path):
    """Carga un csv (o su variante comprimida path + '.zip') y aplica el
    preprocesamiento del Paso 1: crea 'Age' y elimina 'Year' y 'Car_Name'.
    """
    zip_path = path + ".zip"
    if os.path.exists(zip_path):
        df = pd.read_csv(zip_path, index_col=False, compression="zip")
    else:
        df = pd.read_csv(path, index_col=False)
 
    df = df.copy()
    df["Age"] = CURRENT_YEAR - df["Year"]
    df = df.drop(columns=["Year", "Car_Name"])
    return df
 
 
train_dataset = load_and_clean(TRAIN_PATH)
test_dataset = load_and_clean(TEST_PATH)

# ------------------------------------------------------------------------------
# Paso 2.
# División en x_train, y_train, x_test, y_test.
# ------------------------------------------------------------------------------
x_train = train_dataset.drop(columns=["Selling_Price"])
y_train = train_dataset["Selling_Price"]
 
x_test = test_dataset.drop(columns=["Selling_Price"])
y_test = test_dataset["Selling_Price"]
 
# ------------------------------------------------------------------------------
# Paso 3.
# Pipeline: one-hot-encoding (categóricas) + escalado [0, 1] (numéricas)
# + selección de las k mejores variables + regresión lineal.
# ------------------------------------------------------------------------------
numerical_features = [
    col for col in x_train.columns if col not in CATEGORICAL_FEATURES
]
 
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ("num", MinMaxScaler(feature_range=(0, 1)), numerical_features),
    ],
)
 
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("feature_selection", SelectKBest(score_func=f_regression)),
        ("classifier", LinearRegression()),
    ],
)
 
# ------------------------------------------------------------------------------
# Paso 4.
# Optimización de hiperparámetros mediante validación cruzada (10 splits),
# usando el error medio absoluto (MAE) como métrica de desempeño.
# ------------------------------------------------------------------------------
# Número de columnas resultantes tras el encoding + escalado: acota el
# rango válido de "k" para SelectKBest. Se calcula dinámicamente para no
# depender de cuántas categorías tenga cada variable categórica.
n_features_transformed = clone(preprocessor).fit_transform(x_train).shape[1]
 
param_grid = {
    "feature_selection__k": range(1, n_features_transformed + 1),
}
 
model = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=10,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
    refit=True,
    verbose=3
)
 
model.fit(x_train, y_train)
 
# ------------------------------------------------------------------------------
# Paso 5.
# Guardar el modelo comprimido con gzip.
# ------------------------------------------------------------------------------
os.makedirs("files/models", exist_ok=True)
 
with gzip.open("files/models/model.pkl.gz", "wb") as file:
    pickle.dump(model, file)
 
# ------------------------------------------------------------------------------
# Paso 6.
# Cálculo de métricas (r2, error cuadrático medio, error absoluto medio)
# para los conjuntos de entrenamiento y prueba.
# ------------------------------------------------------------------------------
def _compute_metrics(dataset_name, y_true, y_pred):
    return {
        "type": "metrics",
        "dataset": dataset_name,
        "r2": float(r2_score(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "mad": float(mean_absolute_error(y_true, y_pred)),
    }
 
 
y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)
 
metrics = [
    _compute_metrics("train", y_train, y_train_pred),
    _compute_metrics("test", y_test, y_test_pred),
]
 
os.makedirs("files/output", exist_ok=True)
 
with open("files/output/metrics.json", "w", encoding="utf-8") as file:
    for record in metrics:
        file.write(json.dumps(record) + "\n")
 
print(f"Mejor k (SelectKBest): {model.best_params_['feature_selection__k']}")
for record in metrics:
    print(record)