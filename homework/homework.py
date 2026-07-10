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
import gzip
import json
import os
import pickle

import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


TRAIN_PATH = "files/input/train_data.csv"
TEST_PATH = "files/input/test_data.csv"

CURRENT_YEAR = 2021

CATEGORICAL_FEATURES = [
    "Fuel_Type",
    "Selling_type",
    "Transmission","Owner"
]


def load_and_clean(path):

    zip_path = path + ".zip"

    if os.path.exists(zip_path):
        df = pd.read_csv(zip_path, compression="zip")
    else:
        df = pd.read_csv(path)

    df["Age"] = CURRENT_YEAR - df["Year"]

    df = df.drop(columns=["Year", "Car_Name"])

    return df


train_dataset = load_and_clean(TRAIN_PATH)
test_dataset = load_and_clean(TEST_PATH)

x_train = train_dataset.drop(columns=["Present_Price"])
y_train = train_dataset["Present_Price"]

x_test = test_dataset.drop(columns=["Present_Price"])
y_test = test_dataset["Present_Price"]


numerical_features = [
    col for col in x_train.columns
    if col not in CATEGORICAL_FEATURES
]


preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            CATEGORICAL_FEATURES,
        ),
        (
            "num",
            MinMaxScaler(feature_range=(0, 1)),
            numerical_features,
        ),
    ]
)


pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("feature_selection", SelectKBest(score_func=f_regression)),
        ("classifier", LinearRegression()),
    ]
)


n_features_transformed = clone(preprocessor).fit_transform(x_train).shape[1]

param_grid = {
    "feature_selection__k": range(1, n_features_transformed + 1),
}


model = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=10,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
    refit=True,
    verbose=3,
)

model.fit(x_train, y_train)


os.makedirs("files/models", exist_ok=True)

with gzip.open("files/models/model.pkl.gz", "wb") as file:
    pickle.dump(model, file)
from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error

def compute_metrics(name, y_true, y_pred):

    return {
        "type": "metrics",
        "dataset": name,
        "r2": float(r2_score(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "mad": float(mean_absolute_error(y_true, y_pred)),
    }


train_pred = model.predict(x_train)
test_pred = model.predict(x_test)

metrics = [
    compute_metrics("train", y_train, train_pred),
    compute_metrics("test", y_test, test_pred),
]


os.makedirs("files/output", exist_ok=True)

with open("files/output/metrics.json", "w", encoding="utf-8") as file:
    for metric in metrics:
        file.write(json.dumps(metric) + "\n")


print(model.best_params_)
print(model.best_score_)
print(metrics)