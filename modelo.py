
import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

BASE = Path(__file__).resolve().parent
ARCHIVO = BASE / "alumnos.csv"
MODELO = BASE / "modelo_knn.pkl"
ESCALADOR = BASE / "scaler.pkl"
DATOS = BASE / "datos_entrenamiento.pkl"

COLUMNAS = ["horas", "asistencia", "tareas"]


def entrenar_modelo_knn(n_neighbors=5):
    # Leer el dataset
    df = pd.read_csv(ARCHIVO)

    # Revisar que tenga las columnas necesarias
    requeridas = COLUMNAS + ["resultado"]

    if not all(col in df.columns for col in requeridas):
        raise ValueError(
            "El CSV necesita: horas, asistencia, tareas, resultado"
        )

    df = df[requeridas].copy()

    # Convertir los valores a numeros
    for col in requeridas:
        df[col] = pd.to_numeric(df[col], errors="raise")

    if df.isnull().any().any():
        raise ValueError("El dataset contiene valores vacios.")

    if not df["resultado"].isin([0, 1]).all():
        raise ValueError("Resultado debe ser 0 o 1.")

    if not df["horas"].between(0, 10).all():
        raise ValueError("Horas debe estar entre 0 y 10.")

    for col in ["asistencia", "tareas"]:
        if not df[col].between(0, 100).all():
            raise ValueError(f"{col} debe estar entre 0 y 100.")

    X = df[COLUMNAS]
    y = df["resultado"].astype(int)

    if y.nunique() != 2:
        raise ValueError(
            "Se necesitan alumnos aprobados y reprobados."
        )

    if y.value_counts().min() < 2:
        raise ValueError(
            "Se necesitan al menos 2 ejemplos de cada clase."
        )

    # Separar 80% entrenamiento y 20% prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    if len(X_train) < n_neighbors:
        raise ValueError(
            "No hay suficientes alumnos para ese numero de vecinos."
        )

    # Normalizar los datos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Entrenar KNN
    modelo = KNeighborsClassifier(
        n_neighbors=n_neighbors
    )

    modelo.fit(X_train_scaled, y_train)

    # Evaluar con alumnos no vistos
    y_pred = modelo.predict(X_test_scaled)

    exactitud = accuracy_score(y_test, y_pred)

    matriz = confusion_matrix(
        y_test, y_pred, labels=[0, 1]
    )

    # Guardar el modelo y los datos necesarios
    joblib.dump(modelo, MODELO)
    joblib.dump(scaler, ESCALADOR)

    datos = X_train.copy()
    datos["resultado"] = y_train.to_numpy()

    joblib.dump(datos.reset_index(drop=True), DATOS)

    return exactitud, matriz, len(X_train), len(X_test)


def predecir_alumno(horas, asistencia, tareas):

    if asistencia < 80:
        return 0, 100.0

    
    modelo = joblib.load(MODELO)
    scaler = joblib.load(ESCALADOR)

    alumno = pd.DataFrame(
        [[horas, asistencia, tareas]],
        columns=COLUMNAS
    )

    alumno_escalado = scaler.transform(alumno)

    prediccion = int(modelo.predict(alumno_escalado)[0])

    probabilidades = modelo.predict_proba(alumno_escalado)[0]

    indice = list(modelo.classes_).index(prediccion)
    confianza = float(probabilidades[indice]) * 100

    return prediccion, confianza


def obtener_vecinos(horas, asistencia, tareas):
    modelo = joblib.load(MODELO)
    scaler = joblib.load(ESCALADOR)
    datos = joblib.load(DATOS)

    alumno = pd.DataFrame(
        [[horas, asistencia, tareas]],
        columns=COLUMNAS
    )

    alumno_escalado = scaler.transform(alumno)

    distancias, indices = modelo.kneighbors(alumno_escalado)

    vecinos = datos.iloc[indices[0]].copy()
    vecinos["distancia"] = distancias[0]

    vecinos["resultado"] = vecinos["resultado"].map({
        0: "REPRUEBA",
        1: "APRUEBA"
    })

    return vecinos.reset_index(drop=True)


if __name__ == "__main__":
    acc, matriz, train, test = entrenar_modelo_knn()

    print("Modelo KNN entrenado correctamente")
    print(f"Exactitud: {acc:.2%}")
    print("Matriz de confusion:")
    print(matriz)

