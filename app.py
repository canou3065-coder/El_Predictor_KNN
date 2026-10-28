
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from modelo import (
    entrenar_modelo_knn,
    predecir_alumno,
    obtener_vecinos
)

st.set_page_config(
    page_title="El Predictor KNN",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 EL PREDICTOR")
st.subheader("Aprendizaje Supervisado con KNN")

st.write(
    "Sistema que predice si un alumno aprueba o reprueba "
    "basandose en ejemplos de alumnos anteriores."
)

# Configuracion del modelo
st.sidebar.header("Configuracion de KNN")

k = st.sidebar.slider(
    "Numero de vecinos K",
    min_value=1,
    max_value=15,
    value=5,
    step=2
)

try:
    exactitud, matriz, n_train, n_test = entrenar_modelo_knn(k)

except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
    st.error(f"No se pudo entrenar el modelo: {error}")
    st.info(
        "Revisa que exista datos/alumnos.csv "
        "y que contenga ejemplos etiquetados."
    )
    st.stop()

# Resultados del entrenamiento
st.sidebar.metric(
    "Exactitud del modelo",
    f"{exactitud * 100:.2f}%"
)

st.sidebar.write(f"Alumnos para entrenar: {n_train}")
st.sidebar.write(f"Alumnos para probar: {n_test}")

st.sidebar.subheader("Matriz de confusion")

fig, ax = plt.subplots(figsize=(4, 3))

imagen = ax.imshow(matriz, cmap="Blues")

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])

ax.set_xticklabels(["Reprueba", "Aprueba"])
ax.set_yticklabels(["Reprueba", "Aprueba"])

ax.set_xlabel("Prediccion")
ax.set_ylabel("Resultado real")

for i in range(2):
    for j in range(2):
        ax.text(
            j, i, str(matriz[i, j]),
            ha="center",
            va="center"
        )

fig.colorbar(imagen, ax=ax)
fig.tight_layout()

st.sidebar.pyplot(fig)
plt.close(fig)

# Datos del alumno
st.header("Ingresar datos del alumno")

col1, col2, col3 = st.columns(3)

with col1:
    horas = st.slider(
        "Horas de estudio diarias",
        0.0, 10.0, 5.0, 0.5
    )

with col2:
    asistencia = st.slider(
        "Asistencia (%)",
        0, 100, 80
    )

with col3:
    tareas = st.slider(
        "Tareas entregadas (%)",
        0, 100, 70
    )

# Prediccion
if st.button("Predecir con KNN", use_container_width=True):

    resultado, confianza = predecir_alumno(
        horas, asistencia, tareas
    )

    st.divider()

    if resultado == 1:
        st.success("RESULTADO: APRUEBA")
    else:
        st.error("RESULTADO: REPRUEBA")

    st.write(
        f"Porcentaje de votos de la clase ganadora: "
        f"{confianza:.1f}%"
    )

    # Mostrar vecinos utilizados por KNN
    st.subheader("Los alumnos mas parecidos")

    vecinos = obtener_vecinos(
        horas, asistencia, tareas
    )

    st.dataframe(
        vecinos,
        use_container_width=True,
        hide_index=True
    )
    # Descargar tabla de los vecinos más cercanos
    
    csv = vecinos.to_csv(index=False).encode("utf-8-sig")

    st.download_button(
    label="📥 Descargar tabla de alumnos",
    data=csv,
    file_name="tabla_alumnos_knn.csv",
    mime="text/csv",
    use_container_width=True
    )

    st.info(
        "KNN compara al alumno con los ejemplos "
        "mas cercanos del entrenamiento y decide "
        "segun la clase mayoritaria."
    )

st.divider()

st.caption(
    "Proyecto educativo de aprendizaje supervisado. "
    "Las predicciones dependen de la calidad de los datos."
)
