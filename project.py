# Importar librerías

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Configuración página

st.set_page_config(
    page_title="Iris Species Classification",
    layout="wide"
)

st.title("🌸 Iris Species Classification Dashboard")

st.write(
    "Sistema de clasificación de flores Iris usando Machine Learning"
)

# Cargar dataset

iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

df["species"] = iris.target

species_names = {
    0:"Setosa",
    1:"Versicolor",
    2:"Virginica"
}

df["species_name"] = df["species"].map(species_names)

st.subheader("Dataset")

st.dataframe(df.head())
# Separar variables y etiquetas

X = df.iloc[:,0:4]
y = df["species"]

# Dividir datos entrenamiento/prueba

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Crear modelo

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Entrenar

model.fit(X_train,y_train)

# Predicciones

y_pred = model.predict(X_test)

# Métricas

accuracy = accuracy_score(y_test,y_pred)

precision = precision_score(
    y_test,
    y_pred,
    average='weighted'
)

recall = recall_score(
    y_test,
    y_pred,
    average='weighted'
)

f1 = f1_score(
    y_test,
    y_pred,
    average='weighted'
)

# Mostrar métricas

st.subheader("Model Metrics")

col1,col2,col3,col4 = st.columns(4)

col1.metric(
    "Accuracy",
    f"{accuracy:.2f}"
)

col2.metric(
    "Precision",
    f"{precision:.2f}"
)

col3.metric(
    "Recall",
    f"{recall:.2f}"
)

col4.metric(
    "F1 Score",
    f"{f1:.2f}"
)
# Panel de predicción

st.subheader("Predict Iris Species")

col1, col2 = st.columns(2)

with col1:

    sepal_length = st.number_input(
        "Sepal Length",
        min_value=4.0,
        max_value=8.0,
        value=5.1
    )

    sepal_width = st.number_input(
        "Sepal Width",
        min_value=2.0,
        max_value=5.0,
        value=3.5
    )

with col2:

    petal_length = st.number_input(
        "Petal Length",
        min_value=1.0,
        max_value=7.0,
        value=1.4
    )

    petal_width = st.number_input(
        "Petal Width",
        min_value=0.1,
        max_value=3.0,
        value=0.2
    )


if st.button("Predict Species"):

    new_data = np.array([[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]])

    prediction = model.predict(new_data)

    predicted_species = species_names[prediction[0]]

    st.success(
        f"Predicted Species: {predicted_species}"
    )
    # Gráfico 3D

st.subheader("3D Visualization")

fig = px.scatter_3d(
    df,
    x='sepal length (cm)',
    y='petal length (cm)',
    z='petal width (cm)',
    color='species_name',
    title='Iris Dataset Distribution'
)

# Agregar punto de la nueva predicción

if 'new_data' in locals():

    fig.add_scatter3d(
        x=[sepal_length],
        y=[petal_length],
        z=[petal_width],
        mode='markers',
        marker=dict(
            size=10,
            symbol='diamond',
            color='red'
        ),
        name='New Sample'
    )

st.plotly_chart(
    fig,
    use_container_width=True
)
# Visualizaciones adicionales

st.subheader("Additional Data Visualizations")

# Histograma

fig_hist = px.histogram(
    df,
    x='petal length (cm)',
    color='species_name',
    barmode='overlay',
    title='Petal Length Distribution'
)

st.plotly_chart(
    fig_hist,
    use_container_width=True
)

# Scatter Matrix

fig_matrix = px.scatter_matrix(
    df,
    dimensions=[
        'sepal length (cm)',
        'sepal width (cm)',
        'petal length (cm)',
        'petal width (cm)'
    ],
    color='species_name',
    title='Scatter Matrix'
)

st.plotly_chart(
    fig_matrix,
    use_container_width=True
)