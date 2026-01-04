import streamlit as st
import numpy as np
import joblib
import os
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

MODEL_FILE = "iris_model.pkl"

# Train or load model
if not os.path.exists(MODEL_FILE):
    model = LogisticRegression()
    model.fit(X, y)
    joblib.dump(model, MODEL_FILE)
else:
    model = joblib.load(MODEL_FILE)

# Streamlit UI
st.title("🌸 Iris Flower Prediction App")
st.write("Running on Google Colab")

sepal_length = st.number_input("Sepal Length", 0.0, 10.0, 5.0)
sepal_width = st.number_input("Sepal Width", 0.0, 10.0, 3.0)
petal_length = st.number_input("Petal Length", 0.0, 10.0, 4.0)
petal_width = st.number_input("Petal Width", 0.0, 10.0, 1.0)

if st.button("Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)
    flower_name = iris.target_names[prediction[0]]
    st.success(f"🌼 Predicted Flower: **{flower_name}**")
