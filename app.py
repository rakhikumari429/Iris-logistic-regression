import streamlit as st
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

# -----------------------------
# Load dataset and train model
# -----------------------------
iris = load_iris()
X = iris.data
y = iris.target

model = LogisticRegression()
model.fit(X, y)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🌸 Iris Flower Predictor")
st.write("Enter the flower measurements below and click Predict:")

# Input features
sepal_length = st.number_input("Sepal Length (cm)", 0.0, 10.0, 5.0)
sepal_width  = st.number_input("Sepal Width (cm)", 0.0, 10.0, 3.0)
petal_length = st.number_input("Petal Length (cm)", 0.0, 10.0, 4.0)
petal_width  = st.number_input("Petal Width (cm)", 0.0, 10.0, 1.0)

# Predict button
if st.button("Predict"):
    input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_features)
    flower_name = iris.target_names[prediction[0]]
    
    st.success(f"🌼 Predicted Flower: **{flower_name}**")
