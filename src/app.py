# collect user input
# send input to model
# display prediction

import streamlit as st
import numpy as np
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

st.markdown("# :violet[Iris flower Classifier]")

data = load_iris()
X = data.data
y = data.target

knn = KNeighborsClassifier(n_neighbors=3) # create the model

sepal_length = st.number_input("Enter the sepal length")
sepal_width = st.number_input("Enter the sepal width")
petal_length = st.number_input("Enter the petal length")
petal_width = st.number_input("Enter the petal width")

predict_button = st.button("Predict")
knn.fit(X, y) # train model using only taining data

if predict_button: 
    iris = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = knn.predict(iris)
    species = data.target_names[prediction[0]]
    st.success(f"Prediction: {species}")