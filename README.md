# iris-ml-classifier

Overview
A machine learning classification project that predicts the species of an iris flower using measurements of its sepal and petal features. A user can input features using a basic UI. 

This project uses a K-Nearest Neighbours(KNN) classifier to classify flowers into one of three species: setosa, versicolor or virginica. 

Dataset
This project uses the Iris dataset from Scikit-learn

The dataset contains features: sepal length, sepal width, petal length and petal width and target: flower species. 

Machine Learning Approach
Model used: KNN.

KNN stores the training examples and uses distance between flowers when predicting the species. The model was trained using 80% training data and 20% testing data.

Model Evaluation
The model was evaluated using accuracy score and confusion matrix.
Hyperparameter Testing
Different values of n_neighbors were tested:

|  K Value       |       Accuracy    |
|----------------|-------------------|
|   1            |       100%        |
|   3            |       100%        |
|   5            |       100%        |
|   7            |       96.7%       |
|   10           |       100%        |

The selected model uses n_neighbors = 3, because it kept the value small and achieved high accuracy.

Technologies Used
Python
Pandas
NumPy
Scikit-learn
Streamlit
VS Code

Project Structure

iris-ml-classifier/
│
├── src/
│   ├── iris_analysis.py
│   └── model.py
│
├── README.md
└── requirements.txt

Future Improvements
Allow users to enter flower measurements manually.
Deploy the model as a web application.
Experiment with additional classification algorithms.