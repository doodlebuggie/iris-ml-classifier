"""
This module us used to build the classfier.

"""

import pandas as pd
import numpy as np
import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

data = load_iris()
X = data.data
y = data.target
#shape = data.data.shape

df = pd.DataFrame(X, columns=data.feature_names)
df['species'] = y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,shuffle=True, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3) # create the model
knn.fit(X_train, y_train) # train model using only taining data
y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(accuracy)



