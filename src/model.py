"""
This module us used to build the classfier.

"""

import pandas as pd
import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

data = load_iris()
X = data.data
y = data.target
#shape = data.data.shape

df = pd.DataFrame(data.data, columns=data.feature_names)
df['species'] = y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,shuffle=True, random_state=42)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
