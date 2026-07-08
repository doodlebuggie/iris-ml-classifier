"""
Iris flower classfication using machine learning algorithms. 
This module provides functions to load the Iris dataset,
explore dataset, visualize the patterns and understand its features.

"""

import pandas as pd
import sklearn
from sklearn.datasets import load_iris
import numpy as np

data = load_iris()
features = data.feature_names
labels = data.target_names
shape = data.data.shape
print(features)
print(labels)
print(shape)

mapped_species = labels[data.target]
df = pd.DataFrame(data.data, columns=features)
df['species'] = mapped_species
print(df.head())
