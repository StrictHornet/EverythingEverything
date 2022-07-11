# IMPORTING THE LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd #IMPORT PANDAS MODULE


# IMPORTING THE DATASET
dataset = pd.read_csv("Data.csv")  # Users/Ehi/source/repos/ML_A-Z/
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print(x)
print(y)

# Archive this file
