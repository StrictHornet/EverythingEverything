# IMPORTING THE LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# IMPORTING THE DATASET
dataset = pd.read_csv("Data.csv")  # Users/Ehi/source/repos/ML_A-Z/
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print(x)
print(y)
