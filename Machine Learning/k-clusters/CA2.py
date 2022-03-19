# Import libraries used in solution
import numpy as np
import pandas as pd
#import warnings

#Import Training and Test Data
data = pd.read_csv('animals.txt.txt')
animals_data = np.array(data.values)
#print(animals_data)
container = []
for i in animals_data:
    for z in i:
        temp = []
        temp = z.split(" ")
        container.append(temp)
        #print(container)

i = 1
for ele in container:
    ele.pop(0)
    for va in ele:
        ele[i] = float(va)
    i += 1

for i in container:
    print(i, "\n")

print(isinstance(container[0][1], float))