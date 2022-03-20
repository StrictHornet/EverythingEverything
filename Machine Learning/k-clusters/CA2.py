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

for ele in container: 
    i = 0
    ele.pop(0)
    for va in ele:
        ele[i] = float(va)
        i += 1

to_np = np.asarray(container)
means = np.mean(to_np, axis = 1)
print(means)
