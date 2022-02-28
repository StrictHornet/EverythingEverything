x = [-3, -2, -1, -0.2, -1, -2, 0]
min = -3
max = 0
y = []

for no in x:
    y.append(round(((no - min) / (max - min)), 2) * 100)
    
print(y)