a = [2,3,1,1,2,2,4,5,5,4]

array = []
b = sorted(a)
print(b)
index = 0
i = 1
setCount = 0

for element in b:
    count = 1
    index += 1
    if element in array:
        continue
    else:
        array.append(element)
    i = index
    while(i < len(b)):
        if b[i] - element < 2:
            count += 1
        else:
            break
        i += 1
    print(count)    
    if count > setCount:
        setCount = count

print(setCount)
print(array)