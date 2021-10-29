arr = [1,1,3,4,1,2,3,1,0,1,3,4,2]
count = [0, 0, 0, 0, 0]
#result = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#For when all keys match their integers
result = []

for i in arr:
    count[i] += 1

#For when all keys match their integers
j = 0
i = 0
while j < len(arr):
    if count[i] > 0:
        result.append(i)
        count[i] -= 1
    else:
        i += 1
        j -= 1
    j += 1

#Original Algorithm
#i = 1
#while i < len(count):
#    count[i] += count[i-1]
#    i += 1
    
#i = len(arr) - 1
#while i >= 0:
#    x = arr[i]
#    #print(count)
#    count[x] -= 1
#    #print(count)
#    result[count[x]] = x
#    i -= 1

print(result)
