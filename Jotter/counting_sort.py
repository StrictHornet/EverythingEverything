arr = [1,1,3,4,1,2,3,1,0,1,3,4,2]
count = [0, 0, 0, 0, 0]
result = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
for i in arr:
    count[i] += 1

i = 1
while i < len(count):
    count[i] += count[i-1]
    i += 1
    
i = len(arr) - 1
while i >= 0:
    x = arr[i]
    #print(count)
    count[x] -= 1
    #print(count)
    result[count[x]] = x
    i -= 1

print(result)
