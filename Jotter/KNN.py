arr = [1, 4, 5, 7, 2, 3, 10]
count = [0, 0, 0, 0, 0]
#result = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#For when all keys match their integers
result = []

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

arrlen = len(result)
if (len(arr) % 2 == 0):
    med = (result[arrlen/2] + result[(arrlen/2)-1]) / 2
else:
    med = result[arrlen-1/2]

print(result)
