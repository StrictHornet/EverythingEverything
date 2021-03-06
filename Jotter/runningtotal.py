x=y=sum=0

x = int(input("what is the beginning of the range: "))
y = int(input("what is the end of the range: "))

len = (y - x) + 1

for i in range(len):
    sum = sum + x
    x += 1

print(f"Your running sum is: {sum}")
