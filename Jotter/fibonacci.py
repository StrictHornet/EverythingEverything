# x = int(input("what is the first number?"))
# y = int(input("what is the second number?"))
# z = int(input("how many numbers in the sequence?"))
x =0
y =1
z =10
print(f"{x}, {y}", end="")

i=0
while(True):
    next = x+y
    print(f", {next}", end="") 
    if(i >= z-2):
        break
    x = y
    y = next
    i+=1