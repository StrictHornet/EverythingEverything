def UserInput():
    try:
        visa = 0
        while True:
            visa = int(input("What is your card number: "))
            if(type(visa) == int and visa > 0):
                print(visa)
                return visa
    except:
        print("Input a number!")
        return UserInput()
        
card_no = 0
card_no = UserInput()
print(card_no)

arr = [str(x) for x in str(card_no)]
i = len(arr) - 1
index = i - 1
check_sum = []
skip = 2

for var in range(i):
    if skip % 2 != 0:
        skip += 1
        index -= 1
        continue
    check_sum.append(int(arr[index]) * 2)
    arr.remove(arr[index])
    skip += 1
    index -= 1

sums = [int(y) for x in check_sum for y in str(x) ]
arr = [int(x) for x in arr]
totalsum = sums + arr
totalsums = 0
for x in totalsum:
    totalsums += x

print("Valid!") if totalsums % 10 == 0 else print("Invalid!")