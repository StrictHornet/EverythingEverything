def runningSum():
    nums = []
    nums_length = int(input("How many numbers do you want in your list? "))
    for i in range(nums_length):
        uInput = int(input("Add a number: "))
        nums.append(uInput)
    res = []
    i = 1
    sum = nums[0]
    res.append(sum)
    while(i<len(nums)):
        sum = sum + nums[i]
        res.append(sum)
        i+=1
    return res

print(runningSum())