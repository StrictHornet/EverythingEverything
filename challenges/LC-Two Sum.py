nums = [11, 2, 15, 7]
target = 9
i = 0
length = len(nums)

for f_index in range(length-1):
    j = i + 1
    while(j<length):
        if target - nums[i] == nums[j]:
            print(i,j)
            break
        j+=1
    i+=1


