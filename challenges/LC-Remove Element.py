nums = [1,2,2,4,2]
val = 2
i = 0
loop = len(nums)
while(i< len(nums)):
    if nums[i] == val:
        del nums[i]
        print(nums)
        continue
    i+=1

print(nums)