#Runtime: 63 ms, faster than 84.13% of Python3 online submissions for Shuffle the Array.

class Solution:
    def shuffle(self, nums: List[int], n: int) -> List[int]:
        i = 0
        j = n
        shuff = []
        while(n>0):
            shuff.append(nums[i])
            shuff.append(nums[i+j])
            i += 1
            n -= 1
            
        return shuff