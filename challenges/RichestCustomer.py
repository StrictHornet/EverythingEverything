class Solution:
    def maximumWealth(self, accounts: List[List[int]]) -> int:
        max = 0
        for customer in accounts:
            total = 0
            for account in customer:
                total += account
            
            if total > max:
                max = total
                
        return max