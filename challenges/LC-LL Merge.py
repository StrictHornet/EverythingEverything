# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        
        a = list1
        b = list2
        c = b
        arr = []
        if(a == None):
            return b
        
        while(a):
            if(b == None):
                return a
            
            if(a.val <= b.val):
                temp = b.next
                b.next = a
                temp2 = a.next
                a.next = temp
                a = temp2
                b = b.next
            elif(a.val > b.val):
                temp = b.next
                b.next = a
                b = b.next
                temp2 = a.next
                a.next = temp
                a = temp2
                b = b.next
            else:
                a = a.next
        
        return c