class LL:
    def __init__(self):
        self.head = None

class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

listhead = LL()

node1 = Node("a")
node2 = Node("b")
node3 = Node("hello world!")

listhead.head = node1
node1.next = node2
node2.next = node3

print((node1.next).data)

while list1 and list2:
    if list1.val <= list2.val:
        tail.next = list1
        list1 = list1.next
    else:
        tail.next = list2
        list2 = list2.next
    tail = tail.next
if list1:
    tail.next = list1
elif list2:
    tail.next = list2
return dummy.next