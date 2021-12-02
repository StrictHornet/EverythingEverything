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