#--------------------------------------------------------------# Basic class definition
# class Institution:
#  def __init__(self, name, field):
#   self.name = name
#   self.field = field
  
#  def about(self):
#   print(f'{self.name} is a {self.field} intstitution.')
  
# class Company(Institution):
#  pass
 
 
# nameOfComp = input("What is your company's name:")
# fieldOfComp = input("What is your company's field:")

# company = Company(nameOfComp, fieldOfComp)
# company.about()                                      
# 
# #--------------------------------------------------------------# Linked lists
class ListNode:
  def __init__(self, data):
    self.data = data
    self.next = None
    return

  def has_value(self, value):
    return True if self.data == value else False

node1 = ListNode(15)
node2 = ListNode("Berlin")

class LinkedList:
  def __init__(self):
    self.head = None
    self.tail = None
    return

  def add_list_item(self, item):
    if not isinstance(item, ListNode):
      item = ListNode(item)

    if self.head == None:
      self.head = item
    else:
      self.tail.next = item
    
    self.tail = item
    return

  def list_length(self):
    count = 0
    current_node = self.head

    while current_node is not None:
      count+=1
      current_node = current_node.next

    return count

  def output_list(self):
    current_node = self.head

    while current_node is not None:
      print(current_node.data)
      current_node = current_node.next

    return

  def unordered_search(self, value):
    currentNode = self.head
    nodeID = 1
    results = []

    while currentNode is not None:
      if currentNode.has_value(value): results.append(nodeID) 
      currentNode = currentNode.next
      nodeID+=1

    return results

  def remove_item(self, itemID):
    currentID = 1
    currentNode = self.head
    previousNode = None

    while currentNode is not None:
      if currentID == itemID:
        if previousNode is not None:
          previousNode.next = currentNode.next
        else:
          self.head = currentNode.next
          return

      previousNode = currentNode
      currentNode = currentNode.next
      currentID+=1
    
    return

  def test(self, item):
    self.tail = item

    return


node3 = ListNode(90)
node4 = ListNode(12)
dummy = ListNode(2)
track = LinkedList()

mylist = LinkedList()
mylist.test(dummy)
print(mylist.output_list())

# [track.add_list_item(val) for val in [node1, node2, node3, node4]]
# print(f'track length is {track.list_length()}')

# track.output_list()
# print(track.unordered_search(90))

# track.remove_item(3)

# track.output_list()
# print(track.unordered_search(90))

  
    