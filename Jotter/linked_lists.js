var head;

class Node{

    constructor(val){
        this.data = val
        this.next = null
    }
}

function print(){
    n = head
    while(n != null){ 
        console.log(n.data)
        n = n.next  
    }
}

function push(vall){
    var new_node = new Node(vall)

    new_node.next = head
    head = new_node
}

head = new Node(1)
sec = new Node(2)
head.next = sec

print()
push(0)
print()