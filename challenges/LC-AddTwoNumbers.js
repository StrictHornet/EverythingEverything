/**
 * Definition for singly-linked list.
 * function ListNode(val, next) {
 *     this.val = (val===undefined ? 0 : val)
 *     this.next = (next===undefined ? null : next)
 * }
 */
/**
 * @param {ListNode} l1
 * @param {ListNode} l2
 * @return {ListNode}
 */
 var addTwoNumbers = function(l1, l2) {
    var h1, h2, h3;
    var a1, a2;
    h1 = l1
    h2 = l2
    a1 = []
    a2 = []
    
    while(h1 != null){
        //console.log(head.val)
        a1.unshift(h1.val)
        h1 = h1.next
    }
    
    while(h2 != null){
        //console.log(head.val)
        a2.unshift(h2.val)
        h2 = h2.next
    }
    
    var v1, v2;
    
    v1 = a1.join("")
    v2 = a2.join("")
    v1 = parseInt(v1)
    v2 = parseInt(v2)
    //console.log(v1, v2)
    let v3 = v1 + v2
    //console.log(v3)
    v3 = v3.toString()
    
    var head = new ListNode()
    var l = head
    for(let i = v3.length-1; i >=0 ; i--){
        l.val = v3[i]
        
        if(v3[i-1] == undefined){
            l.next = null
            return head
        }
        
        var temp = new ListNode()
        l.next = temp
        l = l.next
    }
    
    /*
    while(head != null){
        console.log(head.val)
        head=head.next
    }

    ##################
            wip
    ##################
    

    */

    
};