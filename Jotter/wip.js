

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

/**
 var addTwoNumbers = function(l1, l2) {
    var h1, h2, h3;
    var a1, a2;
    h1 = l1
    h2 = l2
    a1 = []
    a2 = []
    
    var head = new ListNode()
    l3 = head
    let carry = 0
    while(l1 != null){
        console.log(carry)
        let temp = 0;
        if(carry > 0){
            temp = 1
            console.log(temp)
        }
        temp += l1.val + l2.val
        
        if(temp > 9){
            carry = 1
            var sac = new ListNode()
            l3.val = temp - 10
            if(l1.next==null){
                break
            }
            l3.next = sac
            console.log(l3)
            l3 = l3.next
        }
        else{
            carry = 0
            var sac = new ListNode()
            l3.val = temp
            if(l1.next==null){
                break
            }
            l3.next = sac
            console.log(l3)
            l3 = l3.next
        }
        
        l1 = l1.next
        l2 = l2.next
        
    }
    
    return head
    
    /*
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
};


    */