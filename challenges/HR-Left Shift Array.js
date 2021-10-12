function rotLeft(a, d) {
    // Write your code here
    var array1 = a;
    var rotate = d;
    var len = array1.length 
    while(d>0){
        var i=1;   
        var array2 = [];
        
        while(i < len){
            array2.push(array1[i]);
            i += 1;
        }
        
        array2.push(array1[0])
        array1 = array2
        d -= 1;
    }
    
    return array1;
}