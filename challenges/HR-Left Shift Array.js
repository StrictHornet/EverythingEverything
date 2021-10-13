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

/*
VERSION 2 USING RECURSION
*/
function rotLeft(a, d) {
    // Write your code here
    var rotations = d;
    var A1 = a;
    var A2 = [];
    var left = A1[0];
    var i = 1;
    A1.shift();
    A2 = A1;
    A2.push(left);
    rotations -= 1;
    if(rotations > 0){
        A1 = rotLeft(A2, rotations);
    }
    else{
        //console.log(A2);
        return A2;
    }
    console.log(A1);
    return A1;
}
