function Majority(array){
    len = array.length;
    if (len == 0){
        return "No majority element";
    }

    if(len == 1){
        return array[0];
    }

    if(len % 2 != 0){
        var x = 0;
        x = array[len-1];
        var count = 0;
        for(var i = 0; i < len; i++){
            if(array[i] == x){
                count += 1;
            }
            if(count > len/2){
                return x;
            }
            else{
                array.pop();
                Majority(array);
            }
        }
    }

    var B = [];
    var j = 0;
    for (var i = 0; i < len; i+= 2) {
        if (array[i] == array[i+1]){
            B[j] = array[i];
            j += 1;
        }
    }
    
    var Maj = Majority(B);
    return Maj;
}

var array = [10, 10, 4, 10, 6, 6, 5, 10, 2, 5, 10, 10, 6, 6, 10, 10, 10, 10];
var z = Majority(array);
console.log("Majority element:" + z);