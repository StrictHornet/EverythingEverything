//Pushed mid solution

var removeKdigits = function(num, k) {
    num = "736519";
    k = 3;
    let length = num.length;
    let high = [];
    let max = [];
    if(num.length == k) return 0;
    
    for(let i = 0; i < num.length; i++){
        high.push((num[i]));
    }
    
    high = high.sort((a, b) => a - b);
    //console.log(num.length - 3)
    
    let result = "";
    for(let i = length - 1; i > length - (k+1); i--){
        num = num.replace(high[i], '');
    }
    
    return num;
};