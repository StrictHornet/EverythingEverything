//REVERSE SRTING FUNCTION (10 DAYS OF JS)
function reverseString(s) {
    try{
        var split_s = s.split("");
        var concat_s = "";
        for(let x of split_s){
            concat_s = x.concat(concat_s);
    }
        s = concat_s;
    }
    catch(e){
        console.log(e.message);
    }
    console.log(s);
}

//10-03-2021
//Multiple Linear regressor on google colabs

/*
 * Complete the isPositive function.
 * If 'a' is positive, return "YES".
 * If 'a' is 0, throw an Error with the message "Zero Error"
 * If 'a' is negative, throw an Error with the message "Negative Error"
 */
function isPositive(a) {
    if(a>0){
        return "YES"
    }
    else if(a == 0){
        throw new Error("Zero Error");
    }
    else{
        throw new Error("Negative Error")
    }
}


/*
 * Complete the Rectangle function
 */
function Rectangle(a, b) {
    var Quad = {
        length : a,
        width : b,
        perimeter : 2 * (a+b), 
        area : a * b
    };
    
    return Quad
}