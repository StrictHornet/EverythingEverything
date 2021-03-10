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