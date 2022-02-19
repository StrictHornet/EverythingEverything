function reverse(num){
    //Disregard input that isn't a number
    if(isNaN(num)){
        return "Won't be doing much for you without a number.";
    }

    //Eliminate leading zero's then convert number to string
    let number = "";
    number = parseInt(num).toString();

    let negation = false;
    let reverse_number = "";
    let trailing_zero = true;
    
    //Check if input is a negative value
    if(number[0] == "-"){
        negation = true;
    }

    //Add to reverse_number individual numbers in a reverse 
    //manner disregarding trailing zero's and negation symbol
    for(let i = number.length-1; i >= 0; i--){
        //Set trailing_zero to false when non-zero number is found
        if(number[i] != "0"){
                trailing_zero = false;
            }
        
        //End loop when at negation symbol
        if(negation && i == 0){
            break;
        }
        
        //Skip trailing zero's
        if(number[i] == "0" && trailing_zero == true){
            continue;
        }
        else{
            reverse_number += number[i]; //Adds numbers to reverse_number variable
        }
    }

    //If input was negative, negate result then return reverse
    if(negation){
        return parseInt(reverse_number) * (-1);
    }

    //If input wasn't negative return reverse
    return parseInt(reverse_number);
}

console.log(reverse(-40002000));