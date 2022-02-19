function oddCount(num){
    let number = BigInt(num); //Convert input to BigInt
    let odd = 0;               //Variable to store result
    let two = BigInt(2);       //To store BigInt version of the number 2
    
    //Simple algebra that produces the number of
    //positive odd numbers between 0 and the input.
    if(number % two == 0){    //Checks if input is even
        odd = number/two;
    }
    else{                     //Does this if input is odd
        odd = (number-1n)/two;
    }
    
    return Number(odd);       //Converts result back to the Number primitive
}

console.log(oddCount(4127356121254));