function tripleTrouble(a, b, c){
    var length = a.length;      //Stores the length of an input string.
    var treble = "";            //Variable to store alphabets through traversal

    //Loop through each string concatenating each strings 
    //alphabet letter at index i and storing it in treble.
    for(var i = 0; i < length; i++){
        treble += a[i] + b[i] + c[i];
    }

    return treble;
}       

console.log(tripleTrouble("nace", "oths" , "vett"));