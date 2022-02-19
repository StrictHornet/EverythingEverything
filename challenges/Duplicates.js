function duplicates(input){
    
    //Store input in word variable, 
    //initialise arrays to be used
    var word = input.toLowerCase();
    var search_array = [];
    var duplicate = [];

    //For each letter or number discovered
    //store in search_array. When a duplicate
    //is found store in duplicate array once.
    //Return duplicate array length as result
    for(var i = 0; i < word.length; i++){
        if(search_array.includes(word[i])){
            if(duplicate.includes(word[i])){
                continue;
            }
            else{
                duplicate.push(word[i]);
            }
        }
        else{
            search_array.push(word[i]);
        }
    }
    return duplicate.length;
}

console.log(duplicates("indivisibility121"));