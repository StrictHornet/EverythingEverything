var lengthOfLastWord = function(s) {
    //let su = "   fly me   to   the moon  "
    let input = s
    let input_length = input.length
    
    let j = 0
    let word = []
    for(let i = input_length-1; i >= 0; i--){
        
        //console.log(su[i] + "x")
        if(word.length != 0 && input[i] == " "){
            break
        }
        
        if(input[i] == " "){
            continue
        }
        else{
            word.unshift(input[i])
        }
    }
    
    word = word.join("")
    console.log(word)
    return word.length
};