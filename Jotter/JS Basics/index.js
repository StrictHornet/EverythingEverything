words = ["aback","abaft","abandoned","abashed","aberrant","abhorrent","abiding","abject","ablaze","able","abnormal","aboard","aboriginal","abortive","abounding","abrasive","abrupt","absent","absorbed","absorbing","abstracted","absurd","abundant","abusive","acceptable","accessible","accidental","accurate","acid","acidic","acoustic","acrid","actually","ad hoc","adamant","adaptable"]

rand_int = Math.floor(Math.random() * 30)
document.getElementById("rand_int").innerHTML = rand_int
rand_word = words[rand_int]
var letter;
var guessed_letters = ["a", "b"]
for (letter in rand_word){
    console.log(rand_word[letter])
    if (guessed_letters.includes(rand_word[letter]))
    {
        document.getElementById("rand_word").innerHTML += rand_word[letter]
    }
    else
    { 
        document.getElementById("rand_word").innerHTML += "*"
    }
}
