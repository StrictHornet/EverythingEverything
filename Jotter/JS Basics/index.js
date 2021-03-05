words = ["aback","abaft","abandoned","abashed","aberrant","abhorrent","abiding","abject","ablaze","able","abnormal","aboard","aboriginal","abortive","abounding","abrasive","abrupt","absent","absorbed","absorbing","abstracted","absurd","abundant","abusive","acceptable","accessible","accidental","accurate","acid","acidic","acoustic","acrid","actually","ad hoc","adamant","adaptable"]
var letter, rand_int;
var guessed_letters = [];
var lives = 5;

//Get random word
rand_int = Math.floor(Math.random() * 30)
document.getElementById("lives").innerHTML = lives
rand_word = words[rand_int]
hideletters();

//Collect guessed letters and show progress
function hideletters(){
    for (letter in rand_word){
        if (guessed_letters.includes(rand_word[letter]))
        {
            document.getElementById("rand_word").innerHTML += rand_word[letter]
        }
        else
        { 
            document.getElementById("rand_word").innerHTML += "*"
        }
    }
}

//Collect User Input and update HTML
function PlayGame(){
        var guessedletter = window.prompt("Input a guess letter")
        if (guessed_letters.includes(guessedletter))
            {
                window.alert("You've already guessed that letter!")
            }
        else
            { 
                guessed_letters.push(guessedletter)
            }
        lives--
        document.getElementById("lives").innerHTML = lives
        document.getElementById("rand_word").innerHTML = ""
        hideletters();
        if (lives < 0) {
            document.getElementById("status").innerHTML = "You failed!"
            document.getElementById("button").style.pointerEvents = "none"
            return 1;
        }
        else if (document.getElementById("rand_word").innerHTML == rand_word){
            document.getElementById("status").innerHTML = "YOU GOT THE WORD RIGHT"
        }
        else{
            document.getElementById("status").innerHTML = "Keep going..."
        }
}