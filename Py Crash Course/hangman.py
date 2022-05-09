import random
import string
from words import words

def word_choice():
    word = random.choice(words)
    while "-" in word or " " in word:
        word = random.choice(words)
    
    return word

def play_hangman():
    word = word_choice().upper()
    lives = 10
    used_letters = set()
    word_letters = set(word)
    print(word_letters)
    alphabet = set(string.ascii_uppercase)
    while lives != 0 and bool(word_letters) is True:
        word_list = [] #word_list = [letter if letter in used_letters else "-" for letter in word]
        for letter in word:
            if letter in used_letters:
                word_list.append(letter)
            else:
                word_list.append("-")
        print("The word to guess is", " ".join(word_list))
        print(f"You have {lives} lives left. You have guessed the words {used_letters}")
        u_input = input("Guess a letter to complete the word: ").upper()
        if u_input in alphabet - used_letters:
            used_letters.add(u_input)
            if u_input in word_letters:
                print("You guessed a word correctly :)\n")
                word_letters.remove(u_input)
            else:
                print("You guessed a word incorrectly :(\n")
                lives -= 1
        elif u_input in used_letters:
            print(f"You've used letter {u_input} \n")
        else:
            print("Invalid character. Try again.")

    if lives == 0:
        print("Your lives have finished. \nYou are Dead!")
        print(f"The word was {word}")
    else:
        print(f"##### You guessed the word {word} correctly! #####")

play_hangman()