import random
import string
from words import words

def word_choice():
    word = random.choice(words)
    while "-" in word or " " in word:
        word = random.choice(words)
    
    return word

def play_hangman():
    word = word_choice()
    lives = 10
    used_letters = set()
    word_letters = set(word)
    print(word_letters)
    alphabet = set(string.ascii_uppercase)
    while lives != 0:
        u_input = input("Guess a letter to complete the word: ").upper()
        if u_input in alphabet - used_letters:
            used_letters.add(u_input)
            if u_input in word_letters:
                print("You guessed a word correctly :)")
            else:
                print("You guessed a word incorrectly :(")
                lives -= 1
        else:
            print(f"You've used letter {u_input}")

play_hangman()