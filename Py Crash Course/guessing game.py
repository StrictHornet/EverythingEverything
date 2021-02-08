import random

def guess(x, count):
    random_num = random.randint(1, x)
    while(count > 0):
        user_guess = int(input("What is your guess: "))
        if user_guess == random_num:
            print("You guessed right!!!")
            break
        elif(user_guess < random_num):
            print("You guessed a lesser number!")
        else:
            print("You guessed a higher number!")
        count -= 1

x = int(input("You want the computer to guess from 1 to: "))
count = int(input("How many tries do you want? : "))
guess(x, count)
print("GAME OVER!")

'''
guess = 0
while guess != random_num:
    guess = int(input("Guess a number:"))
    return "Too high" if guess > random_num else "Too low"
'''