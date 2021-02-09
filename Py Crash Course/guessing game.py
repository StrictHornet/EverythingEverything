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

#TO RUN GUESS()
# x = int(input("You want the computer to guess from 1 to: "))
# count = int(input("How many tries do you want? : "))
# guess(x, count)
# print("GAME OVER!")

'''
guess = 0
while guess != random_num:
    guess = int(input("Guess a number:"))
    return "Too high" if guess > random_num else "Too low"
'''

def computer_guess():
    low = 1
    high = int(input(f"You want to pick a number to be guessed \
    between {low} and?: "))
    guess = 0
    feedback = ""
    while feedback != "c":
        guess = random.randint(low, high)
        feedback, low, high = Feedback(feedback, guess, low, high)
        
    print(f"We guessed your number {guess} correctly!")

            
def Feedback(feedback, guess, low, high):
    feedback = input(f"Is the computer's guess {guess} too high (H), too low (L) or correct (C): ").lower()
    if feedback == "h":
        high = guess - 1
    elif feedback == "l":
        low = guess + 1
    elif feedback != "c" and feedback != "l" and feedback != "h":
        feedback, low, high = Feedback(feedback, guess, low, high)

    return feedback, low, high

computer_guess()