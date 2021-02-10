import random

def play_RPS():
    user = input("Pick between Rock 'r', Paper 'p', and Scissors 's': ")
    computer =  random.choice('rps')

    result = player_position(user, computer)

    if user == computer:
        return "It's a Tie"
    
    if result:
        return "You won"
    
    return "Computer won"

#r>s p>r s>p
def player_position(player, opponent):
    if (player == "r" and opponent == "s") or (player == "p" and opponent \
        == "r") or (player == "s" and opponent == "p"):
        return True

for letter in "rps":
    print(play_RPS())