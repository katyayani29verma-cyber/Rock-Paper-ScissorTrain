import random
# from live_predict.py import player_move

# import random

def decide_winner(player_move):

    choices = ["rock", "paper", "scissors"]
    computer_move = random.choice(choices)

    if player_move == computer_move:
        result = "Draw"

    elif player_move == "rock" and computer_move == "scissors":
        result = "You Win"

    elif player_move == "paper" and computer_move == "rock":
        result = "You Win"

    elif player_move == "scissors" and computer_move == "paper":
        result = "You Win"

    else:
        result = "You Lose"

    return computer_move, result