import random
import matplotlib.pyplot as plt


PRINTBOARD = 0

# Hyperparameters
WIN_WEIGHT = 50
LOSS_WEIGHT = -50
TIE_WEIGHT = 5
LR = 0.05
GAMES_TRAIN = 10000

# Train the model's weights
def train(games = 1000):
    wins = 0
    ties = 0
    win_percent = []
    tie_percent = []

    w = randW()
    for i in range(games):
        board = [0,0,0,0,0,0,0,0,0]
        states = []
        
        # Play a game
        while (checkWin(board) == -2):

            # Player move
            successors = succ(board)

            if not successors:
                break

            best_b = successors[0]
            for s in successors: 
                if(v_hat(w, s) > v_hat(w, best_b)):
                    best_b = s
            board = best_b
            printBoard(board, PRINTBOARD)

            if (checkWin(board) != -2):
                break
            else:
                # Opponent move
                board = opponentMove(board)
                printBoard(board, PRINTBOARD)
            
            states.append(board.copy())
            

        # Update weights based on outcome
        for state in states:
            for j in range(9):
                error = calcError(w, state)
                w[j] -= LR * error * state[j] 

        # Tally score
        result = checkWin(board)
        if (result == 1):
            wins += 1
        elif (result == 0):
            ties += 1

        win_percent.append(wins/games*100)
        tie_percent.append(ties/games*100)
    
    
    # Plot win percentage and tie percentage over time
    plt.plot(range(1, games+1), win_percent, label='Win Percentage')
    plt.plot(range(1, games+1), tie_percent, label='Tie Percentage')
    plt.xlabel('Games')
    plt.ylabel('Percentage')
    plt.title('Win Percentage and Tie Percentage vs Games')
    plt.legend()
    plt.show()
    return w

# Random weights
def randW():
    w = []
    for i in range(9):
        w.append(random.random())
    return w

# Score of a specific board
def v_hat(w, b):
    total = 0
    for i in range(9):
        total += w[i] * b[i]
    return total

# calculate the error of the model
def calcError(w, b):
    result = checkWin(b)

    if (result==1):
        return WIN_WEIGHT - v_hat(w, b)
    elif (result==0):
        return TIE_WEIGHT - v_hat(w, b)
    elif (result==-1):
        return LOSS_WEIGHT - v_hat(w, b)
    
    # If the current state is still in progress, estimate the error   
    successors = succ(b)
    v_b = float('-inf')
    for s in successors: 
        v_b = max(v_hat(w, s), v_b)

    return v_b - v_hat(w, b)

# Checks current state of the board  
def checkWin(b):
    if (b[0] == b[1] and b[1] == b[2] and b[0] != 0): # top row
        return b[0]
    elif (b[3] == b[4] and b[4] == b[5] and b[3] != 0): # middle row
        return b[3]
    elif (b[6] == b[7] and b[7] == b[8] and b[6] != 0): # bottom row
        return b[6]
    elif (b[0] == b[3] and b[3] == b[6] and b[0] != 0): # left column
        return b[0]
    elif (b[1] == b[4] and b[4] == b[7] and b[1] != 0): # middle column
        return b[1]
    elif (b[2] == b[5] and b[5] == b[8] and b[2] != 0): # right column
        return b[2]
    elif (b[0] == b[4] and b[4] == b[8] and b[0] != 0): # diagonal left
        return b[0]
    elif (b[2] == b[4] and b[4] == b[6] and b[2] != 0): # diagonal right
        return b[2]
    else:
        for i in range(9): # check for tie
            if b[i] == 0:
                return -2
        return 0

# Creates all successors of a board
def succ(b):
    successors = []
    for i in range(9):
        if b[i] == 0:
            succ_board = b.copy()
            succ_board[i] = 1
            successors.append(succ_board)
    return successors

# Print the board
def printBoard(b, enable_print=1):
    if (enable_print == 0):
        return
    else:
        print(b[0],"|", b[1],"|", b[2])
        print("- - - - -")
        print(b[3],"|", b[4],"|", b[5])
        print("- - - - -")
        print(b[6],"|", b[7],"|", b[8])
        print("\n")


# Play a game with the trained weights      
def play(w):
    board = [0,0,0,0,0,0,0,0,0]
    while (checkWin(board) == -2):
        successors = succ(board)
        best_b = successors[0]
        for s in successors: 
            if(v_hat(w, s) > v_hat(w, best_b)):
                best_b = s

        board = best_b
        printBoard(board)
        if (checkWin(board) != -2):
            break
        board = opponentMove(board)
        printBoard(board)
        if (checkWin(board) != -2):
            break

# Opponent Tic Tac Toe Player
def opponentMove(b):
    # Check if it can win
    for i in range(9):
        if (b[i] == 0):
            b[i] = -1
            if (checkWin(b) == -1):
                return b
            b[i] = 0  # Undo move

    # Check if bot could win
    for i in range(9):
        if (b[i] == 0):
            b[i] = 1
            if (checkWin(b) == 1):
                b[i] = -1 # Block win
                return b
            b[i] = 0  # Undo move

    # Otherwise, pick random
    moves = []
    for i in range(9): 
        if (b[i] == 0):
            moves.append(i)
    if moves:
        move = random.choice(moves)
        b[move] = -1  # Opponent makes a random move
    
    return b

# -=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def main():
    w = train(GAMES_TRAIN)
    print("Trained weights: ", w)
    play(w)

main()