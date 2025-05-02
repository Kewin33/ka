import copy
import random
import time


WIN_MASKS = [
    # horizontal
    0b000000111,
    0b000111000,
    0b111000000,
    # vertical
    0b001001001,
    0b010010010,
    0b100100100,
    #diagonal
    0b100010001,
    0b001010100,
]
    

INITIAL_STATE = (
    0b000000000, #global_state x
    0b000000000, # global state o
    [0b000000000] * 9, # local state x
    [0b000000000] * 9, # local state o
    True, # currentplayer
    9, # 0-8 are the local boards, 9 is free choice
    None, # winner
)

def move(global_state_x, global_state_o, local_state_x, local_state_o, currentPlayer, currentBoard, winner,  global_x, global_y):
        board = (global_y // 3) *3+ (global_x // 3 )
        local_x = global_x % 3
        local_y = global_y % 3
        board_name =  local_state_x if currentPlayer else local_state_o
        
 
        if (checkValidMove(global_state_x, global_state_o, local_state_x, local_state_o, currentBoard, board, local_x, local_y)):
            board_name[board] |= (1 << (local_y * 3 + local_x))

            if checkWin(board_name[board]):
                # if the player is X
                if currentPlayer:
                    global_state_x |= (1 << board)
                    if (checkWin(global_state_x & ~global_state_o)):
                        winner =  "X"
                        return (
                            global_state_x, global_state_o, local_state_x, local_state_o, not currentPlayer, currentBoard, winner
                        )
                # if the player is O
                else:
                    global_state_o |= (1 << board)
                    if (checkWin(global_state_o & ~global_state_x)):
                        winner = "O"
                        return (
                            global_state_x, global_state_o, local_state_x, local_state_o, not currentPlayer, currentBoard, winner
                        )
                    
                if checkDraw(global_state_x, global_state_o):
                    winner = "D"
                    return (
                        global_state_x, global_state_o, local_state_x, local_state_o, not currentPlayer, currentBoard, winner
                    )   
                
            elif checkDraw(local_state_x[board], local_state_o[board]):
                global_state_x |= (1 << board)
                global_state_o |= (1 << board)

                if checkDraw(global_state_x, global_state_o):
                    winner = "D"
                    return (
                        global_state_x, global_state_o, local_state_x, local_state_o, not currentPlayer, currentBoard, "D"
                    )
                    

            # check next board
            currentBoard = local_x + local_y * 3
            if isNotPlayableBoard(global_state_x, global_state_o, currentBoard):
                # if the next board is not playable, set it to 9
                currentBoard = 9


            # change player
            #currentPlayer = not currentPlayer
            return (global_state_x, global_state_o, local_state_x, local_state_o, not currentPlayer, currentBoard, winner)

        else:
            return None
        
### local functions:
def checkWin(board):
    for mask in WIN_MASKS:
        if (board & mask) == mask:
            return True
    return False

def checkDraw(board_x, board_o):
    # check if all local boards are full
    return (board_x | board_o == 0b111111111)

def isSetOnBoard(local_state_x, local_state_o, board, local_x, local_y):
    # check if the board is set
    idx = local_y * 3 + local_x
    if local_state_x[board] & (1 << idx) or local_state_o[board] & (1 << idx):
        return True
    return False

def isNotPlayableBoard(global_state_x, global_state_o, board):
    # check if the board is playable
    if global_state_x & (1 << board) or global_state_o & (1 << board):
        return True
    return False
        
        
### general functions
def checkValidMove(global_state_x, global_state_o, local_state_x, local_state_o, current_board, board, local_x, local_y):
    # check taht the board is valid 
    if not ((0 <= board <=9) and (current_board == 9 or board == current_board)):
        return False
    # check that global board is not set
    if isNotPlayableBoard(global_state_x, global_state_o, board):
        return False
    # check that the move is in bounds
    # TODO: remove later, as ai can only return in bound moves
    if local_x < 0 or local_x > 2 or local_y < 0 or local_y > 2:
        return False
    # check that the move is not already taken
    if isSetOnBoard(local_state_x, local_state_o, board, local_x, local_y):
        return False

    return True
        
    

def getPossibleMoves(global_state_x, global_state_o, local_state_x, local_state_o, currentPlayer, currentBoard, winner):
    if winner is not None:
        # if the game is over, return empty list
        return []
    # returns a list of possible moves
    # each move is a tuple (global_x, global_y)
    possible_moves = []
    if currentBoard == 9:
        #iterate through all local boards
        for board in range(9):
            # check if the board is still in play
            if isNotPlayableBoard(global_state_x, global_state_o, board):
                continue
            for local_x in range(3):
                for local_y in range(3):
                    if not isSetOnBoard(local_state_x, local_state_o, board, local_x, local_y):
                        possible_moves.append((local_x + board % 3 * 3, local_y + board // 3 * 3))
    else: 
        for local_x in range(3):
                for local_y in range(3):
                    if not isSetOnBoard(local_state_x, local_state_o, currentBoard, local_x, local_y):
                        possible_moves.append((local_x + currentBoard % 3 * 3, local_y + currentBoard // 3 * 3))
    return possible_moves



def stringRep(global_state_x, global_state_o, local_state_x, local_state_o, currentPlayer, currentBoard, winner):
    # returns a string representation of the board
    # 0 is empty, 1 is x, 2 is o
    board = ""
    for i in range(3):
        for j in range(3):
            if (global_state_x & global_state_o) & (1 << (i * 3 + j)):
                board += "D"
            elif global_state_x & (1 << (i * 3 + j)):
                board += "X"
            elif global_state_o & (1 << (i * 3 + j)):
                board += "O"
            else:
                board += "."
            # print vertical line
            if j != 2:
                board += "|"

        board += "\n"
    board += "\n"
    # print each row over all 9 boards
    for i in range(9):
        # print vertical line
        for j in range(9):
            board_idx = i // 3 * 3 + j // 3
            local_x = j % 3
            local_y = i % 3
            if local_state_x[board_idx] & (1 << (local_y * 3 + local_x)):
                board += "X"
            elif local_state_o[board_idx] & (1 << (local_y * 3 + local_x)):
                board += "O"
            else:
                board += "."
            board += " "
            if (j % 3 == 2) and (j != 8):
                board += "|"
        # print horizontal line
        board += "\n"
        if i % 3 == 2 and i != 8:
            board += "- - - - - - - - - \n"

    board += "\n"
    return board
        







if __name__ == "__main__":
    # set start time 
    start = time.time()
    for _ in range(10**5):
        game = copy.deepcopy(INITIAL_STATE)
        for i in range(100):
            possible_moves = getPossibleMoves(*game)
            if len(possible_moves) == 0:
                # print(f"Game over after {i} moves")
                break
            my_move = random.choice(possible_moves)
            game = move(*game, my_move[0], my_move[1])
    
    end = time.time()

    print(f"Time taken: {end - start} seconds")
