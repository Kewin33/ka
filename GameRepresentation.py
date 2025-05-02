import random


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
    


class UltimateTicTacToeBitBoard:
    def __init__(self):
        self.global_state_x = 0b000000000
        self.global_state_o = 0b000000000
        self.local_state_x = [0b000000000] * 9
        self.local_state_o = [0b000000000] * 9
        self.currentPlayer = True
        self.currentBoard = 9 # 0-8 are the local boards, 9 is free choice
        self.winner = None

    def move(self, global_x, global_y):
        board = (global_y // 3) *3+ (global_x // 3 )
        local_x = global_x % 3
        local_y = global_y % 3
        print(f"local_x: {local_x}, local_y: {local_y}, board: {board}")
        board_name =  self.local_state_x if self.currentPlayer else self.local_state_o
        
 
        if (self.checkValidMove(board, local_x, local_y)):
            board_name[board] |= (1 << (local_y * 3 + local_x))

            if self.checkWin(board_name[board]):
                print(f"Player {self.currentPlayer} wins on board {board}")
                if self.currentPlayer:
                    self.global_state_x |= (1 << board)
                    if (self.checkWin(self.global_state_x & ~self.global_state_o)):
                        self.winner =  "X"
                        return
                else:
                    self.global_state_o |= (1 << board)
                    if (self.checkWin(self.global_state_o & ~self.global_state_x)):
                        self.winner = "O"
                        return
                

                if self.checkDraw(self.global_state_x, self.global_state_o):
                    return

            elif self.checkDraw(self.local_state_x[board], self.local_state_o[board]):
                self.global_state_x |= (1 << board)
                self.global_state_o |= (1 << board)

                if self.checkDraw(self.global_state_x, self.global_state_o):
                    print("global Draw")
                    self.winner = "Draw"
                    return
                    

            # check next board
            self.currentBoard = local_x + local_y * 3
            if self.isNotPlayableBoard(self.currentBoard):
                # if the next board is not playable, set it to 9
                self.currentBoard = 9


            # change player
            self.currentPlayer = not self.currentPlayer

        
    ### local functions:
    def checkWin(self, board):
        for mask in WIN_MASKS:
            if (board & mask) == mask:
                return True
        return False

    def checkDraw(self, board_x, board_o):
        # check if all local boards are full
        return (board_x | board_o == 0b111111111)

    def isSetOnBoard(self, board, local_x, local_y):
        # check if the board is set
        idx = local_y * 3 + local_x
        if self.local_state_x[board] & (1 << idx) or self.local_state_o[board] & (1 << idx):
            return True
        return False
    
    def isNotPlayableBoard(self, board):
        # check if the board is playable
        if self.global_state_x & (1 << board) or self.global_state_o & (1 << board):
            return True
        return False
        
        
### general functions
    def checkValidMove(self, board, local_x, local_y):
        # check taht the board is valid 
        if not ((0 <= board <=9) and (self.currentBoard == 9 or board == self.currentBoard)):
            return False
        # check that global board is not set
        if self.isNotPlayableBoard(board):
            return False
        # check that the move is in bounds
        # TODO: remove later, as ai can only return in bound moves
        if local_x < 0 or local_x > 2 or local_y < 0 or local_y > 2:
            return False
        # check that the move is not already taken
        if self.isSetOnBoard(board, local_x, local_y):
            return False

        return True
            
        

    def getPossibleMoves(self):
        if self.winner is not None:
            # if the game is over, return empty list
            return []
        # returns a list of possible moves
        # each move is a tuple (global_x, global_y)
        possible_moves = []
        if self.currentBoard == 9:
            #iterate through all local boards
            for board in range(9):
                # check if the board is still in play
                if self.isNotPlayableBoard(board):
                    continue
                for local_x in range(3):
                    for local_y in range(3):
                        if not self.isSetOnBoard(board, local_x, local_y):
                            possible_moves.append((local_x + board % 3 * 3, local_y + board // 3 * 3))
        else: 
            for local_x in range(3):
                    for local_y in range(3):
                        if not self.isSetOnBoard(self.currentBoard, local_x, local_y):
                            possible_moves.append((local_x + self.currentBoard % 3 * 3, local_y + self.currentBoard // 3 * 3))
        return possible_moves
    



    def set_global_state(self, global_x, global_y):
        # set the global state
        self.global_state_x = global_x
        self.global_state_o = global_y



    def __str__(self):
        # returns a string representation of the board
        # 0 is empty, 1 is x, 2 is o
        board = ""
        for i in range(3):
            for j in range(3):
                if (self.global_state_x & self.global_state_o) & (1 << (i * 3 + j)):
                    board += "D"
                elif self.global_state_x & (1 << (i * 3 + j)):
                    board += "X"
                elif self.global_state_o & (1 << (i * 3 + j)):
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
                if self.local_state_x[board_idx] & (1 << (local_y * 3 + local_x)):
                    board += "X"
                elif self.local_state_o[board_idx] & (1 << (local_y * 3 + local_x)):
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
    game = UltimateTicTacToeBitBoard()
    print(len(game.getPossibleMoves()))
    game.move(0, 0)
    for i in range(100):

        move = game.getPossibleMoves()
        if len(move) == 0:
            print(f"No more moves")
            break
        #randomly select a move
        move = random.choice(move)
        game.move(move[0], move[1])
        print(game)
    
    
