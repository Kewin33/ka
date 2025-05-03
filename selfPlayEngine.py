import GameRepresentationFunctional 
from MCTS import MCTS
from NNet import UltimateTTTNet

def best_move(state):
    mcts = MCTS(state, 10000)
    mcts.search()
    best_action = mcts.get_best_action()
    if best_action:
        return best_action
    else:
        print("no action found")

def play_interface():
    state = GameRepresentationFunctional.INITIAL_STATE
    while(1):
        mcts = MCTS(state, 10000)#, UltimateTTTNet())
        mcts.search()
        best_action = mcts.get_best_action()
        state = GameRepresentationFunctional.move(*state, *best_action)
        print(f"last_move {best_action}")
        print(GameRepresentationFunctional.stringRep(*state))

        zahl1 = -1
        zahl2 = -1
        print(GameRepresentationFunctional.getPossibleMoves(*state))
        while (zahl1,zahl2) not in GameRepresentationFunctional.getPossibleMoves(*state):
            zahl1 = int(input("Gib die erste Zahl ein: "))
            zahl2 = int(input("Gib die zweite Zahl ein: "))

        state = GameRepresentationFunctional.move(*state, zahl1, zahl2)
        print(GameRepresentationFunctional.stringRep(*state))

def play_person():
    state = GameRepresentationFunctional.INITIAL_STATE
    zahl1 = -1
    zahl2 = -1
    while(1):
        print(f"last_move {zahl1}, {zahl2}")
        zahl1 = -1
        zahl2 = -1
        print(GameRepresentationFunctional.stringRep(*state))


        print(GameRepresentationFunctional.getPossibleMoves(*state))
        while (zahl1,zahl2) not in GameRepresentationFunctional.getPossibleMoves(*state):
            zahl1 = int(input("Gib die erste Zahl ein: "))
            zahl2 = int(input("Gib die zweite Zahl ein: "))

        state = GameRepresentationFunctional.move(*state, zahl1, zahl2)
        print(GameRepresentationFunctional.stringRep(*state))



if __name__ == "__main__":

    print(f" Press 0 to enter two player \n Press 1 to enter get destroyed by AI \n")
    zahl1 = int(input("Your Choice: "))
    if zahl1 == 0:
        play_person()
    elif zahl1 == 1:
        play_interface()
    else:
        print("What the hell is that?")