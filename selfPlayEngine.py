import GameRepresentationFunctional 
from MCTS import MCTS

def best_move(state):
    mcts = MCTS(state, 10000)
    mcts.search()
    best_action = mcts.get_best_action()
    if best_action:
        return best_action
    else:
        print("no action found")

if __name__ == "__main__":
    mcts = MCTS(GameRepresentationFunctional.INITIAL_STATE, 10000)
    for i in range(90):
        mcts.search()
        best_action = mcts.get_best_action()
        if best_action:
            print(GameRepresentationFunctional.stringRep(*mcts.root.state))
            mcts.update_root(mcts.get_best_action())
        else:
            print(GameRepresentationFunctional.stringRep(*mcts.root.state))
            break