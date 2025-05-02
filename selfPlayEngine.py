import GameRepresentationFunctional 
from MCTS import MCTS



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