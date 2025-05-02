import GameRepresentationFunctional 
from MCTS import MCTS



if __name__ == "__main__":
    mcts = MCTS(GameRepresentationFunctional.INITIAL_STATE, 1000)
    for i in range(10):
        mcts.search()
        print("Best action:", mcts.get_best_action())
        print(GameRepresentationFunctional.stringRep(*mcts.root.state))
        mcts.update_root(mcts.get_best_action())