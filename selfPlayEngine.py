import GameRepresentationFunctional 
from MCTS import MCTS



if __name__ == "__main__":
    mcts = MCTS(GameRepresentationFunctional.INITIAL_STATE, 10)
    for i in range(1):
        mcts.search()
        print("Best action:", mcts.get_best_action())
        print(GameRepresentationFunctional.stringRep(*mcts.root.state))
    # mcts.print_tree(mcts.root)
    print(len(mcts.root.children))