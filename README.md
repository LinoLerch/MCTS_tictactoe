﻿# MCTS tictactoe
Monte Carlo Tree Search (MCTS) Python implementation with tictactoe and Gridworld Gymnasium environments. The implementation of the MCTS algorithm is based on the book Reinforcement Learning: An Introduction by Sutton and Barto (2018) Section 8.11.
## File structure
-`gym_examples/envs/`:  
---`grid_world.py`: Gridworld Env from [GitHub Repo gym-examples](https://github.com/Farama-Foundation/gym-examples)  
---`tictactoe_env.py`: TicTacToe Env from [GitHub Repo gym-tic-tac-toe](https://github.com/LudwigStumpp/gym-tic-tac-toe)  
-`demos/`: Demos for the two envs Gridworld and TicTacToe  
-`analysis/`: Analysis of the hyperparams `iteration_budget` and `exploration_c`  
-`mcts.py`: Monte Carlo Tree Search Algorithm (own implementation)

## Installation
```shell
pip install -e .
```

## References
Sutton, R. S. and Barto, A. G. (2018). Reinforcement Learning: An Introduction. A Bradford Book,
Cambridge, MA, USA.
