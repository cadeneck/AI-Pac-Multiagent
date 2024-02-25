
# Multi-Agent Pacman Project

## Introduction
In this stage of the project, you will design agents for the classic version of Pacman, including ghosts. This project focuses on implementing both minimax and expectimax search algorithms, as well as designing evaluation functions for more sophisticated agent behavior.

### Key Features
- Implementation of minimax and expectimax search algorithms.
- Design of evaluation functions for agents.
- Extension to multi-agent scenarios, including ghosts.

### Dependencies
- Python (version 3.x recommended)
- No additional dependencies are required for this stage.

## Usage
The autograder is included for self-assessment and can be run as follows for all questions or a specific question:

```bash
python autograder.py
python autograder.py -q q2
```

To run for a specific test with optional graphics:

```bash
python autograder.py -t test_cases/q2/0-small-tree --graphics
```

## Files and Directories
- `multiAgents.py`: Where you will implement your multi-agent search agents.
- `pacman.py`: Main file to run Pacman games, also describes the GameState.
- `game.py`: Contains the logic behind the Pacman world.
- `util.py`: Provides useful data structures for implementing search algorithms.
- `test_cases/`: Contains test cases for each question.
- Supporting files for graphics, ghost agents, keyboard interfaces, and the autograder.


## Welcome to Multi-Agent Pacman
Begin by playing a game of classic Pacman and experimenting with the provided ReflexAgent. This will serve as a foundation for the more complex agents you will develop.

### Reflex Agent
Start by improving the ReflexAgent to play more effectively by considering both food and ghost locations.

### Minimax
Implement the MinimaxAgent class, extending it to work with any number of ghosts and to an arbitrary search depth.

### Alpha-Beta Pruning
Enhance your agent with alpha-beta pruning to explore the minimax tree more efficiently.

### Expectimax
Implement the ExpectimaxAgent for scenarios where opponents may not make optimal decisions.

### Evaluation Function
Design a better evaluation function for Pacman to evaluate states rather than actions.

## Contributing
While this project was primarily an academic exercise, contributions or suggestions for improvements are welcome. Please open an issue or a pull request if you have suggestions.

## License
This project is licensed under the MIT License - see the LICENSE.md file for details.

## Acknowledgments
- Course staff and fellow students for their support and feedback.
- The original creators of Pacman for inspiring this educational project.
