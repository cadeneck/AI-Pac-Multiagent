# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions

        scores = [self.evaluationFunction(gameState, action) for action in gameState.getLegalActions(0)]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        randomIndex = random.choice(bestIndices)
        "Add more of your code here if you want to"

        return legalMoves[randomIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # Extracting ghost information
        ghostPosition = successorGameState.getGhostPosition(1)
        ghostDistance = util.manhattanDistance(ghostPosition, newPos)
       
        # Initial score based on game state & Extracting food and capsule information
        score = successorGameState.getScore()
        capsules = currentGameState.getCapsules()
        foodList = newFood.asList()
        
        # Calculate distance to the nearest food
        nearestFoodDistance = min([util.manhattanDistance(food, newPos) for food in foodList]) if foodList else 0
       
        # Evaluate based on game elements
        score += max(ghostDistance, 3)
        score += 200 if newPos in capsules else 0
        score += 100 if len(foodList) < len(currentGameState.getFood().asList()) else 0
        score += 100 / nearestFoodDistance if nearestFoodDistance else 0
        score -= 10 if action == Directions.STOP else 0


        return score

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        return self.minimax(gameState, 0, self.depth)[1]
   
    def minimax(self, gameState, agentIndex, depth):
        # Base case
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState), "Stop"
       
        agentsNum = gameState.getNumAgents()
        agentIndex %= agentsNum
        # If it's the last agent's turn, decrease depth
        if agentIndex == agentsNum - 1:
            depth -= 1

        actionsValues = [(self.minimax(gameState.generateSuccessor(agentIndex, action), agentIndex + 1, depth)[0], action) for action in gameState.getLegalActions(agentIndex)]

        # If current agent is Max agent (agentIndex == 0), return max value, otherwise return min value
        return max(actionsValues) if agentIndex == 0 else min(actionsValues)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.minimax_alpha_beta(gameState, 0, self.depth)[1]
   
    def minimax_alpha_beta(self, gameState, agentIndex, depth, alpha=float('-inf'), beta=float('inf')):
        # Base case: return the evaluation if it's terminal state or max depth is reached.
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState), "Stop"

        # Adjust agent index and depth based on number of agents
        agentsNum = gameState.getNumAgents()
        agentIndex %= agentsNum
        if agentIndex == agentsNum - 1:
            depth -= 1

        # Initialize best value and best action for either max or min agent
        if agentIndex == 0:  # Max agent
            bestValue = float('-inf')
            actionValueCheck = lambda v: v > beta
            updateValue = lambda a, v: max(a, v)
        else:  # Min agent
            bestValue = float('inf')
            actionValueCheck = lambda v: v < alpha
            updateValue = lambda a, v: min(a, v)

        bestAction = None

        # Evaluate actions
        for action in gameState.getLegalActions(agentIndex):
            value, _ = self.minimax_alpha_beta(gameState.generateSuccessor(agentIndex, action), agentIndex + 1, depth, alpha, beta)
            
            # Update alpha or beta value
            if agentIndex == 0:
                alpha = updateValue(alpha, value)
            else:
                beta = updateValue(beta, value)

            # Alpha-beta pruning
            if actionValueCheck(value):
                return value, action

            # Update best value and action
            if (agentIndex == 0 and value > bestValue) or (agentIndex != 0 and value < bestValue):
                bestValue, bestAction = value, action

        return bestValue, bestAction


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.expectimax(gameState, 0, self.depth)[1]


    def expectimax(self, gameState, agentIndex, depth):
        # Base case: return the evaluation if it's terminal state or max depth is reached.
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState), "Stop"


        # Update agentIndex and depth for next turn
        agentsNum = gameState.getNumAgents()
        agentIndex %= agentsNum
        if agentIndex == agentsNum - 1:
            depth -= 1

        # Evaluate actions
        actions = gameState.getLegalActions(agentIndex)
        successorValues = [self.expectimax(gameState.generateSuccessor(agentIndex, action), agentIndex + 1, depth)[0] for action in actions]


        # Max agent: Return the max value and corresponding action
        if agentIndex == 0:
            maxValue = max(successorValues)
            bestAction = actions[successorValues.index(maxValue)]
            return maxValue, bestAction


        # Chance agent (previously Min agent): Return the expected value (average)
        else:
            expectedValue = sum(successorValues) / len(successorValues)
            return expectedValue, "Chance"


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # Current position
    currPos = currentGameState.getPacmanPosition()

    # Ghost distances and score calculation
    ghostPosition = currentGameState.getGhostPosition(1)
    ghostDistance = manhattanDistance(ghostPosition, currPos)
    ghostTimer = currentGameState.getGhostStates()[0].scaredTimer
    ghostScore = (70 - ghostDistance) if ghostTimer > 0 else -(70 - ghostDistance)

    # Capsule distances and score calculation
    capsules = currentGameState.getCapsules()
    capsuleDistances = [manhattanDistance(currPos, c) for c in capsules]
    capsuleScore = 160 - len(capsules) * 80 - min(capsuleDistances, default=0)

    # Food distances and score calculation
    foodList = currentGameState.getFood().asList()
    foodDistances = [manhattanDistance(currPos, food) for food in foodList]
    foodScore = 530 - len(foodList) * 10 - min(foodDistances, default=0)

    # Total score
    score = currentGameState.getScore() + ghostScore + capsuleScore + foodScore 

    return score

def manhattanDistance(xy1, xy2):
    """Returns the Manhattan distance between points xy1 and xy2"""
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

# Abbreviation
better = betterEvaluationFunction
