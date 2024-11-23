# multiAgents.py
# --------------


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
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [
            index for index in range(len(scores)) if scores[index] == bestScore
        ]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

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

        # base score
        score = successorGameState.getScore()

        # evaluate food distances
        foodList = newFood.asList()
        if foodList:
            foodDistances = [util.manhattanDistance(newPos, food) for food in foodList]
            score += 10.0 / min(foodDistances)  # prioritize closer food

        # add incentives for eating food
        foodEaten = currentGameState.getNumFood() - successorGameState.getNumFood()
        score += 100 * foodEaten  # Reward for eating food

        # evaluate ghost distances
        for ghostState, scaredTime in zip(newGhostStates, newScaredTimes):
            ghostPos = ghostState.getPosition()
            ghostDistance = util.manhattanDistance(newPos, ghostPos)

            if scaredTime > 0:  # scared ghost
                if ghostDistance <= scaredTime:  # can potentially eat the ghost
                    score += 200 / (
                        ghostDistance + 1
                    )  # encourage approaching scared ghosts
            else:  # active ghost
                if ghostDistance <= 1:  # very close
                    score -= 1000  # major penalty for dying
                else:
                    score -= 1.0 / ghostDistance  # avoid active ghosts

        if action == Directions.STOP:
            score -= 50  # discourage standing still

        # encourage moving toward food for future rewards
        if foodList:
            clusterScore = sum(
                1.0 / util.manhattanDistance(newPos, food)
                for food in foodList
                if util.manhattanDistance(newPos, food) < 5
            )
            score += clusterScore

        # avoid dead ends
        legalMoves = successorGameState.getLegalActions()
        if len(legalMoves) == 1:  # a dead end if only one move is possible
            score -= 50

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

    def __init__(self, evalFn="scoreEvaluationFunction", depth="2"):
        self.index = 0  # Pacman is always agent index 0
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
        pacmanIndex = 0
        numAgents = gameState.getNumAgents()
        
        def isTerminalState(state, depth):
            """
            Checks if the game state is terminal or if the maximum depth is reached.
            """
            return state.isWin() or state.isLose() or depth == self.depth

        def min_value(state, depth, ghostIndex):
            """
            Minimizing agent logic (Ghosts).
            """
            if isTerminalState(state, depth):
                return self.evaluationFunction(state)

            legalActions = state.getLegalActions(ghostIndex)
            score = float("inf")

            for action in legalActions:
                successor = state.generateSuccessor(ghostIndex, action)
                if ghostIndex == numAgents - 1:  # Last ghost, next is Pacman's turn
                    score = min(score, max_value(successor, depth + 1))
                else:  # More ghosts to process
                    score = min(score, min_value(successor, depth, ghostIndex + 1))

            return score if legalActions else self.evaluationFunction(state)

        def max_value(state, depth):
            """
            Maximizing agent logic (Pacman).
            """
            if isTerminalState(state, depth):
                return self.evaluationFunction(state)

            legalActions = state.getLegalActions(pacmanIndex)
            score = float("-inf")

            for action in legalActions:
                successor = state.generateSuccessor(pacmanIndex, action)
                score = max(score, min_value(successor, depth, 1))  # Ghosts start at index 1

            return score if legalActions else self.evaluationFunction(state)

        # Top-level logic to determine the best action for Pacman
        bestAction = None
        bestScore = float("-inf")

        legalActions = gameState.getLegalActions(pacmanIndex)

        for action in legalActions:
            successor = gameState.generateSuccessor(pacmanIndex, action)
            score = min_value(successor, 0, 1)  # Depth starts at 0, first ghost index = 1

            if score > bestScore:
                bestScore = score
                bestAction = action

        return bestAction

        util.raiseNotDefined()


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        pacmanIndex = 0  # pacman's index is always 0.
        numAgents = gameState.getNumAgents()  # total agents including ghosts.
        ghostIndices = range(1, numAgents)

        # initialize tracking for the best action and score.
        bestAction = None
        alpha = float('-inf')
        beta = float('inf')

        # max function for Pacman.
        def max_value(state, depth, alpha, beta):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            value = float('-inf')
            for action in state.getLegalActions(pacmanIndex):
                successor = state.generateSuccessor(pacmanIndex, action)
                value = max(value, min_value(successor, depth, ghostIndices[0], alpha, beta))
                if value > beta:
                    return value  
                alpha = max(alpha, value)

            return value

        # min function for ghosts.
        def min_value(state, depth, ghostIndex, alpha, beta):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            value = float('inf')
            for action in state.getLegalActions(ghostIndex):
                successor = state.generateSuccessor(ghostIndex, action)

                # if last ghost, move to pacman's turn.
                if ghostIndex == numAgents - 1:
                    value = min(value, max_value(successor, depth + 1, alpha, beta))
                else:
                    value = min(value, min_value(successor, depth, ghostIndex + 1, alpha, beta))

                if value < alpha:
                    return value 
                beta = min(beta, value)

            return value

        # iterate over all legal actions to choose the best action.
        bestScore = float('-inf')
        for action in gameState.getLegalActions(pacmanIndex):
            successor = gameState.generateSuccessor(pacmanIndex, action)
            score = min_value(successor, 0, ghostIndices[0], alpha, beta)

            if score > bestScore:
                bestScore = score
                bestAction = action

            alpha = max(alpha, bestScore)

        return bestAction

        util.raiseNotDefined()


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
        pacmanIndex = 0 
        ghostIndices = range(1, gameState.getNumAgents())  # indices for ghosts
        legalActions = gameState.getLegalActions(pacmanIndex)

        # if no legal actions are available, return "Stop"
        if not legalActions:
            return Directions.STOP

        def max_value(state, depth):
            """
            calc the maximum value for pacman at a given state.

            Args:
                state (GameState): The current game state.
                depth (int): The current depth in the search tree.

            Returns:
                float: The maximum score for paacman.
            """
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            legalActions = state.getLegalActions(pacmanIndex)
            if not legalActions:
                return self.evaluationFunction(state)

            maxScore = float("-inf")
            for action in legalActions:
                successor = state.generateSuccessor(pacmanIndex, action)
                score = expected_value(successor, depth, ghostIndices[0])
                maxScore = max(maxScore, score)

            return maxScore

        def expected_value(state, depth, ghostIndex):
            """
            calc the expected value for a ghost at a given state.

            Args:
                state (GameState): The current game state.
                depth (int): The current depth in the search tree.
                ghostIndex (int): The index of the current ghost.

            Returns:
                float: The expected score for the current ghost.
            """
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            legalActions = state.getLegalActions(ghostIndex)
            if not legalActions:
                return self.evaluationFunction(state)

            probability = 1.0 / len(legalActions)
            expectedScore = 0

            for action in legalActions:
                successor = state.generateSuccessor(ghostIndex, action)
                if ghostIndex == max(ghostIndices): 
                    expectedScore += probability * max_value(successor, depth + 1)
                else:
                    expectedScore += probability * expected_value(successor, depth, ghostIndex + 1)

            return expectedScore

        # determine the best action for Pacman
        bestAction = None
        bestScore = float("-inf")

        for action in legalActions:
            successor = gameState.generateSuccessor(pacmanIndex, action)
            score = expected_value(successor, 0, ghostIndices[0])

            if score > bestScore:
                bestScore = score
                bestAction = action

        return bestAction
        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pacmanPosition = currentGameState.getPacmanPosition()
    foods = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimers = [ghostState.scaredTimer for ghostState in ghostStates]
    ghostPositions = currentGameState.getGhostPositions()

    # weights for scoring components
    food_weight = 10         
    ghost_weight = -200      
    scared_ghost_weight = 150  
    food_bonus = 1000        
    ghost_proximity_penalty = 500  

    # start with the game's current score
    score = currentGameState.getScore()

    food_list = foods.asList()
    if food_list:
        # find the closest food distance
        closest_food_dist = min([util.manhattanDistance(pacmanPosition, food) for food in food_list])
        if closest_food_dist == 0:
            score += food_bonus  # immediate reward for eating food
        else:
            # reward increases as distance to food decreases
            score += food_weight / (closest_food_dist + 1)

    for ghostState, scaredTimer, ghostPosition in zip(ghostStates, scaredTimers, ghostPositions):
        ghost_dist = util.manhattanDistance(pacmanPosition, ghostPosition)
        if scaredTimer > 0:
            # reward for chasing scared ghosts (higher if closer)
            score += scared_ghost_weight / (ghost_dist + 1)
        else:
            # penalize proximity to active ghosts
            if ghost_dist <= 1:
                score -= ghost_proximity_penalty 
            else:
                # exponential decay for penalty as ghost distance increases
                score += ghost_weight / (ghost_dist ** 2 + 1)

    # return the final score
    return score
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
