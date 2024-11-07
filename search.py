# search.py
# ---------

"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    """
    "*** YOUR CODE HERE ***"
    fringe = util.Stack()
    visited = set()
    fringe.push((problem.getStartState(), []))

    while not fringe.isEmpty():
        # pop a state and the path to reach it from the fringe
        currentState, currentPath = fringe.pop() 

        # return the path if the goal is reached
        if problem.isGoalState(currentState):
            print(currentPath)
            return currentPath  

        if currentState not in visited:
            visited.add(currentState)

            # push all unvisited successors to the fringe
            for successor, direction, cost in problem.getSuccessors(currentState):
                if successor not in visited:
                    # push the successor state and the updated path to it
                    fringe.push((successor, currentPath + [direction])) 

    return [] # if no solution is found, return an empty path 

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    fringe = util.Queue() 
    visited = set()       
    fringe.push((problem.getStartState(), []))

    while not fringe.isEmpty():
        currentState, path = fringe.pop()

        # return the path if the goal is reached
        if problem.isGoalState(currentState):
            print(path)
            return path

        # process the current state if not visited
        if currentState not in visited:
            visited.add(currentState)
            # expand the node and push each successor into the fringe
            for successor, direction, cost in problem.getSuccessors(currentState):
                if successor not in visited:
                    # push the successor state and updated path to the fringe
                    fringe.push((successor, path + [direction]))

    return [] # if no solution is found, return an empty path 

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    # priority Queue to store (state, path to reach state, cumulative cost) with priority as cost.
    fringe = util.PriorityQueue()
    fringe.push((problem.getStartState(), [], 0), 0)

    # dictionary to hold visited states and their corresponding minimum cumulative costs.
    # provide O(1) average time complexity for key lookups
    visited = {} 


    while not fringe.isEmpty():
        # pop the state with the lowest cumulative cost
        currentState, path, currentCost = fringe.pop()

        # if the current state is the goal, return the path to reach it.
        if problem.isGoalState(currentState):
            print(path)
            return path

        # only process this state if it's unvisited or reached with a lower cost.
        if currentState not in visited or currentCost < visited[currentState]:
            # update the visited dictionary with the cost to reach this state.
            visited[currentState] = currentCost

            # expand the current state and push successors to the priority queue.
            for child, direction, cost in problem.getSuccessors(currentState):
                newCost = currentCost + cost
                newPath = path + [direction]
                
                # add the successor to the fringe with the updated cost and path.
                fringe.push((child, newPath, newCost), newCost)

    return []
    # util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    # initialize start state, costs, and priority queue
    startState = problem.getStartState()
    backwardCost = 0 
    forwardCost = heuristic(startState, problem)  # estimated cost from the current node to goal
    totalCost = backwardCost + forwardCost

    fringe = util.PriorityQueue()
    fringe.push((startState, [], backwardCost), totalCost)
    explored = set()

    while not fringe.isEmpty():
        # pop the node with the lowest total cost (backwardCost + forwardCost)
        currentState, path, backwardCost = fringe.pop()
        
        # check the goal
        if problem.isGoalState(currentState):
            print(path)
            return path 
        
        # only process the node if it hasn't been explored
        if currentState not in explored:
            explored.add(currentState)
            
            for successor, action, stepCost in problem.getSuccessors(currentState):
                newBackwardCost = backwardCost + stepCost
                newForwardCost = heuristic(successor, problem)
                totalCost = newBackwardCost + newForwardCost

                # push successor into the fringe with updated path and cost
                fringe.push((successor, path + [action], newBackwardCost), totalCost)
    
    return []  # return empty list if no path to goal is found
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
