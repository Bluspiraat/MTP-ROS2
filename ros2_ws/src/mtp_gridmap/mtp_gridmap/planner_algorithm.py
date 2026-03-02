import numpy as np
import heapq
from time import time

def _heuristic(node, goal):
    """
    Computes the manhattan distance from the input node to the goal node.

    Args:
        node: The for which the heuristic is to be calculated.
        goal: The goal node location.

    Returns:
        An integer value denoting the manhattan distance to the goal node
    """
    return abs(node[0] - goal[0]) + abs(node[1] - goal[1])

def _reconstruct_path(node, came_from):
    path = [node]
    while node in came_from:
        node = came_from[node]
        path.append(node)
    return path[::-1]

def _traversal_cost(current_cost, current_height, neighbour_cost, neighbour_height, distance):
    """
    Computes the average cost between the two cells

    Args:
        current_cost: The traversability cost of the current cell
        neighbour_cost: The traversability cost of the neighbour cell

    Returns:
        An float value denoting the average cost between the two cells
    """
    traversal_cost = (current_cost + neighbour_cost)/2.0
    angle_cost = np.arctan((neighbour_height-current_height)/distance)
    return traversal_cost + angle_cost

def _get_neighbors(location, grid):
    """
    Check the neighbors in all four perpendicular directions and returns if they are accessible.

    Args:
        location Tuple[int, int]: The row and column of the cell of which the neighbors are te be returned.

    Returns:
        neighbours Tuple[Tuple[int, int], ...]: The neighbors of the cell at the given location.
    """
    shape = grid.shape
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    neighbours = []
    for row_dir, col_dir in directions:
        row_nb, col_nb = location[0] + row_dir, location[1] + col_dir
        if shape[0] - 1 >= row_nb >= 0 and shape[1] - 1 >= col_nb >= 0:
            if grid[row_nb, col_nb] != 100:
                neighbours.append((row_nb, col_nb))
    return tuple(neighbours)

class AstarROS:
    def __init__(self, height_grid, trav_cost, resolution):
        self.height_grid = height_grid
        self.trav_cost = trav_cost
        self.resolution = resolution
        self.start = None
        self.goal = None
        self.path = []

    def start_search(self, start, goal):
        """
        Performs A* search on a grid given a start and goal coordinates.

        Args:
            start: Is a Tuple[int, int] representing the start coordinate.
            goal: Is a Tuple[int, int] representing the goal coordinate.

        Returns:
            A list of shortest path from the start to the goal coordinates. None if no path exists.
        """
        self.start = start
        self.goal = goal

        open_heap = []
        open_set = set()

        g_scores = {start: 0}  # Cost to travel to a certain node
        came_from = {}  # The parent node of a give node

        f_start = _heuristic(start, self.goal)
        heapq.heappush(open_heap, (f_start, start))
        open_set.add(start)  # Possible nodes to check
        closed_set = set()  # Set of nodes with fully optimized paths

        time_start = time()  # Track starting time of the search
        while open_heap:
            f_current, current = heapq.heappop(open_heap)  # Retrieves pair with the lowest f-score

            # Skip outdated entries --> Node might be in the closed set, but not popped from the heap
            if current in closed_set:
                continue

            # If current node is the goal node
            if current == goal:
                self.path = _reconstruct_path(current, came_from)
                print(f'Path found, length: {len(self.path)} in {time() - time_start:.4f} seconds')
                return self.path

            # Move current into closed set
            open_set.remove(current)
            closed_set.add(current)

            # Evaluate all neighbours of current node
            for neighbor in _get_neighbors(current, self.trav_cost):

                # If a neighbour is in the closed set, then its value is already optimal
                if neighbor in closed_set:
                    continue

                # The cost to move to the next cell is 1 as it only uses wind directions
                tentative_g = g_scores[current] + _traversal_cost(self.trav_cost[current], self.height_grid[current],
                                                                  self.trav_cost[neighbor], self.height_grid[neighbor],
                                                                  self.resolution)

                # The old score is the current best path to the neighbour, if it has not been computed before, then the
                # value None is returned
                g_old = g_scores.get(neighbor, None)

                # If tentative_g is larger than the old, nothing should happen as no optimization took place, however,
                # if the node has no associated cost or the cost reduced, then the node should be updated.
                if g_old is None or tentative_g < g_old:
                    g_scores[neighbor] = tentative_g
                    came_from[neighbor] = current

                    f_score: int = tentative_g + _heuristic(neighbor, self.goal)

                    # It might occur that a node exists multiple times in the heap with different f-scores.
                    # Re-processing of a node is avoided by the check if the node is in the closed set after selecting
                    # a new current node
                    heapq.heappush(open_heap, (f_score, neighbor))
                    open_set.add(neighbor)  # Adding does not matter, if it is already in there, it will be overwritten

        print('No path found')
        return None