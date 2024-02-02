import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt


np.set_printoptions(precision=4)


def distance(a, b):
    return np.sqrt(np.sum((a - b) * (a - b)))


def get_adj_matrix(points):
    adj = np.zeros((len(points), len(points)))
    for i, point_a in enumerate(points):
        for j, point_b in enumerate(points):
            adj[i, j] = distance(point_a, point_b)
    return adj


class AntColony:

    def __init__(self, points, ants=20, alpha=1, beta=1, evaporation=0.6, q=5, ):

        self.EPSILON = 1e-10

        self.points = points
        self.n_nodes = len(points)
        self.node_idxs = np.arange(self.n_nodes)

        dist = get_adj_matrix(points)
        self.proximity = 1 / (dist + self.EPSILON)
        self.pheromones = np.full_like(self.proximity, 1/self.n_nodes)
        self.pheromones -= np.eye(self.n_nodes, self.n_nodes) * self.pheromones
        self.preferences = None

        # Parameters
        self.ants = ants
        self.alpha = alpha
        self.beta = beta
        self.evaporation = evaporation
        self.q = q

    def run_ant(self):
        node = np.random.choice(self.n_nodes)
        ant_preferences = deepcopy(self.preferences)
        ant_preferences[:, node] = 0
        path = [node]
        cost = 0
        for step in range(self.n_nodes - 1):
            # Calculate probability of all nodes from current one
            current_preferences = ant_preferences[node]
            probabilities = current_preferences / np.sum(current_preferences)

            # Choose next node based on probability
            node = np.random.choice(self.node_idxs, p=probabilities)

            # Remove current node from probabilities to avoid visiting it twice
            ant_preferences[:, node] = 0

            # Add selected node to path and increment cost
            path.append(node)
            cost += distance(self.points[path[-1]], self.points[path[-2]])

        return path, cost

    # points, ants=50, iterations=100, alpha=1.8, beta=3, evaporation=0.1, q=5
    def run_colony(self, iterations=100):
        best_path = {"path": None, "cost": np.inf}
        for i in range(iterations):
            # Calculate preferences based on pheromones and distance heuristic
            self.preferences = np.power(self.pheromones, self.alpha) * np.power(self.proximity, self.beta)

            # Initialize pheromone update matrix
            pheromone_update = np.zeros_like(self.preferences)
            for ant in range(self.ants):
                path, cost = self.run_ant()

                # Leave pheromones
                prev_node = path[0]
                reward = self.q / cost

                for node in path[1:]:
                    pheromone_update[prev_node, node] += reward
                    pheromone_update[node, prev_node] += reward
                    prev_node = node

                # Store best path
                if cost < best_path["cost"]:
                    best_path["path"] = path
                    best_path["cost"] = cost

            # Evaporation and pheromone update
            self.pheromones *= self.evaporation
            self.pheromones += pheromone_update
        return self.points[best_path["path"]], best_path["cost"]

if __name__ == '__main__':

    n = 40
    checkpoints = np.random.uniform(-20, 20, size=(n, 2))

    colony = AntColony(checkpoints, ants=80, alpha=1, beta=5, evaporation=0.8, q=40)
    path, cost = colony.run_colony(iterations=100)
    print(path, cost)

    plt.scatter(checkpoints[:, 0], checkpoints[:, 1])
    plt.plot(path[:, 0], path[:, 1])
    plt.tight_layout()
    plt.show()
