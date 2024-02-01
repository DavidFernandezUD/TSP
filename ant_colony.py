import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

EPSILON = 1e-10
np.set_printoptions(precision=4)


def distance(a, b):
    return np.sqrt(np.sum((a-b) * (a-b)))


def get_adj_matrix(points):
    adj = np.zeros((len(points), len(points)))
    for i, point_a in enumerate(points):
        for j, point_b in enumerate(points):
            adj[i, j] = distance(point_a, point_b)
    return adj


def run_colony(points, ants=50, iterations=100, alpha=1.8, beta=3, evaporation=0.1, q=5):

    n_nodes = len(points)
    node_idxs = np.arange(n_nodes)

    # Proximity
    dist = get_adj_matrix(points)
    proximity = 1 / (dist + EPSILON)

    # Pheromones
    pheromones = np.full_like(dist, 1/n_nodes)
    pheromones -= np.eye(n_nodes, n_nodes) * pheromones

    best_path = {"path": None, "cost": np.inf}

    for i in range(iterations):
        preferences = np.power(pheromones, alpha) * np.power(proximity, beta)
        pheromone_update = np.zeros_like(preferences)
        for ant in range(ants):
            node = np.random.choice(node_idxs)
            ant_preferences = deepcopy(preferences)
            ant_preferences[:, node] = 0
            path = [node]
            cost = 0
            for step in range(n_nodes-1):
                current_preferences = ant_preferences[node]
                probabilities = current_preferences / np.sum(current_preferences)
                node = np.random.choice(node_idxs, p=probabilities)
                ant_preferences[:, node] = 0
                path.append(node)
                cost += distance(points[path[-1]], points[path[-2]])

            # Leave Pheromones
            prev_node = 0
            reward = q / cost

            for node in path:
                pheromone_update[prev_node, node] += reward
                pheromone_update[node, prev_node] += reward
                prev_node = node

            # Store best path
            if cost < best_path["cost"]:
                best_path["path"] = path
                best_path["cost"] = cost

        # Evaporation
        pheromones *= evaporation
        pheromones += pheromone_update
    return points[best_path["path"]], best_path["cost"]

if __name__ == '__main__':

    points = np.random.uniform(-20, 20, size=(50, 2))

    path, cost = run_colony(points)
    print(path, cost)

    plt.scatter(points[:, 0], points[:, 1])
    plt.plot(path[:, 0], path[:, 1])
    plt.tight_layout()
    plt.show()
