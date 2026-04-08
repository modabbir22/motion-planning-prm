import numpy as np
import matplotlib.pyplot as plt
import sys
import random
import heapq
from collections import defaultdict

sys.path.append('osr_examples/scripts/')
import environment_2d


# -----------------------------
# Helper functions
# -----------------------------

def dist(p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) ** 0.5


def is_free(env, p):
    return not env.check_collision(p[0], p[1])


def random_free_point(env, xlim=(0, 10), ylim=(0, 6)):
    while True:
        x = random.uniform(xlim[0], xlim[1])
        y = random.uniform(ylim[0], ylim[1])
        if not env.check_collision(x, y):
            return (x, y)


def collision_free_segment(env, p1, p2, step=0.05):
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)

    d = np.linalg.norm(p2 - p1)
    if d == 0:
        return is_free(env, p1)

    n_steps = max(2, int(d / step))

    for i in range(n_steps + 1):
        t = i / n_steps
        p = (1 - t) * p1 + t * p2
        if env.check_collision(p[0], p[1]):
            return False

    return True


# -----------------------------
# PRM functions
# -----------------------------

def build_prm(env, num_samples=200, connection_radius=1.8):
    nodes = [random_free_point(env) for _ in range(num_samples)]
    graph = defaultdict(list)

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if dist(nodes[i], nodes[j]) <= connection_radius:
                if collision_free_segment(env, nodes[i], nodes[j]):
                    graph[i].append(j)
                    graph[j].append(i)

    return nodes, graph


def connect_query_point(env, nodes, graph, q, connection_radius):
    q_idx = len(nodes)
    nodes.append(q)
    graph[q_idx] = []

    for i in range(len(nodes) - 1):
        if dist(nodes[i], q) <= connection_radius:
            if collision_free_segment(env, nodes[i], q):
                graph[i].append(q_idx)
                graph[q_idx].append(i)

    return q_idx


def dijkstra(graph, nodes, start_idx, goal_idx):
    pq = [(0, start_idx)]
    distances = {start_idx: 0}
    parent = {start_idx: None}

    while pq:
        curr_dist, u = heapq.heappop(pq)

        if u == goal_idx:
            break

        if curr_dist > distances.get(u, float('inf')):
            continue

        for v in graph[u]:
            weight = dist(nodes[u], nodes[v])
            new_dist = curr_dist + weight

            if new_dist < distances.get(v, float('inf')):
                distances[v] = new_dist
                parent[v] = u
                heapq.heappush(pq, (new_dist, v))

    if goal_idx not in parent:
        return None

    path = []
    curr = goal_idx
    while curr is not None:
        path.append(nodes[curr])
        curr = parent[curr]

    path.reverse()
    return path


def plot_prm(env, nodes, graph, q_start, q_goal, path=None):
    plt.clf()
    env.plot()
    env.plot_query(q_start[0], q_start[1], q_goal[0], q_goal[1])

    # plot roadmap edges
    for i in graph:
        for j in graph[i]:
            if i < j:
                x1, y1 = nodes[i]
                x2, y2 = nodes[j]
                plt.plot([x1, x2], [y1, y2], 'c-', linewidth=0.5)

    # plot sampled nodes (exclude last 2 if they are start/goal)
    if len(nodes) >= 2:
        xs = [p[0] for p in nodes[:-2]]
        ys = [p[1] for p in nodes[:-2]]
        plt.plot(xs, ys, 'ko', markersize=3)

    # plot final path
    if path is not None:
        px = [p[0] for p in path]
        py = [p[1] for p in path]
        plt.plot(px, py, 'b-', linewidth=3, zorder=5)
        plt.plot(px, py, 'bo', markersize=5, zorder=6)

    plt.show(block=True)


# -----------------------------
# Main
# -----------------------------

random.seed(4)
np.random.seed(4)

env = environment_2d.Environment(10, 6, 5)

q = env.random_query()
if q is None:
    print("Could not generate query")
    exit()

x_start, y_start, x_goal, y_goal = q
q_start = (x_start, y_start)
q_goal = (x_goal, y_goal)

print("Start:", q_start)
print("Goal:", q_goal)

# build roadmap
nodes, graph = build_prm(env, num_samples=200, connection_radius=1.8)

# connect start and goal
start_idx = connect_query_point(env, nodes, graph, q_start, 1.8)
goal_idx = connect_query_point(env, nodes, graph, q_goal, 1.8)

print("Start neighbors:", len(graph[start_idx]))
print("Goal neighbors:", len(graph[goal_idx]))

# find shortest path in roadmap
path = dijkstra(graph, nodes, start_idx, goal_idx)

if path is None:
    print("No path found")
else:
    print("Path found with", len(path), "points")
    print("First point:", path[0])
    print("Last point:", path[-1])

plot_prm(env, nodes, graph, q_start, q_goal, path)