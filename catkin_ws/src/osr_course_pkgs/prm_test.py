import numpy as np
import matplotlib.pyplot as plt
import sys
import random
import heapq
import time
from collections import defaultdict

sys.path.append('osr_examples/scripts/')
import environment_2d


def dist(p1, p2):
    """Simple Euclidean distance between two points."""
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) ** 0.5


def is_free(env, p):
    """Checks if a single point is clear of obstacles."""
    return not env.check_collision(p[0], p[1])


def random_free_point(env, xlim=(0, 10), ylim=(0, 6)):
    """Keep picking random coordinates until we hit an empty spot."""
    while True:
        x = random.uniform(xlim[0], xlim[1])
        y = random.uniform(ylim[0], ylim[1])
        if not env.check_collision(x, y):
            return (x, y)


def collision_free_segment(env, p1, p2, resolution=0.01):
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)

    d = np.linalg.norm(p2 - p1)
    if d == 0:
        return is_free(env, p1)

    n_steps = max(2, int(d / resolution))

    for i in range(n_steps + 1):
        t = i / n_steps
        p = (1 - t) * p1 + t * p2
        if env.check_collision(p[0], p[1]):
            return False

    return True


def path_length(path):
    if path is None or len(path) < 2:
        return None
    total = 0.0
    for i in range(len(path) - 1):
        total += dist(path[i], path[i + 1])
    return total


#PRM implementation

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


def connect_query_point(env, nodes, graph, q, connection_radius=1.8):
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
    pq = [(0.0, start_idx)]
    distances = {start_idx: 0.0}
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


def copy_graph(graph):
    return defaultdict(list, {k: list(v) for k, v in graph.items()})


#plotting the graph

def plot_prm(env, nodes, graph, q_start, q_goal, path=None, title="PRM"):
    plt.figure(figsize=(8, 5))
    env.plot()
    env.plot_query(q_start[0], q_start[1], q_goal[0], q_goal[1])

    for i in graph:
        for j in graph[i]:
            if i < j:
                x1, y1 = nodes[i]
                x2, y2 = nodes[j]
                plt.plot([x1, x2], [y1, y2], 'c-', linewidth=0.5)

    if len(nodes) >= 2:
        xs = [p[0] for p in nodes[:-2]]
        ys = [p[1] for p in nodes[:-2]]
        plt.plot(xs, ys, 'ko', markersize=3)

    if path is not None:
        px = [p[0] for p in path]
        py = [p[1] for p in path]
        plt.plot(px, py, 'b-', linewidth=3, zorder=5)
        plt.plot(px, py, 'bo', markersize=4, zorder=6)

    plt.title(title)
    plt.show(block=True)


#Manual queries used in this instance

manual_queries = [
    ((5.27, 5.62), (5.21, 0.64)),
    ((1.00, 1.00), (9.00, 5.50)),
    ((2.00, 5.50), (8.50, 1.00)),
    ((0.80, 3.00), (7.50, 4.20)),
]


#Main

random.seed(4)
np.random.seed(4)

#Random environment generated
num_environments = 3
num_samples = 300
connection_radius = 2.0

for env_idx in range(num_environments):
    print("\n" + "=" * 60)
    print(f"PRM - ENVIRONMENT {env_idx + 1}")
    print("=" * 60)

    env = environment_2d.Environment(10, 6, 5)

    valid_queries = []
    print("\nChecking manual queries in this environment...")
    for i, (q_start, q_goal) in enumerate(manual_queries):
        start_ok = is_free(env, q_start)
        goal_ok = is_free(env, q_goal)

        if start_ok and goal_ok:
            valid_queries.append((q_start, q_goal))
            print(f"  Query {i+1}: VALID   start={q_start}, goal={q_goal}")
        else:
            print(f"  Query {i+1}: SKIPPED start={q_start}, goal={q_goal}")
            if not start_ok:
                print("     -> start is inside an obstacle")
            if not goal_ok:
                print("     -> goal is inside an obstacle")

    if len(valid_queries) == 0:
        print("No valid manual queries for this environment.")
        continue

    build_start = time.time()
    base_nodes, base_graph = build_prm(env, num_samples=num_samples, connection_radius=connection_radius)
    build_time = time.time() - build_start

    edge_count = sum(len(base_graph[i]) for i in base_graph) // 2
    print(f"\nRoadmap built with {len(base_nodes)} nodes and {edge_count} edges")
    print(f"Roadmap build time: {build_time:.4f} sec")

    for query_id, (q_start, q_goal) in enumerate(valid_queries):
        print("\n" + "-" * 50)
        print(f"PRM QUERY {query_id + 1} IN ENVIRONMENT {env_idx + 1}")
        print("-" * 50)

        nodes = list(base_nodes)
        graph = copy_graph(base_graph)

        t0 = time.time()
        start_idx = connect_query_point(env, nodes, graph, q_start, connection_radius)
        goal_idx = connect_query_point(env, nodes, graph, q_goal, connection_radius)
        path = dijkstra(graph, nodes, start_idx, goal_idx)
        solve_time = time.time() - t0

        print("Start:", q_start)
        print("Goal: ", q_goal)
        print("Start neighbors:", len(graph[start_idx]))
        print("Goal neighbors: ", len(graph[goal_idx]))
        print(f"Solve time: {solve_time:.4f} sec")

        if path is None:
            print("No path found")
        else:
            print("Path found with", len(path), "points")
            print("Path length:", f"{path_length(path):.4f}")

        title = f"PRM - Env {env_idx + 1}, Query {query_id + 1}"
        plot_prm(env, nodes, graph, q_start, q_goal, path, title=title)