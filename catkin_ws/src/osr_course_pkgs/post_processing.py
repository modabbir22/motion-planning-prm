import numpy as np
import matplotlib.pyplot as plt
import sys
import random
import heapq
import time
from collections import defaultdict

# Pull in the specific 2D environment setup
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


def collision_free_segment(env, p1, p2, resolution=0.002):
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


def sample_point_on_polyline(path):
    """
    Sample a random point along a polyline path and return:
    - segment index i such that point lies on [path[i], path[i+1]]
    - interpolation parameter alpha in [0, 1]
    - sampled point
    """
    seg_lengths = []
    total_len = 0.0

    for i in range(len(path) - 1):
        L = dist(path[i], path[i + 1])
        seg_lengths.append(L)
        total_len += L

    if total_len == 0:
        return 0, 0.0, path[0]

    s = random.uniform(0, total_len)
    accum = 0.0

    for i, L in enumerate(seg_lengths):
        if accum + L >= s:
            alpha = (s - accum) / L if L > 0 else 0.0
            p1 = np.array(path[i], dtype=float)
            p2 = np.array(path[i + 1], dtype=float)
            p = (1 - alpha) * p1 + alpha * p2
            return i, alpha, (float(p[0]), float(p[1]))
        accum += L

    return len(path) - 2, 1.0, path[-1]


def splice_path(path, i1, p1, i2, p2):
    """
    Replace the subpath between sampled points p1 and p2
    by the direct segment [p1, p2].
    i1, i2 are segment indices with i1 <= i2.
    """
    new_path = []

    # Keep prefix up to path[i1]
    for k in range(i1 + 1):
        new_path.append(path[k])

    # Replace end of segment i1 with p1 if needed
    if dist(new_path[-1], p1) > 1e-9:
        new_path.append(p1)

    # Direct shortcut to p2
    if dist(new_path[-1], p2) > 1e-9:
        new_path.append(p2)

    # Continue from end of segment i2
    for k in range(i2 + 1, len(path)):
        if dist(new_path[-1], path[k]) > 1e-9:
            new_path.append(path[k])

    return new_path


def shortcut_path(env, path, maxrep=100):
    """
    Implements path shortcutting:
    repeatedly pick two random points along the path;
    if the direct straight segment is collision-free,
    replace the original portion by that segment.
    """
    if path is None or len(path) < 2:
        return path

    path = list(path)

    for _ in range(maxrep):
        if len(path) < 2:
            break

        i1, a1, q1 = sample_point_on_polyline(path)
        i2, a2, q2 = sample_point_on_polyline(path)

        # Order points along the path
        if (i1 > i2) or (i1 == i2 and a1 > a2):
            i1, i2 = i2, i1
            a1, a2 = a2, a1
            q1, q2 = q2, q1

        # Ignore degenerate choices
        if i1 == i2 and abs(a1 - a2) < 1e-6:
            continue

        if collision_free_segment(env, q1, q2):
            path = splice_path(path, i1, q1, i2, q2)

    return path


def build_prm(env, num_samples=300, connection_radius=2.0):
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


def plot_path_comparison(env, q_start, q_goal, original_path, shortcut_path_result, title="Path Shortcutting"):
    plt.figure(figsize=(8, 5))
    env.plot()
    env.plot_query(q_start[0], q_start[1], q_goal[0], q_goal[1])

    if original_path is not None:
        ox = [p[0] for p in original_path]
        oy = [p[1] for p in original_path]
        plt.plot(ox, oy, 'b--', linewidth=2, label='Original PRM path')
        plt.plot(ox, oy, 'bo', markersize=3)

    if shortcut_path_result is not None:
        sx = [p[0] for p in shortcut_path_result]
        sy = [p[1] for p in shortcut_path_result]
        plt.plot(sx, sy, 'g-', linewidth=3, label='Shortcut path')
        plt.plot(sx, sy, 'go', markersize=4)

    plt.title(title)
    plt.legend()
    plt.show(block=True)


manual_queries = [
    ((5.27, 5.62), (5.21, 0.64)),
    ((1.00, 1.00), (9.00, 5.50)),
    ((2.00, 5.50), (8.50, 1.00)),
    ((0.80, 3.00), (7.50, 4.20)),
]


random.seed(4)
np.random.seed(4)

num_environments = 3
show_plots = True

# PRM parameters
prm_num_samples = 300
prm_connection_radius = 2.0

# Shortcutting parameter
shortcut_iterations = 100

for env_idx in range(num_environments):
    print("\n" + "=" * 70)
    print(f"POST-PROCESSING - ENVIRONMENT {env_idx + 1}")
    print("=" * 70)

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
        print("No valid queries in this environment.")
        continue

    for query_id, (q_start, q_goal) in enumerate(valid_queries):
        print("\n" + "-" * 70)
        print(f"QUERY {query_id + 1} IN ENVIRONMENT {env_idx + 1}")
        print("-" * 70)

        # PRM path timing
        
        t0 = time.perf_counter()

        prm_nodes, prm_graph = build_prm(
            env,
            num_samples=prm_num_samples,
            connection_radius=prm_connection_radius
        )
        prm_start_idx = connect_query_point(env, prm_nodes, prm_graph, q_start, prm_connection_radius)
        prm_goal_idx = connect_query_point(env, prm_nodes, prm_graph, q_goal, prm_connection_radius)
        prm_path = dijkstra(prm_graph, prm_nodes, prm_start_idx, prm_goal_idx)

        t1 = time.perf_counter()
        prm_generation_time = t1 - t0

        # Shortcut timing
     
        t2 = time.perf_counter()
        prm_short = shortcut_path(env, prm_path, maxrep=shortcut_iterations)
        t3 = time.perf_counter()

        prm_shortcut_time = t3 - t2
        prm_total_time = prm_generation_time + prm_shortcut_time

        print("Start:", q_start)
        print("Goal: ", q_goal)

        print("\nPRM:")
        if prm_path is None:
            print("  No original path found")
            print(f"  PRM generation time: {prm_generation_time:.6f} seconds")
            print(f"  Shortcut time:       {prm_shortcut_time:.6f} seconds")
            print(f"  Total time:          {prm_total_time:.6f} seconds")
        else:
            orig_len = path_length(prm_path)
            short_len = path_length(prm_short)

            print(f"  Original path points: {len(prm_path)}")
            print(f"  Original path length: {orig_len:.4f}")
            print(f"  Shortcut path points: {len(prm_short)}")
            print(f"  Shortcut path length: {short_len:.4f}")
            print(f"  Improvement: {orig_len - short_len:.4f}")

            print(f"  PRM generation time: {prm_generation_time:.6f} seconds")
            print(f"  Shortcut time:       {prm_shortcut_time:.6f} seconds")
            print(f"  Total time:          {prm_total_time:.6f} seconds")

        if show_plots and prm_path is not None:
            plot_path_comparison(
                env,
                q_start,
                q_goal,
                prm_path,
                prm_short,
                title=f"PRM Shortcutting - Env {env_idx + 1}, Query {query_id + 1}"
            )