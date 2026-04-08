import numpy as np
import matplotlib.pyplot as plt
import sys
import random
import heapq
import time
from collections import defaultdict

sys.path.append('osr_examples/scripts/')
import environment_2d


# =========================================================
# Helper functions
# =========================================================

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


# =========================================================
# PRM
# =========================================================

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


def prm_single_query(env, q_start, q_goal, num_samples=200, connection_radius=1.8):
    t0 = time.time()

    nodes, graph = build_prm(env, num_samples=num_samples, connection_radius=connection_radius)
    start_idx = connect_query_point(env, nodes, graph, q_start, connection_radius)
    goal_idx = connect_query_point(env, nodes, graph, q_goal, connection_radius)
    path = dijkstra(graph, nodes, start_idx, goal_idx)

    total_time = time.time() - t0
    return nodes, graph, path, total_time


def prm_multiple_queries(env, queries, num_samples=200, connection_radius=1.8):
    t0 = time.time()
    base_nodes, base_graph = build_prm(env, num_samples=num_samples, connection_radius=connection_radius)
    build_time = time.time() - t0

    results = []

    for q_start, q_goal in queries:
        nodes = list(base_nodes)
        graph = copy_graph(base_graph)

        q0 = time.time()
        start_idx = connect_query_point(env, nodes, graph, q_start, connection_radius)
        goal_idx = connect_query_point(env, nodes, graph, q_goal, connection_radius)
        path = dijkstra(graph, nodes, start_idx, goal_idx)
        query_time = time.time() - q0

        results.append({
            "q_start": q_start,
            "q_goal": q_goal,
            "nodes": nodes,
            "graph": graph,
            "path": path,
            "query_time": query_time
        })

    total_time = build_time + sum(r["query_time"] for r in results)
    return base_nodes, base_graph, results, build_time, total_time


# =========================================================
# RRT
# =========================================================

def nearest_node(nodes, q_rand):
    best_idx = 0
    best_dist = float('inf')

    for i, node in enumerate(nodes):
        d = dist(node, q_rand)
        if d < best_dist:
            best_dist = d
            best_idx = i

    return best_idx


def steer(q_near, q_rand, step_size):
    dx = q_rand[0] - q_near[0]
    dy = q_rand[1] - q_near[1]
    d = (dx * dx + dy * dy) ** 0.5

    if d == 0:
        return None

    if d <= step_size:
        return q_rand

    ux = dx / d
    uy = dy / d
    return (q_near[0] + step_size * ux, q_near[1] + step_size * uy)


def reconstruct_path(nodes, parents, goal_idx):
    path = []
    curr = goal_idx
    while curr is not None:
        path.append(nodes[curr])
        curr = parents[curr]
    path.reverse()
    return path


def rrt_plan(env, q_start, q_goal, step_size=0.2, max_iters=5000, goal_sample_rate=0.1):
    nodes = [q_start]
    parents = {0: None}

    for _ in range(max_iters):
        if random.random() < goal_sample_rate:
            q_rand = q_goal
        else:
            q_rand = random_free_point(env)

        nearest_idx = nearest_node(nodes, q_rand)
        q_near = nodes[nearest_idx]

        q_new = steer(q_near, q_rand, step_size)
        if q_new is None:
            continue

        if not is_free(env, q_new):
            continue

        if not collision_free_segment(env, q_near, q_new):
            continue

        nodes.append(q_new)
        new_idx = len(nodes) - 1
        parents[new_idx] = nearest_idx

        if dist(q_new, q_goal) <= step_size:
            if collision_free_segment(env, q_new, q_goal):
                nodes.append(q_goal)
                goal_idx = len(nodes) - 1
                parents[goal_idx] = new_idx
                path = reconstruct_path(nodes, parents, goal_idx)
                return nodes, parents, path

    return nodes, parents, None


def rrt_single_query(env, q_start, q_goal, step_size=0.2, max_iters=5000, goal_sample_rate=0.1):
    t0 = time.time()
    nodes, parents, path = rrt_plan(
        env,
        q_start,
        q_goal,
        step_size=step_size,
        max_iters=max_iters,
        goal_sample_rate=goal_sample_rate
    )
    total_time = time.time() - t0
    return nodes, parents, path, total_time


def rrt_multiple_queries(env, queries, step_size=0.2, max_iters=5000, goal_sample_rate=0.1):
    results = []
    total_time = 0.0

    for q_start, q_goal in queries:
        t0 = time.time()
        nodes, parents, path = rrt_plan(
            env,
            q_start,
            q_goal,
            step_size=step_size,
            max_iters=max_iters,
            goal_sample_rate=goal_sample_rate
        )
        solve_time = time.time() - t0
        total_time += solve_time

        results.append({
            "q_start": q_start,
            "q_goal": q_goal,
            "nodes": nodes,
            "parents": parents,
            "path": path,
            "solve_time": solve_time
        })

    return results, total_time


# =========================================================
# Plotting
# =========================================================

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


def plot_rrt(env, nodes, parents, q_start, q_goal, path=None, title="RRT"):
    plt.figure(figsize=(8, 5))
    env.plot()
    env.plot_query(q_start[0], q_start[1], q_goal[0], q_goal[1])

    for child_idx, parent_idx in parents.items():
        if parent_idx is not None:
            x1, y1 = nodes[child_idx]
            x2, y2 = nodes[parent_idx]
            plt.plot([x1, x2], [y1, y2], 'm-', linewidth=0.7)

    xs = [p[0] for p in nodes]
    ys = [p[1] for p in nodes]
    plt.plot(xs, ys, 'ko', markersize=2)

    if path is not None:
        px = [p[0] for p in path]
        py = [p[1] for p in path]
        plt.plot(px, py, 'b-', linewidth=3, zorder=5)
        plt.plot(px, py, 'bo', markersize=4, zorder=6)

    plt.title(title)
    plt.show(block=True)


# =========================================================
# Manual queries shared by BOTH algorithms
# =========================================================

manual_queries = [
    ((5.27, 5.62), (5.21, 0.64)),
    ((1.00, 1.00), (9.00, 5.50)),
    ((2.00, 5.50), (8.50, 1.00)),
    ((0.80, 3.00), (7.50, 4.20)),
]


# =========================================================
# Experiment settings
# =========================================================

random.seed(4)
np.random.seed(4)

num_environments = 3

# PRM params
prm_num_samples = 200
prm_connection_radius = 1.8

# RRT params
rrt_step_size = 0.2
rrt_max_iters = 5000
rrt_goal_sample_rate = 0.1

show_plots = True


# =========================================================
# Main comparison loop
# =========================================================

for env_idx in range(num_environments):
    print("\n" + "=" * 70)
    print(f"ENVIRONMENT {env_idx + 1}")
    print("=" * 70)

    env = environment_2d.Environment(10, 6, 5)

    # Same valid queries for BOTH PRM and RRT
    valid_queries = []
    print("\nChecking shared manual queries in this environment...")
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
        print("No valid shared queries for this environment.")
        continue

    # -----------------------------------------------------
    # SINGLE-QUERY COMPARISON
    # -----------------------------------------------------
    print("\n" + "-" * 70)
    print("SINGLE-QUERY COMPARISON")
    print("-" * 70)

    single_q_start, single_q_goal = valid_queries[0]

    # PRM single query
    prm_nodes, prm_graph, prm_path, prm_single_time = prm_single_query(
        env,
        single_q_start,
        single_q_goal,
        num_samples=prm_num_samples,
        connection_radius=prm_connection_radius
    )

    # RRT single query
    rrt_nodes, rrt_parents, rrt_path, rrt_single_time = rrt_single_query(
        env,
        single_q_start,
        single_q_goal,
        step_size=rrt_step_size,
        max_iters=rrt_max_iters,
        goal_sample_rate=rrt_goal_sample_rate
    )

    print(f"Single query start: {single_q_start}")
    print(f"Single query goal:  {single_q_goal}")

    print("\nPRM (single query):")
    print(f"  Time: {prm_single_time:.4f} sec")
    print(f"  Success: {'Yes' if prm_path is not None else 'No'}")
    if prm_path is not None:
        print(f"  Path length: {path_length(prm_path):.4f}")

    print("\nRRT (single query):")
    print(f"  Time: {rrt_single_time:.4f} sec")
    print(f"  Success: {'Yes' if rrt_path is not None else 'No'}")
    if rrt_path is not None:
        print(f"  Path length: {path_length(rrt_path):.4f}")

    if show_plots:
        plot_prm(
            env, prm_nodes, prm_graph, single_q_start, single_q_goal,
            prm_path, title=f"PRM Single Query - Env {env_idx + 1}"
        )
        plot_rrt(
            env, rrt_nodes, rrt_parents, single_q_start, single_q_goal,
            rrt_path, title=f"RRT Single Query - Env {env_idx + 1}"
        )

    # -----------------------------------------------------
    # MULTIPLE-QUERY COMPARISON
    # -----------------------------------------------------
    print("\n" + "-" * 70)
    print("MULTIPLE-QUERY COMPARISON")
    print("-" * 70)

    # PRM multiple queries: build once, solve many
    prm_base_nodes, prm_base_graph, prm_results, prm_build_time, prm_total_time = prm_multiple_queries(
        env,
        valid_queries,
        num_samples=prm_num_samples,
        connection_radius=prm_connection_radius
    )

    # RRT multiple queries: solve fresh each time
    rrt_results, rrt_total_time = rrt_multiple_queries(
        env,
        valid_queries,
        step_size=rrt_step_size,
        max_iters=rrt_max_iters,
        goal_sample_rate=rrt_goal_sample_rate
    )

    prm_successes = sum(1 for r in prm_results if r["path"] is not None)
    rrt_successes = sum(1 for r in rrt_results if r["path"] is not None)

    print(f"Number of shared valid queries: {len(valid_queries)}")

    print("\nPRM (multiple queries):")
    print(f"  Roadmap build time: {prm_build_time:.4f} sec")
    print(f"  Total time (build + all queries): {prm_total_time:.4f} sec")
    print(f"  Successes: {prm_successes}/{len(valid_queries)}")

    print("\nRRT (multiple queries):")
    print(f"  Total time (all queries): {rrt_total_time:.4f} sec")
    print(f"  Successes: {rrt_successes}/{len(valid_queries)}")

    print("\nPer-query details:")
    for i, ((q_start, q_goal), prm_r, rrt_r) in enumerate(zip(valid_queries, prm_results, rrt_results)):
        print(f"\n  Query {i + 1}: start={q_start}, goal={q_goal}")

        print("    PRM:")
        print(f"      Query time: {prm_r['query_time']:.4f} sec")
        print(f"      Success: {'Yes' if prm_r['path'] is not None else 'No'}")
        if prm_r["path"] is not None:
            print(f"      Path length: {path_length(prm_r['path']):.4f}")

        print("    RRT:")
        print(f"      Solve time: {rrt_r['solve_time']:.4f} sec")
        print(f"      Success: {'Yes' if rrt_r['path'] is not None else 'No'}")
        if rrt_r["path"] is not None:
            print(f"      Path length: {path_length(rrt_r['path']):.4f}")

        if show_plots:
            plot_prm(
                env,
                prm_r["nodes"],
                prm_r["graph"],
                q_start,
                q_goal,
                prm_r["path"],
                title=f"PRM Multi Query {i + 1} - Env {env_idx + 1}"
            )
            plot_rrt(
                env,
                rrt_r["nodes"],
                rrt_r["parents"],
                q_start,
                q_goal,
                rrt_r["path"],
                title=f"RRT Multi Query {i + 1} - Env {env_idx + 1}"
            )