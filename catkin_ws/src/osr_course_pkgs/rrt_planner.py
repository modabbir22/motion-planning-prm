import numpy as np
import matplotlib.pyplot as plt
import sys
import random
import time

sys.path.append('osr_examples/scripts/')
import environment_2d


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


def path_length(path):
    if path is None or len(path) < 2:
        return None
    total = 0.0
    for i in range(len(path) - 1):
        total += dist(path[i], path[i + 1])
    return total

# RRT planner implementation


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



# Plotting the graph

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


# Manual queries (same queries for all environments)

manual_queries = [
    ((5.27, 5.62), (5.21, 0.64)),
    ((1.00, 1.00), (9.00, 5.50)),
    ((2.00, 5.50), (8.50, 1.00)),
    ((0.80, 3.00), (7.50, 4.20)),
]


# Experiment settings

random.seed(4)
np.random.seed(4)

num_environments = 3
step_size = 0.2
max_iters = 5000
goal_sample_rate = 0.1
show_plots = True


# Main loop over environments

for env_idx in range(num_environments):
    print("\n" + "=" * 60)
    print(f"ENVIRONMENT {env_idx + 1}")
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

    for query_id, (q_start, q_goal) in enumerate(valid_queries):
        print("\n" + "-" * 50)
        print(f"QUERY {query_id + 1} IN ENVIRONMENT {env_idx + 1}")
        print("-" * 50)

        start_time = time.time()
        nodes, parents, path = rrt_plan(
            env,
            q_start,
            q_goal,
            step_size=step_size,
            max_iters=max_iters,
            goal_sample_rate=goal_sample_rate
        )
        solve_time = time.time() - start_time

        print("Start:", q_start)
        print("Goal: ", q_goal)
        print("Tree nodes:", len(nodes))
        print(f"Solve time: {solve_time:.4f} sec")

        if path is None:
            print("No path found")
        else:
            print("Path found with", len(path), "points")
            print("Path length:", f"{path_length(path):.4f}")

        if show_plots:
            title = f"RRT - Env {env_idx + 1}, Query {query_id + 1}"
            plot_rrt(env, nodes, parents, q_start, q_goal, path, title=title)