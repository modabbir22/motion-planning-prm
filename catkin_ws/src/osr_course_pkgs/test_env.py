import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append('osr_examples/scripts/')
import environment_2d

np.random.seed(4)

env = environment_2d.Environment(10, 6, 5)

plt.clf()
env.plot()

q = env.random_query()

if q is not None:
    x_start, y_start, x_goal, y_goal = q
    env.plot_query(x_start, y_start, x_goal, y_goal)

plt.show(block=True)
input("Press Enter to close...")