# 📄 Motion Planning with PRM, RRT, and Post-Processing

## 📌 Overview

This project implements and compares three motion planning components:

- **Probabilistic Roadmap (PRM)**
- **Rapidly-exploring Random Tree (RRT)**
- **Path Post-processing (Shortcutting)**

The goal is to compute collision-free paths for a point robot in a 2D environment with triangular obstacles.

The environment provides a collision-checking function:

```python
env.check_collision(x, y)


🧪 Experiments
Environments
3 randomly generated environments
Each contains triangular obstacles
Queries

Same queries used for all algorithms:

manual_queries = [
    ((5.27, 5.62), (5.21, 0.64)),
    ((1.00, 1.00), (9.00, 5.50)),
    ((2.00, 5.50), (8.50, 1.00)),
    ((0.80, 3.00), (7.50, 4.20)),
]

Invalid queries are skipped if they lie inside obstacles.

⚙️ How to Run
Step 1: Setup
cd ~/catkin_ws/src/osr_course_pkgs
pip install numpy matplotlib
Step 2: Run PRM
python prm_manual_examples.py
Step 3: Run RRT
python rrt_manual_examples.py
Step 4: Run Post-processing
python post_processing_shortcutting.py