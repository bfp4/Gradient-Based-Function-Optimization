"""
Gradient-Based Function Optimization
=====================================
Final Project: Gradient Descent on Convex and Non-Convex Functions

Functions used:
  1D Convex:     f(x) = x^2
  2D Convex:     f(x,y) = x^2 + 3y^2
  1D Non-Convex: Gramacy & Lee (2012) from sfu.ca/~ssurjano
  2D Non-Convex: Rastrigin function from sfu.ca/~ssurjano
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# ──────────────────────────────────────────────────────────────────────────────
# 1. DEFINE THE FUNCTIONS AND THEIR GRADIENTS
# ──────────────────────────────────────────────────────────────────────────────

# --- 1D Convex: f(x) = x^2 ---
def f_1d_convex(x):
    return x ** 2

def grad_1d_convex(x):
    return 2 * x

# --- 2D Convex: f(x,y) = x^2 + 3y^2 ---
def f_2d_convex(x, y):
    return x ** 2 + 3 * y ** 2

def grad_2d_convex(x, y):
    return np.array([2 * x, 6 * y])

# --- 1D Non-Convex: Gramacy & Lee (2012) ---
# f(x) = sin(10*pi*x) / (2*x) + (x - 1)^4
# Domain: x in [0.5, 2.5]
def f_1d_nonconvex(x):
    return np.sin(10 * np.pi * x) / (2 * x) + (x - 1) ** 4

def grad_1d_nonconvex(x):
    term1 = (10 * np.pi * np.cos(10 * np.pi * x) * (2 * x) - np.sin(10 * np.pi * x) * 2) / (4 * x ** 2)
    term2 = 4 * (x - 1) ** 3
    return term1 + term2

# --- 2D Non-Convex: Rastrigin ---
# f(x,y) = 20 + (x^2 - 10*cos(2*pi*x)) + (y^2 - 10*cos(2*pi*y))
# Domain: x,y in [-5.12, 5.12]
def f_2d_nonconvex(x, y):
    return 20 + (x ** 2 - 10 * np.cos(2 * np.pi * x)) + (y ** 2 - 10 * np.cos(2 * np.pi * y))

def grad_2d_nonconvex(x, y):
    dx = 2 * x + 20 * np.pi * np.sin(2 * np.pi * x)
    dy = 2 * y + 20 * np.pi * np.sin(2 * np.pi * y)
    return np.array([dx, dy])


# ──────────────────────────────────────────────────────────────────────────────
# 2. GRADIENT DESCENT ALGORITHM
# ──────────────────────────────────────────────────────────────────────────────

def gradient_descent_1d(grad_f, x0, lr, num_steps):
    """Run gradient descent in 1D. Returns the path of x values."""
    path = [x0]
    x = x0
    for _ in range(num_steps):
        x = x - lr * grad_f(x)
        path.append(x)
    return np.array(path)


def gradient_descent_2d(grad_f, x0, y0, lr, num_steps):
    """Run gradient descent in 2D. Returns arrays of (x, y) positions."""
    path_x = [x0]
    path_y = [y0]
    x, y = x0, y0
    for _ in range(num_steps):
        g = grad_f(x, y)
        x = x - lr * g[0]
        y = y - lr * g[1]
        path_x.append(x)
        path_y.append(y)
    return np.array(path_x), np.array(path_y)


# ──────────────────────────────────────────────────────────────────────────────
# 3. PLOTTING HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def set_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 12,
    })

set_style()

COLORS = ["#e74c3c", "#2ecc71", "#3498db", "#f39c12", "#9b59b6"]

CONV_TOL = 0.01  # convergence tolerance
MAX_STEPS = 50   # iteration cap for all experiments

def iters_to_converge(values, tol=CONV_TOL):
    """Return the iteration number where |f - f_final| first drops below tol.
    If it never converges within the run, return None."""
    f_final = values[-1]
    for k, v in enumerate(values):
        if abs(v - f_final) < tol:
            return k
    return None


def conv_tag(n):
    """Format iteration count for legend: show '50+ iters' if it hit the cap."""
    if n is None or n >= MAX_STEPS:
        return f"{MAX_STEPS}+ iters"
    return f"{n} iters"


# ──────────────────────────────────────────────────────────────────────────────
# 4. EXPERIMENTS
# ──────────────────────────────────────────────────────────────────────────────

def experiment_1d_convex():
    """1D Convex: f(x) = x^2"""
    x_plot = np.linspace(-6, 6, 300)

    # --- (a) Different initial points, same learning rate ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.plot(x_plot, f_1d_convex(x_plot), "k-", linewidth=2, label="f(x) = x²")

    starts = [-5, -2, 3, 5]
    lr = 0.1
    for i, x0 in enumerate(starts):
        path = gradient_descent_1d(grad_1d_convex, x0, lr, num_steps=MAX_STEPS)
        n = iters_to_converge(f_1d_convex(path))
        ax.plot(path, f_1d_convex(path), "o-", color=COLORS[i], markersize=5,
                label=f"start = {x0}  ({conv_tag(n)})")

    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_title(f"Different Starting Points  (lr = {lr})")
    ax.legend(fontsize=9)

    # --- (b) Same initial point, different learning rates ---
    ax = axes[1]
    ax.plot(x_plot, f_1d_convex(x_plot), "k-", linewidth=2, label="f(x) = x²")

    x0 = 5.0
    learning_rates = [0.01, 0.1, 0.5, 0.9]
    for i, lr in enumerate(learning_rates):
        path = gradient_descent_1d(grad_1d_convex, x0, lr, num_steps=MAX_STEPS)
        n = iters_to_converge(f_1d_convex(path))
        path_clipped = np.clip(path, -10, 10)
        ax.plot(path_clipped, f_1d_convex(path_clipped), "o-", color=COLORS[i],
                markersize=5, label=f"lr = {lr}  ({conv_tag(n)})")

    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_ylim(-1, 40)
    ax.set_title(f"Different Learning Rates  (start = {x0})")
    ax.legend(fontsize=9)

    fig.suptitle("1D Convex Function: f(x) = x²", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig("fig1_1d_convex.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig1_1d_convex.png")

    # --- Convergence plot ---
    fig, ax = plt.subplots(figsize=(8, 5))
    x0 = 5.0
    learning_rates = [0.01, 0.1, 0.5, 0.9]
    markers = ["o", "s", "D", "^"]
    linestyles = ["-", "--", "-.", ":"]
    for i, lr in enumerate(learning_rates):
        path = gradient_descent_1d(grad_1d_convex, x0, lr, num_steps=MAX_STEPS)
        values = f_1d_convex(path)
        n = iters_to_converge(values)
        ax.plot(values, linestyles[i], color=COLORS[i], marker=markers[i],
                markersize=5, linewidth=2, markevery=3, label=f"lr = {lr}  ({conv_tag(n)})")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("f(x)")
    ax.set_title("Convergence: f(x) = x²  (start = 5)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("fig2_1d_convex_convergence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig2_1d_convex_convergence.png")


def experiment_2d_convex():
    """2D Convex: f(x,y) = x^2 + 3y^2"""
    # Create contour grid
    xx = np.linspace(-6, 6, 200)
    yy = np.linspace(-6, 6, 200)
    X, Y = np.meshgrid(xx, yy)
    Z = f_2d_convex(X, Y)

    # --- (a) Different initial points ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.contour(X, Y, Z, levels=20, cmap="gray", alpha=0.5)
    starts = [(-5, 5), (4, -4), (-3, -5), (5, 2)]
    lr = 0.1
    for i, (x0, y0) in enumerate(starts):
        px, py = gradient_descent_2d(grad_2d_convex, x0, y0, lr, num_steps=MAX_STEPS)
        n = iters_to_converge(f_2d_convex(px, py))
        ax.plot(px, py, "o-", color=COLORS[i], markersize=4,
                label=f"({x0},{y0})  ({conv_tag(n)})")
        ax.plot(px[0], py[0], "s", color=COLORS[i], markersize=10)

    ax.plot(0, 0, "k*", markersize=15, label="minimum (0,0)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Different Starting Points  (lr = {lr})")
    ax.legend(fontsize=9)

    # --- (b) Different learning rates ---
    ax = axes[1]
    ax.contour(X, Y, Z, levels=20, cmap="gray", alpha=0.5)
    x0, y0 = 5.0, 5.0
    learning_rates = [0.01, 0.1, 0.15, 0.3]
    for i, lr in enumerate(learning_rates):
        px, py = gradient_descent_2d(grad_2d_convex, x0, y0, lr, num_steps=MAX_STEPS)
        n = iters_to_converge(f_2d_convex(px, py))
        px_c = np.clip(px, -8, 8)
        py_c = np.clip(py, -8, 8)
        ax.plot(px_c, py_c, "o-", color=COLORS[i], markersize=4,
                label=f"lr = {lr}  ({conv_tag(n)})")

    ax.plot(0, 0, "k*", markersize=15, label="minimum (0,0)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Different Learning Rates  (start = (5,5))")
    ax.legend(fontsize=9)
    ax.set_xlim(-7, 7)
    ax.set_ylim(-7, 7)

    fig.suptitle("2D Convex Function: f(x,y) = x² + 3y²", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig("fig3_2d_convex.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig3_2d_convex.png")

    # --- Convergence plot ---
    fig, ax = plt.subplots(figsize=(8, 5))
    x0, y0 = 5.0, 5.0
    learning_rates = [0.01, 0.1, 0.15, 0.3]
    for i, lr in enumerate(learning_rates):
        px, py = gradient_descent_2d(grad_2d_convex, x0, y0, lr, num_steps=MAX_STEPS)
        values = f_2d_convex(px, py)
        n = iters_to_converge(values)
        ax.plot(values, "o-", color=COLORS[i], markersize=4,
                label=f"lr = {lr}  ({conv_tag(n)})")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("f(x, y)")
    ax.set_title("Convergence: f(x,y) = x² + 3y²  (start = (5,5))")
    ax.legend()
    fig.tight_layout()
    fig.savefig("fig4_2d_convex_convergence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig4_2d_convex_convergence.png")


def experiment_1d_nonconvex():
    """1D Non-Convex: Gramacy & Lee (2012)"""
    x_plot = np.linspace(0.5, 2.5, 500)

    # Show the function shape
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(x_plot, f_1d_nonconvex(x_plot), "k-", linewidth=2, label="Gramacy & Lee")

    starts = [0.55, 0.9, 1.5, 2.3]
    lr = 0.001
    for i, x0 in enumerate(starts):
        path = gradient_descent_1d(grad_1d_nonconvex, x0, lr, num_steps=MAX_STEPS)
        path = np.clip(path, 0.5, 2.5)
        n = iters_to_converge(f_1d_nonconvex(path))
        ax.plot(path, f_1d_nonconvex(path), "o-", color=COLORS[i], markersize=3,
                label=f"start = {x0}  ({conv_tag(n)})")
        ax.plot(path[0], f_1d_nonconvex(path[0]), "s", color=COLORS[i], markersize=10)
        ax.plot(path[-1], f_1d_nonconvex(path[-1]), "D", color=COLORS[i], markersize=8)

    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_title(f"Different Starts → Different Local Minima  (lr = {lr})")
    ax.legend(fontsize=9)

    # Convergence from different starts
    ax = axes[1]
    for i, x0 in enumerate(starts):
        path = gradient_descent_1d(grad_1d_nonconvex, x0, lr, num_steps=MAX_STEPS)
        path = np.clip(path, 0.5, 2.5)
        values = f_1d_nonconvex(path)
        ax.plot(values, "-", color=COLORS[i], linewidth=2, label=f"start = {x0}")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("f(x)")
    ax.set_title("Convergence (each start lands at different minimum)")
    ax.legend(fontsize=9)

    fig.suptitle("1D Non-Convex: Gramacy & Lee (2012)", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig("fig5_1d_nonconvex.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig5_1d_nonconvex.png")


def experiment_2d_nonconvex():
    """2D Non-Convex: Rastrigin function"""
    xx = np.linspace(-5.12, 5.12, 400)
    yy = np.linspace(-5.12, 5.12, 400)
    X, Y = np.meshgrid(xx, yy)
    Z = f_2d_nonconvex(X, Y)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- (a) Contour plot with GD paths from different starts ---
    ax = axes[0]
    ax.contourf(X, Y, Z, levels=30, cmap="viridis", alpha=0.7)
    ax.contour(X, Y, Z, levels=30, colors="k", alpha=0.2, linewidths=0.5)

    starts = [(4, 4), (-3, 3), (2, -4), (-4, -2)]
    lr = 0.001
    for i, (x0, y0) in enumerate(starts):
        px, py = gradient_descent_2d(grad_2d_nonconvex, x0, y0, lr, num_steps=MAX_STEPS)
        n = iters_to_converge(f_2d_nonconvex(px, py))
        ax.plot(px, py, "o-", color=COLORS[i], markersize=3,
                label=f"({x0},{y0})→({px[-1]:.1f},{py[-1]:.1f})  ({conv_tag(n)})")
        ax.plot(px[0], py[0], "s", color=COLORS[i], markersize=10)
        ax.plot(px[-1], py[-1], "D", color=COLORS[i], markersize=8)

    ax.plot(0, 0, "w*", markersize=15, label="global min (0,0)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"GD Gets Stuck in Local Minima  (lr = {lr})")
    ax.legend(fontsize=7, loc="lower right")

    # --- (b) Convergence ---
    ax = axes[1]
    linestyles = ["-", "--", "-.", ":"]
    markers = ["o", "s", "D", "^"]
    for i, (x0, y0) in enumerate(starts):
        px, py = gradient_descent_2d(grad_2d_nonconvex, x0, y0, lr, num_steps=MAX_STEPS)
        values = f_2d_nonconvex(px, py)
        ax.plot(values, linestyles[i], color=COLORS[i], marker=markers[i],
                linewidth=2, markersize=5, markevery=3,
                label=f"start=({x0},{y0})")

    ax.axhline(y=0, color="black", linestyle="--", alpha=0.5, label="global min = 0")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("f(x, y)")
    ax.set_xlim(-1, 15)
    ax.set_title("Convergence (none reach global minimum!)")
    ax.legend(fontsize=8)

    fig.suptitle("2D Non-Convex: Rastrigin Function", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig("fig6_2d_nonconvex.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig6_2d_nonconvex.png")

    # --- 3D surface plot for visual appeal ---
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.8, edgecolor="none")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x,y)")
    ax.set_title("Rastrigin Function — Many Local Minima", fontsize=14, fontweight="bold")
    ax.view_init(elev=35, azim=135)
    fig.tight_layout()
    fig.savefig("fig7_rastrigin_3d.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig7_rastrigin_3d.png")


def step_size_demo():
    """Show why step size can't be too large or too small."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    x_plot = np.linspace(-6, 6, 300)
    x0 = 5.0

    configs = [
        ("Too Small (lr = 0.001)", 0.001, 60),
        ("Good (lr = 0.1)", 0.1, 60),
        ("Too Large (lr = 1.1)", 1.1, 15),
    ]

    for ax, (title, lr, steps) in zip(axes, configs):
        ax.plot(x_plot, f_1d_convex(x_plot), "k-", linewidth=2)
        path = gradient_descent_1d(grad_1d_convex, x0, lr, num_steps=steps)
        path_c = np.clip(path, -10, 10)
        ax.plot(path_c, f_1d_convex(path_c), "o-", color="#e74c3c", markersize=5)
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylim(-1, 40)

    fig.suptitle("Step Size Effect on f(x) = x²  (start = 5)", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig("fig8_step_size_demo.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig8_step_size_demo.png")


# ──────────────────────────────────────────────────────────────────────────────
# 5. RUN ALL EXPERIMENTS
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Gradient Descent Optimization — Generating Figures")
    print("=" * 60)

    experiment_1d_convex()
    experiment_2d_convex()
    experiment_1d_nonconvex()
    experiment_2d_nonconvex()
    step_size_demo()

    print("\nAll figures saved! Open the .png files to view them.")
