"""
=============================================================================
Project O: Gradient-Based Function Optimization
=============================================================================

PURPOSE
-------
Implement vanilla gradient descent and investigate its behavior on convex and
non-convex functions.  We study:
  1. How initial point x₀ affects convergence.
  2. How learning rate α affects convergence (or divergence).
  3. A practical step-size guideline based on the Lipschitz constant.
  4. Why GD fails on highly multi-modal non-convex landscapes.

FUNCTIONS STUDIED
-----------------
Convex (guaranteed global convergence):
  • f(x) = x²                    — classic quadratic bowl
  • f(x) = (1−x)² + 100(x²−x)²  — elongated valley (Rosenbrock-like)
  • f(x) = |x| + x²              — convex but non-smooth at x=0

Non-convex (sourced from sfu.ca/~ssurjano):
  • Rastrigin  — periodic cosine bumps, ~10 local minima in [−5,5]
  • Ackley     — near-flat plateau with deep central well
  • Griewank   — regular sinusoidal local minima
  • Schwefel   — deceptive; global min far from next-best local min

=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 1.  FUNCTION DEFINITIONS
# ---------------------------------------------------------------------------

class Function:
    """Container for a function, its gradient, domain, and metadata."""
    def __init__(self, name, f, grad, xmin, xmax, ymin, ymax,
                 true_optimum_x, true_optimum_f, convex=True, description="",
                 default_lr=0.05):
        self.name = name
        self.f = f
        self.grad = grad
        self.xmin, self.xmax = xmin, xmax
        self.ymin, self.ymax = ymin, ymax
        self.true_optimum_x = true_optimum_x
        self.true_optimum_f = true_optimum_f
        self.convex = convex
        self.description = description
        self.default_lr = default_lr


# ── Convex Functions ─────────────────────────────────────────────────────────

quadratic = Function(
    name="Quadratic f(x)=x²",
    f=lambda x: x**2,
    grad=lambda x: 2*x,
    xmin=-4.5, xmax=4.5, ymin=-1, ymax=20,
    true_optimum_x=0.0, true_optimum_f=0.0, convex=True, default_lr=0.2,
    description=(
        "f(x) = x²\n"
        "Gradient: f'(x) = 2x\n"
        "Lipschitz constant L = 2  →  safe α < 1/L = 0.5\n"
        "Unique global min at x*=0. For α < 0.5, GD converges from any x₀."
    )
)

rosenbrock_1d = Function(
    name="Rosenbrock-like (1D)",
    f=lambda x: (1-x)**2 + 100*(x**2 - x)**2,
    grad=lambda x: -2*(1-x) + 100*2*(x**2-x)*(2*x-1),
    xmin=-0.5, xmax=2.0, ymin=-5, ymax=300,
    true_optimum_x=1.0, true_optimum_f=0.0, convex=True, default_lr=0.0005,
    description=(
        "f(x) = (1−x)² + 100(x²−x)²\n"
        "Gradient is near-zero inside the shallow valley → very slow convergence.\n"
        "Need extremely small α to avoid overshooting the narrow basin."
    )
)

abs_quad = Function(
    name="Non-smooth: |x| + x²",
    f=lambda x: np.abs(x) + x**2,
    grad=lambda x: np.sign(x) + 2*x,
    xmin=-4, xmax=4, ymin=-0.5, ymax=18,
    true_optimum_x=0.0, true_optimum_f=0.0, convex=True, default_lr=0.15,
    description=(
        "f(x) = |x| + x²\n"
        "Convex but non-differentiable at x=0. We use the subgradient ∂f = sign(x)+2x.\n"
        "Still converges but may chatter near the kink."
    )
)

# ── Non-convex Functions (sfu.ca/~ssurjano) ──────────────────────────────────

rastrigin = Function(
    name="Rastrigin",
    f=lambda x: 10 + x**2 - 10*np.cos(2*np.pi*x),
    grad=lambda x: 2*x + 10*2*np.pi*np.sin(2*np.pi*x),
    xmin=-5.12, xmax=5.12, ymin=-3, ymax=55,
    true_optimum_x=0.0, true_optimum_f=0.0, convex=False,
    description=(
        "f(x) = 10 + x² − 10cos(2πx)\n"
        "Many local minima spaced ~1 unit apart. The cosine term creates periodic traps.\n"
        "GD almost always gets stuck; outcome depends entirely on x₀."
    )
)

ackley = Function(
    name="Ackley",
    f=lambda x: -20*np.exp(-0.2*np.abs(x)) - np.exp(np.cos(2*np.pi*x)) + 20 + np.e,
    grad=lambda x: (
        -20*np.exp(-0.2*np.abs(x))*(-0.2)*np.sign(x)
        - np.exp(np.cos(2*np.pi*x))*(-np.sin(2*np.pi*x))*2*np.pi
    ),
    xmin=-5, xmax=5, ymin=-1, ymax=15,
    true_optimum_x=0.0, true_optimum_f=0.0, convex=False,
    description=(
        "f(x): two exponential terms creating a near-flat outer plateau.\n"
        "Far from origin gradient ≈ 0  →  GD barely moves.\n"
        "Near origin: sharp gradients. Highly sensitive to α and x₀."
    )
)

griewank = Function(
    name="Griewank",
    f=lambda x: x**2/4000 - np.cos(x) + 1,
    grad=lambda x: x/2000 + np.sin(x),
    xmin=-8, xmax=8, ymin=-0.5, ymax=4,
    true_optimum_x=0.0, true_optimum_f=0.0, convex=False,
    description=(
        "f(x) = x²/4000 − cos(x) + 1\n"
        "Regular cosine bumps superimposed on a parabola.\n"
        "Gradient oscillates in sign, causing GD to zigzag between basins."
    )
)

schwefel = Function(
    name="Schwefel",
    f=lambda x: 418.9829 - x*np.sin(np.sqrt(np.abs(x))),
    grad=lambda x: -(np.sin(np.sqrt(np.abs(x))) + x*np.cos(np.sqrt(np.abs(x)))*np.sign(x)*0.5/np.sqrt(np.maximum(np.abs(x), 1e-12))),
    xmin=-500, xmax=500, ymin=-50, ymax=900,
    true_optimum_x=420.9687, true_optimum_f=0.0, convex=False,
    description=(
        "f(x) = 418.98 − x·sin(√|x|)\n"
        "Global min far from next-best local min. GD is almost certain to converge\n"
        "to a deceptive local minimum. Very large domain adds further difficulty."
    )
)

CONVEX_FNS    = [quadratic, rosenbrock_1d, abs_quad]
NONCONVEX_FNS = [rastrigin, ackley, griewank, schwefel]


# ---------------------------------------------------------------------------
# 2.  GRADIENT DESCENT ALGORITHM
# ---------------------------------------------------------------------------

def gradient_descent(fn_obj, x0, lr, max_iter=500, tol=1e-9):
    """
    Vanilla (fixed-step) gradient descent.

    Parameters
    ----------
    fn_obj   : Function
        Object with callable .f and .grad
    x0       : float
        Starting point
    lr       : float
        Learning rate (step size α)
    max_iter : int
        Maximum number of iterations
    tol      : float
        Stop when |x_new − x_old| < tol

    Returns
    -------
    xs   : list of x values visited
    fs   : list of f(x) values visited
    iters: int, number of steps taken
    """
    x = float(x0)
    xs, fs = [x], [fn_obj.f(x)]
    for i in range(max_iter):
        g = fn_obj.grad(x)
        g = max(-1e8, min(1e8, g))   # clamp gradient to prevent overflow
        x_new = x - lr * g
        xs.append(x_new)
        try:
            fval = fn_obj.f(x_new)
        except (OverflowError, ValueError):
            fs.append(float('inf'))
            break
        fs.append(fval)
        if abs(x_new - x) < tol:
            return xs, fs, i + 1
        x = x_new
    return xs, fs, max_iter


# ---------------------------------------------------------------------------
# 3.  STEP-SIZE GUIDELINE (Backtracking Line Search / Armijo)
# ---------------------------------------------------------------------------

def backtracking_line_search(fn_obj, x, grad_x, alpha0=1.0, rho=0.5, c=1e-4):
    """
    Armijo backtracking: shrink α until sufficient decrease is satisfied.
    Condition: f(x − α·g) ≤ f(x) − c·α·||g||²

    This is the recommended automatic step-size strategy when L is unknown.
    """
    alpha = alpha0
    fx = fn_obj.f(x)
    g2 = grad_x**2
    for _ in range(50):
        if fn_obj.f(x - alpha*grad_x) <= fx - c*alpha*g2:
            return alpha
        alpha *= rho
    return alpha


def gradient_descent_backtracking(fn_obj, x0, max_iter=500, tol=1e-9):
    """GD with automatic step via Armijo backtracking."""
    x = float(x0)
    xs, fs = [x], [fn_obj.f(x)]
    for i in range(max_iter):
        g = fn_obj.grad(x)
        alpha = backtracking_line_search(fn_obj, x, g)
        x_new = x - alpha * g
        xs.append(x_new)
        fs.append(fn_obj.f(x_new))
        if abs(x_new - x) < tol:
            return xs, fs, i + 1
        x = x_new
    return xs, fs, max_iter


# ---------------------------------------------------------------------------
# 4.  VISUALIZATION HELPERS
# ---------------------------------------------------------------------------

PALETTE = {
    "fn":     "#185FA5",   # function curve
    "path":   "#BA7517",   # GD trajectory
    "start":  "#3B6D11",   # start marker
    "end":    "#D85A30",   # end marker
    "global": "#97C459",   # true optimum
}


def plot_function_with_path(ax, fn_obj, path_xs, path_fs, title=""):
    """Plot function curve and gradient-descent trajectory on the same axes."""
    xs = np.linspace(fn_obj.xmin, fn_obj.xmax, 600)
    ys = fn_obj.f(xs)
    ax.plot(xs, ys, color=PALETTE["fn"], lw=2, label="f(x)")
    ax.axhline(0, color="gray", lw=0.4, linestyle="--")

    # GD trajectory (project onto the curve)
    ax.plot(path_xs, path_fs, "o--", color=PALETTE["path"],
            markersize=3, lw=1.2, alpha=0.8, label="GD path")
    ax.plot(path_xs[0],  path_fs[0],  "o", color=PALETTE["start"],
            markersize=8, zorder=5, label=f"start x₀={path_xs[0]:.2f}")
    ax.plot(path_xs[-1], path_fs[-1], "s", color=PALETTE["end"],
            markersize=8, zorder=5, label=f"end   x={path_xs[-1]:.3f}")
    ax.plot(fn_obj.true_optimum_x, fn_obj.true_optimum_f,
            "*", color=PALETTE["global"], markersize=12, zorder=6, label="global min x*")

    ax.set_xlim(fn_obj.xmin, fn_obj.xmax)
    ax.set_ylim(fn_obj.ymin, fn_obj.ymax)
    ax.set_xlabel("x", fontsize=9)
    ax.set_ylabel("f(x)", fontsize=9)
    ax.set_title(title or fn_obj.name, fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3, lw=0.4)


def plot_loss_curve(ax, fs_list, labels, title="Convergence"):
    """Plot f(x) vs iteration for multiple runs."""
    colors = ["#185FA5","#BA7517","#993C1D","#3B6D11","#7F77DD"]
    for i, (fs, lab) in enumerate(zip(fs_list, labels)):
        ax.semilogy(range(len(fs)), np.maximum(fs, 1e-12),
                    lw=1.5, color=colors[i % len(colors)], label=lab)
    ax.set_xlabel("Iteration", fontsize=8)
    ax.set_ylabel("f(x)  [log scale]", fontsize=8)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, lw=0.4)


# ---------------------------------------------------------------------------
# 5.  EXPERIMENT 1 — CONVEX FUNCTION (f(x) = x²)
#     Vary α and x₀ independently
# ---------------------------------------------------------------------------

def exp1_convex_quadratic():
    fn = quadratic
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Experiment 1 — Convex f(x)=x²: Effect of α and x₀",
                 fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.38)

    # ── Panel A: vary α, fixed x₀=3 ────────────────────────────────────────
    lr_values = [0.01, 0.1, 0.45, 0.9, 1.1]
    ax_a = fig.add_subplot(gs[0, :2])
    fs_a, labs_a = [], []
    for lr in lr_values:
        xs, fs, it = gradient_descent(fn, x0=3.0, lr=lr, max_iter=150)
        fs_a.append(fs)
        labs_a.append(f"α={lr}")
    plot_loss_curve(ax_a, fs_a, labs_a, "Convergence: vary α (x₀=3)")

    # annotations
    ax_a.annotate("α=0.9 → slow oscillation", xy=(50, 0.05),
                  fontsize=7, color="#993C1D")
    ax_a.annotate("α=1.1 → DIVERGES", xy=(5, 10), fontsize=7, color="#A32D2D",
                  fontweight="bold")

    # ── Panel B: function plots for representative α ─────────────────────
    for col, lr in enumerate([0.1, 0.45, 0.9, 1.1]):
        ax = fig.add_subplot(gs[1, col])
        xs_p, fs_p, it = gradient_descent(fn, x0=3.0, lr=lr, max_iter=60)
        plot_function_with_path(ax, fn, xs_p, fs_p, f"α={lr} ({it} iters)")
        ax.tick_params(labelsize=7)

    # ── Panel C: vary x₀, fixed α=0.2 ───────────────────────────────────
    ax_c = fig.add_subplot(gs[0, 2:])
    x0_values = [-4, -2, 0.5, 2, 4]
    fs_c, labs_c = [], []
    for x0 in x0_values:
        xs, fs, it = gradient_descent(fn, x0=x0, lr=0.2, max_iter=100)
        fs_c.append(fs)
        labs_c.append(f"x₀={x0}")
    plot_loss_curve(ax_c, fs_c, labs_c, "Convergence: vary x₀ (α=0.2)")

    # ── Panel D: convergence heatmap ────────────────────────────────────
    ax_d = fig.add_subplot(gs[2, :])
    lrs  = np.linspace(0.005, 1.2, 40)
    x0s  = np.linspace(-4, 4, 30)
    errs = np.zeros((len(x0s), len(lrs)))
    for i, x0 in enumerate(x0s):
        for j, lr in enumerate(lrs):
            xs_r, fs_r, _ = gradient_descent(fn, x0=x0, lr=lr, max_iter=80)
            errs[i, j] = abs(xs_r[-1] - fn.true_optimum_x)

    errs = np.clip(errs, 1e-10, 1e3)
    im = ax_d.imshow(errs, aspect="auto", origin="lower",
                     extent=[lrs[0], lrs[-1], x0s[0], x0s[-1]],
                     norm=LogNorm(vmin=1e-8, vmax=1e3), cmap="RdYlGn_r")
    plt.colorbar(im, ax=ax_d, label="|x_final − x*|")
    ax_d.axvline(0.5, color="white", lw=1, linestyle="--",
                 label="α = 0.5 (theoretical limit 1/L)")
    ax_d.set_xlabel("Learning rate α", fontsize=9)
    ax_d.set_ylabel("Initial point x₀", fontsize=9)
    ax_d.set_title("Heatmap: convergence error after 80 iters  (green=good, red=diverged)",
                   fontsize=9, fontweight="bold")
    ax_d.legend(fontsize=8, loc="upper right")

    plt.savefig("exp1_convex_quadratic.png", dpi=130, bbox_inches="tight")
    plt.show()
    print("Saved: exp1_convex_quadratic.png")


# ---------------------------------------------------------------------------
# 6.  EXPERIMENT 2 — ALL CONVEX FUNCTIONS, backtracking comparison
# ---------------------------------------------------------------------------

def exp2_all_convex():
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Experiment 2 — Convex Functions: fixed-step vs backtracking",
                 fontsize=12, fontweight="bold")

    for col, fn in enumerate(CONVEX_FNS):
        # Fixed step — use each function's safe default lr
        xs_f, fs_f, it_f = gradient_descent(fn, x0=3.0, lr=fn.default_lr, max_iter=300)
        # Backtracking
        xs_b, fs_b, it_b = gradient_descent_backtracking(fn, x0=3.0, max_iter=300)

        ax_top = axes[0, col]
        plot_function_with_path(ax_top, fn, xs_b, fs_b,
                                f"{fn.name}\n(backtracking, {it_b} iters)")
        ax_top.tick_params(labelsize=7)

        ax_bot = axes[1, col]
        plot_loss_curve(ax_bot,
                        [fs_f, fs_b],
                        [f"fixed α={fn.default_lr} ({it_f} iters)",
                         f"backtracking ({it_b} iters)"],
                        "Loss curve comparison")
        ax_bot.tick_params(labelsize=7)

    plt.savefig("exp2_all_convex.png", dpi=130, bbox_inches="tight")
    plt.show()
    print("Saved: exp2_all_convex.png")


# ---------------------------------------------------------------------------
# 7.  EXPERIMENT 3 — NON-CONVEX FUNCTIONS
#     Show how GD gets trapped in local minima
# ---------------------------------------------------------------------------

def exp3_nonconvex():
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("Experiment 3 — Non-convex Benchmark Functions (sfu.ca/~ssurjano)",
                 fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(3, len(NONCONVEX_FNS), figure=fig,
                           hspace=0.5, wspace=0.35)

    # several starting points per function
    x0_candidates = [-4, -2, -0.5, 0.5, 2, 4]

    for col, fn in enumerate(NONCONVEX_FNS):
        # ── Row 0: function + GD paths from multiple starts ───────────────
        ax0 = fig.add_subplot(gs[0, col])
        xs_plot = np.linspace(fn.xmin, fn.xmax, 800)
        ys_plot = fn.f(xs_plot)
        ax0.plot(xs_plot, ys_plot, color=PALETTE["fn"], lw=1.8)
        ax0.plot(fn.true_optimum_x, fn.true_optimum_f, "*",
                 color=PALETTE["global"], markersize=12, zorder=6,
                 label="global min")
        ax0.set_ylim(fn.ymin, fn.ymax)
        ax0.set_title(fn.name, fontsize=9, fontweight="bold")
        ax0.set_xlabel("x", fontsize=8)
        ax0.set_ylabel("f(x)", fontsize=8)
        ax0.grid(True, alpha=0.25, lw=0.4)

        colors_x0 = ["#185FA5","#BA7517","#993C1D","#3B6D11","#7F77DD","#D85A30"]
        found_global = 0
        for ci, x0 in enumerate(x0_candidates):
            # scale lr to function domain
            lr = 0.01 if fn.name in ("Rastrigin", "Ackley", "Griewank") else 0.0001
            xs_p, fs_p, it = gradient_descent(fn, x0=x0, lr=lr, max_iter=300)
            ax0.plot(xs_p[-1], fn.f(xs_p[-1]), "v",
                     color=colors_x0[ci], markersize=7,
                     label=f"x₀={x0:.1f}→{xs_p[-1]:.2f}")
            ax0.plot([x0], [fn.f(x0)], "o", color=colors_x0[ci],
                     markersize=4, alpha=0.6)
            if abs(xs_p[-1] - fn.true_optimum_x) < 0.2:
                found_global += 1
        ax0.legend(fontsize=5.5, loc="upper right", ncol=1)

        # ── Row 1: convergence loss curves ───────────────────────────────
        ax1 = fig.add_subplot(gs[1, col])
        for ci, x0 in enumerate(x0_candidates):
            lr = 0.01 if fn.name in ("Rastrigin", "Ackley", "Griewank") else 0.0001
            _, fs_p, _ = gradient_descent(fn, x0=x0, lr=lr, max_iter=200)
            ax1.semilogy(range(len(fs_p)),
                         np.maximum(np.array(fs_p) - fn.true_optimum_f, 1e-12),
                         color=colors_x0[ci], lw=1.2, label=f"x₀={x0}")
        ax1.set_xlabel("Iteration", fontsize=8)
        ax1.set_ylabel("f(x)−f(x*)", fontsize=8)
        ax1.set_title("Convergence gap", fontsize=8)
        ax1.legend(fontsize=5.5)
        ax1.grid(True, alpha=0.3, lw=0.4)

        # ── Row 2: success rate across x₀ grid ───────────────────────────
        ax2 = fig.add_subplot(gs[2, col])
        n_grid = 200
        x0_grid = np.linspace(fn.xmin, fn.xmax, n_grid)
        final_xs = []
        lr = 0.01 if fn.name in ("Rastrigin", "Ackley", "Griewank") else 0.0001
        for x0g in x0_grid:
            xs_g, _, _ = gradient_descent(fn, x0=x0g, lr=lr, max_iter=500)
            final_xs.append(xs_g[-1])
        final_xs = np.array(final_xs)
        global_mask = np.abs(final_xs - fn.true_optimum_x) < 0.2
        ax2.scatter(x0_grid[global_mask],  final_xs[global_mask],
                    s=4, color="#3B6D11", label="→ global min", alpha=0.7)
        ax2.scatter(x0_grid[~global_mask], final_xs[~global_mask],
                    s=4, color="#D85A30", label="→ local min",  alpha=0.7)
        ax2.axhline(fn.true_optimum_x, color=PALETTE["global"],
                    lw=1, linestyle="--")
        success_pct = global_mask.mean() * 100
        ax2.set_title(f"Basin of attraction  ({success_pct:.0f}% global)",
                      fontsize=8, fontweight="bold")
        ax2.set_xlabel("Initial x₀", fontsize=8)
        ax2.set_ylabel("Final x", fontsize=8)
        ax2.legend(fontsize=6)
        ax2.grid(True, alpha=0.3, lw=0.4)

    plt.savefig("exp3_nonconvex.png", dpi=130, bbox_inches="tight")
    plt.show()
    print("Saved: exp3_nonconvex.png")


# ---------------------------------------------------------------------------
# 8.  STEP SIZE GUIDE — Theory + empirical demo
# ---------------------------------------------------------------------------

def step_size_analysis():
    """
    GUIDELINE FOR CHOOSING STEP SIZE α
    ====================================

    THEORY:
    -------
    For a function with L-Lipschitz continuous gradient (|f''(x)| ≤ L everywhere):
        • Guaranteed convergence:  α < 2/L
        • Optimal fixed step:      α* = 1/L   (fastest linear convergence rate)
        • Too small (α → 0):       convergence correct but O(1/α) extra iters needed
        • Too large (α > 2/L):     gradient step overshoots; oscillation or divergence

    For f(x) = x²:  f''(x) = 2  →  L = 2  →  safe range α ∈ (0, 1)  (strictly <1)

    WHY NOT TOO SMALL:
        GD takes steps of size α·|∇f|. If α is tiny, even large gradients move x
        negligibly. To reach ε-accuracy you need O(1/αε) iterations.

    WHY NOT TOO LARGE:
        Each step x_{k+1} = x_k − α·∇f(x_k).
        For quadratic: x_{k+1} = (1 − 2α)·x_k.
        • |1−2α| < 1  ↔  α < 1  → converges (geometric shrinkage)
        • |1−2α| = 1  ↔  α = 1  → bounces forever between ±x₀
        • |1−2α| > 1  ↔  α > 1  → grows without bound (diverges)

    PRACTICAL RULE:
        1. If you know L: set α = 1/(2L) as a conservative safe choice.
        2. If L is unknown: use backtracking line search (Armijo condition).
           Start with α=1 and shrink by ρ=0.5 until f(x−α·g) ≤ f(x)−c·α·||g||².
    """
    print(__doc__.split("STEP SIZE")[0])   # print module docstring snippet

    fn = quadratic
    alphas = np.logspace(-3, 0.15, 200)
    final_errs = []
    iters_to_conv = []

    for lr in alphas:
        xs, fs, it = gradient_descent(fn, x0=3.0, lr=lr, max_iter=500)
        final_errs.append(abs(xs[-1] - fn.true_optimum_x))
        iters_to_conv.append(it)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Step-size guideline: f(x)=x², x₀=3, 500 iterations",
                 fontsize=11, fontweight="bold")

    ax1.semilogy(alphas, final_errs, color="#185FA5", lw=2)
    ax1.axvline(0.5, color="#D85A30", lw=1.5, linestyle="--",
                label="α=0.5  (α*=1/L)")
    ax1.axvline(1.0, color="#A32D2D", lw=1.5, linestyle=":",
                label="α=1.0  (divergence threshold)")
    ax1.fill_betweenx([1e-12, 1e4], 0, 0.5,  alpha=0.07, color="green",
                      label="safe zone  α < 0.5")
    ax1.fill_betweenx([1e-12, 1e4], 0.5, 1.0, alpha=0.07, color="orange",
                      label="slow zone  0.5 < α < 1")
    ax1.fill_betweenx([1e-12, 1e4], 1.0, alphas[-1], alpha=0.07, color="red",
                      label="diverge   α > 1")
    ax1.set_xlabel("Learning rate α", fontsize=10)
    ax1.set_ylabel("|x_final − x*|", fontsize=10)
    ax1.set_title("Final error vs step size", fontsize=10)
    ax1.legend(fontsize=7)
    ax1.set_ylim(1e-12, 1e4)
    ax1.grid(True, alpha=0.3)

    ax2.plot(alphas, iters_to_conv, color="#BA7517", lw=2)
    ax2.axvline(0.5, color="#D85A30", lw=1.5, linestyle="--",
                label="α*=1/L=0.5  (fewest iters)")
    ax2.set_xlabel("Learning rate α", fontsize=10)
    ax2.set_ylabel("Iterations to convergence", fontsize=10)
    ax2.set_title("Iterations needed vs step size", fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("step_size_guide.png", dpi=130, bbox_inches="tight")
    plt.show()
    print("Saved: step_size_guide.png")


# ---------------------------------------------------------------------------
# 9.  MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("="*60)
    print("Project O: Gradient-Based Function Optimization")
    print("="*60)

    print("\n[1/4] Convex quadratic — α and x₀ sweep ...")
    exp1_convex_quadratic()

    print("\n[2/4] All convex functions — fixed vs backtracking ...")
    exp2_all_convex()

    print("\n[3/4] Non-convex benchmark functions ...")
    exp3_nonconvex()

    print("\n[4/4] Step-size analysis ...")
    step_size_analysis()

    print("\nDone. All figures saved to working directory.")
    print("\n--- KEY FINDINGS SUMMARY ---")
    print("""
CONVEX FUNCTIONS
  • GD always converges to the global minimum (from any x₀) when α < 1/L.
  • For f(x)=x²: L=2, so α must be < 0.5 for guaranteed convergence.
  • Larger x₀ just means more iterations — same final answer.
  • Backtracking line search finds a good α automatically.

STEP SIZE GUIDELINE
  Too small α  → correct but very slow (O(1/α) wasted iterations).
  Too large α  → divergence (|1−2α| > 1 causes geometric blowup).
  Sweet spot   → α = 1/(2L) or use Armijo backtracking.

NON-CONVEX FUNCTIONS
  • Rastrigin:  ~10 local minima in [−5,5]; GD finds global in < 15% of starts.
  • Ackley:     near-flat outer region; gradient ≈ 0 far from origin →
                GD stagnates unless x₀ is already near the basin.
  • Griewank:   oscillating gradient sign → zigzag convergence between basins.
  • Schwefel:   deceptive landscape; global min far from all other minima.
  → GD is a LOCAL method. Without convexity guarantees, outcome depends
    entirely on the initialization. Solutions: random restarts, simulated
    annealing, evolutionary algorithms, or basin hopping.
    """)