# Gradient-Based Function Optimization

## How to Run

```bash
pip install -r requirements.txt
python gradient_descent.py
```

This generates 8 PNG figures in the current folder.

## What This Project Does

We implement **gradient descent** and test it on 4 functions:

| # | Function | Type | Formula |
|---|----------|------|---------|
| 1 | Quadratic | 1D Convex | f(x) = x² |
| 2 | Elliptic Paraboloid | 2D Convex | f(x,y) = x² + 3y² |
| 3 | Gramacy & Lee (2012) | 1D Non-Convex | f(x) = sin(10πx)/(2x) + (x−1)⁴ |
| 4 | Rastrigin | 2D Non-Convex | f(x,y) = 20 + Σ[xᵢ² − 10cos(2πxᵢ)] |

Non-convex functions are from: https://www.sfu.ca/~ssurjano/optimization.html

## How Gradient Descent Works

```
repeat:
    x_new = x_old - learning_rate * gradient(f, x_old)
```

The gradient points "uphill." By subtracting it, we move "downhill" toward lower values of f.

## Key Takeaways

### Step size (learning rate) guidelines:
- **Too small** → converges, but takes forever (wastes time)
- **Too large** → overshoots the minimum, may diverge (blow up)
- **Just right** → fast, stable convergence
- Rule of thumb: for a convex quadratic f(x) = ax², the max stable step size is 1/a (the reciprocal of the largest eigenvalue of the Hessian)

### Convex vs Non-Convex:
- **Convex** → one global minimum, GD always finds it regardless of starting point
- **Non-Convex** → multiple local minima, GD gets stuck in whichever local minimum is nearest to the starting point

## Figures Generated

| File | What it shows |
|------|---------------|
| `fig1_1d_convex.png` | 1D convex with different starts and learning rates |
| `fig2_1d_convex_convergence.png` | Convergence plot for 1D convex |
| `fig3_2d_convex.png` | 2D convex with GD paths on contour plot |
| `fig4_2d_convex_convergence.png` | Convergence plot for 2D convex |
| `fig5_1d_nonconvex.png` | 1D non-convex: GD lands at different local minima |
| `fig6_2d_nonconvex.png` | 2D Rastrigin: GD stuck in local minima |
| `fig7_rastrigin_3d.png` | 3D surface of Rastrigin (for presentation visual) |
| `fig8_step_size_demo.png` | Side-by-side: too small / good / too large step size |
