"""Visualise gradient descent on a 2D loss surface.

Compares vanilla GD, momentum and Adam on the Beale function (narrow curved
valley, non-convex), showing the paths on a contour plot and loss vs. step.

Usage:
    python gradient_descent.py                 # save gradient_descent.png
    python gradient_descent.py --animate       # also save gradient_descent.gif
    python gradient_descent.py --lr 0.005 --steps 400
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter


def loss(p):
    x, y = p
    return (1.5 - x + x * y) ** 2 + (2.25 - x + x * y**2) ** 2 + (2.625 - x + x * y**3) ** 2


def grad(p):
    x, y = p
    a, b, c = 1.5 - x + x * y, 2.25 - x + x * y**2, 2.625 - x + x * y**3
    dx = 2 * a * (y - 1) + 2 * b * (y**2 - 1) + 2 * c * (y**3 - 1)
    dy = 2 * a * x + 2 * b * 2 * x * y + 2 * c * 3 * x * y**2
    return np.array([dx, dy])


def clip(g, max_norm=10.0):
    """Gradient clipping keeps the steep walls of the surface from diverging."""
    n = np.linalg.norm(g)
    return g if n <= max_norm else g * max_norm / n


def vanilla(start, lr, steps):
    p, path = np.array(start, float), [np.array(start, float)]
    for _ in range(steps):
        p = p - lr * clip(grad(p))
        path.append(p)
    return np.array(path)


def momentum(start, lr, steps, beta=0.9):
    p, v, path = np.array(start, float), np.zeros(2), [np.array(start, float)]
    for _ in range(steps):
        v = beta * v + clip(grad(p))
        p = p - lr * v
        path.append(p)
    return np.array(path)


def adam(start, lr, steps, b1=0.9, b2=0.999, eps=1e-8):
    p, m, v, path = np.array(start, float), np.zeros(2), np.zeros(2), [np.array(start, float)]
    for t in range(1, steps + 1):
        g = grad(p)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g**2
        p = p - lr * (m / (1 - b1**t)) / (np.sqrt(v / (1 - b2**t)) + eps)
        path.append(p)
    return np.array(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, default=0.005, help="learning rate for GD/momentum")
    ap.add_argument("--adam-lr", type=float, default=0.1)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--animate", action="store_true")
    args = ap.parse_args()

    start = (1.0, 1.5)
    runs = {
        "Vanilla GD": (vanilla(start, args.lr, args.steps), "#1f77b4"),
        "Momentum": (momentum(start, args.lr / 4, args.steps), "#d95f02"),
        "Adam": (adam(start, args.adam_lr, args.steps), "#1b9e77"),
    }

    xs, ys = np.linspace(-4, 4, 400), np.linspace(-3, 3.5, 400)
    X, Y = np.meshgrid(xs, ys)
    Z = loss((X, Y))
    levels = np.logspace(-1, 3.2, 25)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1.3, 1]})
    ax.contourf(X, Y, Z, levels=levels, norm=matplotlib.colors.LogNorm(), cmap="Greys", alpha=0.6)
    ax.contour(X, Y, Z, levels=levels, colors="k", linewidths=0.3)
    ax.plot(3, 0.5, "r*", ms=16, label="Global minimum (3, 0.5)")
    ax.plot(*start, "ks", label="Start")
    ax2.set_yscale("log")
    ax2.set(xlabel="Step", ylabel="Loss (log)", title="Convergence")
    ax.set(xlabel="x", ylabel="y", title="Gradient descent paths on the Beale function")

    lines, dots = {}, {}
    for name, (path, color) in runs.items():
        (lines[name],) = ax.plot([], [], "-", color=color, lw=1.8, label=name)
        (dots[name],) = ax.plot([], [], "o", color=color)
        ax2.plot([loss(p) + 1e-12 for p in path], color=color, label=name)
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend()
    ax2.grid(alpha=0.3)

    def draw(i):
        for name, (path, _) in runs.items():
            lines[name].set_data(path[: i + 1, 0], path[: i + 1, 1])
            dots[name].set_data([path[i, 0]], [path[i, 1]])
        return [*lines.values(), *dots.values()]

    draw(args.steps)
    fig.tight_layout()
    fig.savefig("gradient_descent.png", dpi=130)
    print("Saved gradient_descent.png")
    for name, (path, _) in runs.items():
        print(f"{name:11s} final loss {loss(path[-1]):.5f} at ({path[-1][0]:.3f}, {path[-1][1]:.3f})")

    if args.animate:
        anim = FuncAnimation(fig, draw, frames=range(0, args.steps + 1, 3), blit=True)
        anim.save("gradient_descent.gif", writer=PillowWriter(fps=20))
        print("Saved gradient_descent.gif")


if __name__ == "__main__":
    main()
