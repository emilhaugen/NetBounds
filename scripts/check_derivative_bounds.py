#!/usr/bin/env python3
"""Compare sampled derivatives with the local neural-network derivative bound."""

import argparse
from itertools import combinations_with_replacement
from math import ceil
from pathlib import Path

import matplotlib
import torch
from torch import nn

from netbounds.numerics._bounds import (
    compute_first_order_bounds,
    compute_fourth_order_bounds,
    compute_second_order_bounds,
    compute_third_order_bounds,
)
from netbounds.numerics._bounds.activation_bounds import TanhBounds
from netbounds.numerics.cases import CASES
from netbounds.numerics.model import load_checkpoint


matplotlib.use("Agg")
from matplotlib import pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_CHUNK_SIZE = 64
TRAINED = {
    (case.d + 1, case.L, case.width): case
    for case in CASES.values()
}


class RandomTanhNetwork(nn.Module):
    def __init__(self, n, L, w):
        super().__init__()
        self.input_dim = n
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(n, w, dtype=torch.float64)]
            + [nn.Linear(w, w, dtype=torch.float64) for _ in range(1, L)]
        )
        self.output_layer = nn.Linear(w, 1, dtype=torch.float64)
        self.activation = torch.tanh
        self.AB = TanhBounds(dtype=torch.float64)

    @property
    def input_layer(self):
        return self.hidden_layers[0]

    def forward(self, x):
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        return self.output_layer(x)


def make_model(args):
    case = TRAINED.get((args.n, args.L, args.w))
    if case is None:
        return RandomTanhNetwork(args.n, args.L, args.w), "random initialization"

    model = load_checkpoint(
        ROOT / case.checkpoint,
        width=case.width,
        d=case.d,
        L=case.L,
        nested=case.nested,
        storage_dtype=getattr(torch, case.storage_dtype),
        device="cpu",
        dtype=torch.float64,
    )
    return model, f"trained {case.equation.lower()}{case.d} ({Path(case.checkpoint).name})"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check all symmetric derivative bounds of a chosen order."
    )
    parser.add_argument("--n", type=int, default=2)
    parser.add_argument("--order", type=int, default=1)
    parser.add_argument("--L", type=int, default=2)
    parser.add_argument("--w", type=int, default=128)
    parser.add_argument("--N", type=int, default=100)
    parser.add_argument("--Nsamples", type=int, default=10)
    parser.add_argument("--eps", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()
    args.plot = (
        args.plot
        or ROOT / "scripts" / "plots" / f"derivative_bounds_order_{args.order}.png"
    )
    return args


def sample_boxes(args):
    generator = torch.Generator().manual_seed(args.seed + 1)
    centers = args.eps + (1 - 2 * args.eps) * torch.rand(
        args.N, args.n, dtype=torch.float64, generator=generator
    )
    offsets = args.eps * (
        2
        * torch.rand(
            args.N,
            args.Nsamples,
            args.n,
            dtype=torch.float64,
            generator=generator,
        )
        - 1
    )
    return centers, centers[:, None, :] + offsets


def compute_bounds(model, centers, eps, order):
    multi_indices = tuple(combinations_with_replacement(range(model.input_dim), order))
    if order == 1:
        result = compute_first_order_bounds(model, centers, eps)
        bounds = result.base_gradient.abs() + result.derivative_bounds
    if order == 2:
        result = compute_second_order_bounds(
            model, centers, eps, multi_indices="full_input"
        )
        bounds = result.base_hessian.abs() + result.second_derivative_bounds
    if order == 3:
        result = compute_third_order_bounds(
            model, centers, eps, multi_indices="full_input"
        )
        bounds = result.base_third_derivatives.abs() + result.third_derivative_bounds
    if order == 4:
        result = compute_fourth_order_bounds(
            model, centers, eps, multi_indices="full_input"
        )
        bounds = result.base_fourth_derivatives.abs() + result.fourth_derivative_bounds
    return multi_indices, bounds


def sample_derivatives(model, points, order, multi_indices):
    def f(x):
        return model(x[None]).squeeze()

    derivative = f
    for _ in range(order):
        derivative = torch.func.jacrev(derivative)

    flat_points = points.reshape(-1, model.input_dim)
    values = []
    for chunk in flat_points.split(SAMPLE_CHUNK_SIZE):
        full_derivative = torch.vmap(derivative)(chunk)
        components = [
            full_derivative[(slice(None), *coordinates)]
            for coordinates in multi_indices
        ]
        values.append(torch.stack(components, dim=-1).detach().abs())
    return torch.cat(values).reshape(*points.shape[:2], len(multi_indices))


def alpha_text(coordinates, n, latex=False):
    terms = []
    for coordinate in range(n):
        multiplicity = coordinates.count(coordinate)
        if multiplicity:
            subscript = f"_{{{coordinate + 1}}}" if latex else f"_{coordinate + 1}"
            coefficient = "" if multiplicity == 1 else str(multiplicity)
            terms.append(f"{coefficient}e{subscript}")
    return "+".join(terms)


def make_plot(bounds, samples, multi_indices, model_source, args):
    box_numbers = range(1, args.N + 1)
    sample_maxima = samples.max(dim=1).values
    columns = min(2, len(multi_indices))
    rows = ceil(len(multi_indices) / columns)
    figure, axes = plt.subplots(
        rows, columns, figsize=(8 * columns, 4 * rows), squeeze=False
    )
    for column, (coordinates, axis) in enumerate(zip(multi_indices, axes.flat)):
        axis.bar(
            box_numbers,
            bounds[:, column].numpy(),
            width=0.9,
            color="#f4a261",
            edgecolor="#d97706",
            label="Derivative bound",
        )
        axis.bar(
            box_numbers,
            sample_maxima[:, column].numpy(),
            width=0.55,
            color="#245a88",
            label="Maximum sampled value",
        )
        axis.set_title(
            rf"$\alpha={alpha_text(coordinates, args.n, latex=True)}$"
        )
        axis.set_xlabel("Box index")
        axis.set_ylabel(r"$|\partial^\alpha f|$")
        axis.legend()
        axis.grid(axis="y", alpha=0.25)
    for axis in axes.flat[len(multi_indices):]:
        axis.remove()
    figure.text(
        0.5,
        0.015,
        (
            rf"{model_source.capitalize()}: order {args.order}, $n={args.n}$, $L={args.L}$, "
            rf"$w={args.w}$, $\varepsilon={args.eps:g}$; "
            rf"$N={args.N}$ boxes and $N_s={args.Nsamples}$ samples per box."
        ),
        ha="center",
    )
    figure.tight_layout(rect=(0, 0.07, 1, 1))
    args.plot.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.plot, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    model, model_source = make_model(args)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    centers, points = sample_boxes(args)
    multi_indices, bounds = compute_bounds(model, centers, args.eps, args.order)
    samples = sample_derivatives(model, points, args.order, multi_indices)
    make_plot(bounds, samples, multi_indices, model_source, args)

    violations = samples > bounds[:, None]
    ratio = samples / bounds[:, None]
    margin = bounds[:, None] - samples

    print(
        f"model={model_source}; order={args.order}, "
        f"n={args.n}, L={args.L}, w={args.w}, N={args.N}, "
        f"N_s={args.Nsamples}, eps={args.eps:g}, seed={args.seed}"
    )
    print(
        "alpha          max |d^alpha f(x)|    worst sample/bound    "
        "min bound-sample   violations  result"
    )
    for column, coordinates in enumerate(multi_indices):
        count = int(violations[:, :, column].sum())
        print(
            f"{alpha_text(coordinates, args.n):<14} "
            f"{samples[:, :, column].max():20.6e} "
            f"{ratio[:, :, column].max():21.6e} "
            f"{margin[:, :, column].min():19.6e} "
            f"{count:11d}  {'PASS' if count == 0 else 'FAIL'}"
        )
    print(f"plot: {args.plot.resolve()}")


if __name__ == "__main__":
    main()
