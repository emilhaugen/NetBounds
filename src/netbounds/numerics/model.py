"""The only neural-network architectures retained by the paper reproducer.

The network represents the free factor $F(x,t)$; the homogeneous Dirichlet
trial solution is $B(x)F(x,t)$ with the spatial product boundary
$B(x)=\\prod_{j=1}^d x_j(1-x_j)$.  The retained architectures are exactly the
six paper checkpoints:

    Heat (1,2,128), (2,3,128), (3,4,128);  Wave (1,2,256), (2,3,256), (3,3,256).

It is the reached architecture for producer revisions 2add560, 1003b544, and
53bb374; storage-state and execution-dtype provenance are checked explicitly.
The historical loader also sets the activation-bound evaluator to the same
float32 execution dtype after model conversion.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn

from ._bounds.activation_bounds import TanhBounds


class SpatialBoundaryFunction(nn.Module):
    """The spatial product factor $B(x)=\\prod_j x_j(1-x_j)$ for any $d$."""

    def __init__(self, d: int) -> None:
        super().__init__()
        self.d = d

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        spatial = inputs[..., : self.d]
        return torch.prod(spatial * (1.0 - spatial), dim=-1, keepdim=True)


class TanhNetwork(nn.Module):
    """Fully-connected tanh network for $d$ spatial variables and time.

    Class attributes ``d = 1``, ``L = 2`` and ``input_dim = 2`` keep the
    original one-dimensional defaults; the paper cases pass explicit
    ``(d, L, width)`` triples.  This deliberately exposes only the attributes
    consumed by the certified derivative recurrence.  Training, arbitrary
    activations, and non-paper architectures are outside the reproduction API.
    """

    d = 1
    input_dim = 2
    L = 2

    def __init__(self, width: int, *, d: int = 1, L: int = 2) -> None:
        super().__init__()
        self.d = int(d)
        self.input_dim = int(d) + 1
        self.width = int(width)
        self.L = int(L)
        if self.L < 2:
            raise ValueError("the paper architectures all have L >= 2 hidden layers")
        self.widths = (self.width,) * self.L
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(self.input_dim, self.width, dtype=torch.float64)]
            + [
                nn.Linear(self.width, self.width, dtype=torch.float64)
                for _ in range(1, self.L)
            ]
        )
        self.output_layer = nn.Linear(self.width, 1, dtype=torch.float64)
        self.activation = torch.tanh
        self.boundary_function = SpatialBoundaryFunction(self.d)
        self.AB = TanhBounds()
        self.max_derivative_order = self.AB.MAX_DERIVATIVE

    @property
    def input_layer(self) -> nn.Linear:
        return self.hidden_layers[0]

    @property
    def layers(self) -> tuple[nn.Linear, ...]:
        return (*self.hidden_layers, self.output_layer)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        value = inputs
        for layer in self.hidden_layers:
            value = self.activation(layer(value))
        return self.output_layer(value)

    def compute_first_order_bounds(self, y, eps, n_taylor=None):
        from ._bounds import compute_first_order_bounds

        return compute_first_order_bounds(self, y, eps, n_taylor)

    def compute_value_and_first_derivative_bounds_over_spatial_box(
        self, y, spatial_eps, n_taylor=None
    ):
        from ._bounds import compute_value_and_first_derivative_bounds_over_spatial_box

        return compute_value_and_first_derivative_bounds_over_spatial_box(
            self, y, spatial_eps, n_taylor
        )


def _expected_state_shapes(
    width: int, *, d: int = 1, L: int = 2
) -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {
        "hidden_layers.0.weight": (width, d + 1),
        "hidden_layers.0.bias": (width,),
    }
    for index in range(1, L):
        shapes[f"hidden_layers.{index}.weight"] = (width, width)
        shapes[f"hidden_layers.{index}.bias"] = (width,)
    shapes["output_layer.weight"] = (1, width)
    shapes["output_layer.bias"] = (1,)
    return shapes


def _validate_checkpoint_state(
    state: object,
    *,
    width: int,
    d: int,
    L: int,
    storage_dtype: torch.dtype,
    path: Path,
) -> Mapping[str, torch.Tensor]:
    """Reject anything other than the declared paper state-dict schema.

    The archived heat and wave-1/2 checkpoints are stored in float64; the
    nested wave-3 training checkpoint stores its model state in float32.
    Historical certificate kernels then explicitly cast the fixed
    architecture to float32, so storage dtype and execution dtype are
    intentionally different provenance fields.
    """

    if not isinstance(state, Mapping):
        raise ValueError(f"{path}: checkpoint must be a state dictionary")
    expected = _expected_state_shapes(width, d=d, L=L)
    if set(state) != set(expected):
        raise ValueError(f"{path}: checkpoint keys do not match the fixed architecture")
    for key, shape in expected.items():
        value = state[key]
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"{path}: {key} is not a tensor")
        if tuple(value.shape) != shape:
            raise ValueError(f"{path}: {key} has shape {tuple(value.shape)}, expected {shape}")
        if value.dtype != storage_dtype:
            raise ValueError(
                f"{path}: {key} has dtype {value.dtype}, expected {storage_dtype} storage"
            )
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"{path}: {key} contains non-finite values")
    return state


def load_checkpoint(
    path: str | Path,
    *,
    width: int,
    d: int = 1,
    L: int = 2,
    nested: bool = False,
    storage_dtype: torch.dtype = torch.float64,
    device: str | torch.device,
    dtype: torch.dtype = torch.float32,
) -> TanhNetwork:
    """Load a hash-verified fixed checkpoint into float32 execution state."""

    checkpoint = Path(path)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if nested:
        if not isinstance(payload, Mapping) or "model_state_dict" not in payload:
            raise ValueError(f"{checkpoint}: nested checkpoint lacks 'model_state_dict'")
        state = payload["model_state_dict"]
    else:
        state = payload
    state = _validate_checkpoint_state(
        state,
        width=width,
        d=d,
        L=L,
        storage_dtype=storage_dtype,
        path=checkpoint,
    )
    model = TanhNetwork(width, d=d, L=L)
    model.load_state_dict(state, strict=True)
    model.to(device=device, dtype=dtype)
    model.AB.dtype = dtype
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        target = torch.device(device)
        device_ok = (
            target.type == "cuda"
            and target.index is None
            and parameter.device.type == "cuda"
        ) or parameter.device == target
        if parameter.dtype != dtype or not device_ok:
            raise RuntimeError("model did not reach the requested execution dtype/device")
    return model
