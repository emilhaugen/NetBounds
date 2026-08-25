"""
Activation derivative bounds for tanh networks.
==============================================

Implements the bound from the lemma:

    |mu^(m)(t) - mu^(m)(s)|
        <= sum_{i=1}^{n}  delta^i / i!  *  |mu^(m+i)(t)|
         + delta^{n+1} * (m+n+1)! / (n+1)!  *  2^{m+n+2}
           * exp(-2 * max(0, |t| - delta))

for all |t_i - s_i| < delta_i, m in {1, 2, 3, 4}.

t and delta are tensors of the same arbitrary shape.
delta may vary element-wise.
Default Taylor depth: n = 6 - m (remainder always bounds mu^(7)).
Computations use torch.float64 by default.
"""

import math
import torch
from typing import Optional


class TanhBounds:
    """
    Element-wise upper bound on |mu^(m)(t) - mu^(m)(s)| for all s with
    |t_i - s_i| < delta_i, via Taylor expansion of mu^(m) at t with
    Lagrange remainder bounded by the exponential envelope of tanh derivatives.

    Parameters
    ----------
    n_taylor_default : int or None
        Default number of Taylor terms n.  None -> maximum depth 6 - m.
        Must be in {1, ..., 5} or None.
    dtype : torch.dtype or None
        Floating-point dtype used for derivative and bound computations.
        Defaults to torch.float64.  Use None to preserve the input dtype.
    """

    MAX_DERIVATIVE = 6
    DEFAULT_DTYPE  = torch.float64
    # The paper originally needs m = 1, 2, 3; fourth-order PDE Taylor-cell
    # bounds also need variation bounds for m = 4.
    # only as Taylor terms in the bound formula.
    M_RANGE        = (1, 2, 3, 4)

    def __init__(
        self,
        n_taylor_default: Optional[int] = None,
        dtype: Optional[torch.dtype] = DEFAULT_DTYPE,
    ):
        if n_taylor_default is not None and not (1 <= n_taylor_default <= 5):
            raise ValueError("n_taylor_default must be in {1,...,5} or None.")
        if dtype is not None and not torch.empty((), dtype=dtype).is_floating_point():
            raise ValueError("dtype must be a floating-point torch dtype or None.")
        self.n_taylor_default = n_taylor_default
        self.dtype = dtype

    def _as_compute_dtype(self, x: torch.Tensor) -> torch.Tensor:
        if self.dtype is None or x.dtype == self.dtype:
            return x
        return x.to(dtype=self.dtype)

    # ------------------------------------------------------------------
    # tanh and its derivatives
    # ------------------------------------------------------------------

    def activation(self, x: torch.Tensor) -> torch.Tensor:
        x = self._as_compute_dtype(x)
        return torch.tanh(x)

    def activation_prime(self, x: torch.Tensor) -> torch.Tensor:
        x = self._as_compute_dtype(x)
        return 1.0 - torch.tanh(x) ** 2

    def activation_m_prime(self, x: torch.Tensor, m: int) -> torch.Tensor:
        """Analytical m-th derivative of tanh, m = 0, ..., 6."""
        x = self._as_compute_dtype(x)
        f  = self.activation(x)
        df = self.activation_prime(x)
        if m == 0: return f
        if m == 1: return df
        if m == 2: return -2 * f * df
        if m == 3: return -2 * (df**2 + f * self.activation_m_prime(x, 2))
        if m == 4: return -2 * (3 * self.activation_m_prime(x, 2) * df
                                + f * self.activation_m_prime(x, 3))
        if m == 5: return -2 * (4 * self.activation_m_prime(x, 3) * df
                                + 3 * self.activation_m_prime(x, 2) ** 2
                                + f * self.activation_m_prime(x, 4))
        if m == 6: return -2 * (10 * self.activation_m_prime(x, 2) * self.activation_m_prime(x, 3)
                                +  5 * df * self.activation_m_prime(x, 4)
                                +  f * self.activation_m_prime(x, 5))
        raise NotImplementedError(f"Derivative order {m} not implemented (max 6).")

    def activation_m_prime_stable(self, x: torch.Tensor, m: int) -> torch.Tensor:
        """Numerically stable analytical m-th derivative of tanh, m = 0, ..., 6."""
        x = self._as_compute_dtype(x)
        q = torch.exp(-2.0 * x.abs())
        den = 1.0 + q
        sign = torch.sign(x)

        if m == 0:
            return sign * (1.0 - q) / den
        if m == 1:
            return 4.0 * q / den**2
        if m == 2:
            y = -8.0 * q * (1.0 - q) / den**3
        elif m == 3:
            return 16.0 * q * (1.0 - 4.0 * q + q**2) / den**4
        elif m == 4:
            y = -32.0 * q * (1.0 - 11.0 * q + 11.0 * q**2 - q**3) / den**5
        elif m == 5:
            return 64.0 * q * (
                1.0 - 26.0 * q + 66.0 * q**2 - 26.0 * q**3 + q**4
            ) / den**6
        elif m == 6:
            y = -128.0 * q * (
                1.0 - 57.0 * q + 302.0 * q**2 - 302.0 * q**3
                + 57.0 * q**4 - q**5
            ) / den**7
        else:
            raise NotImplementedError(f"Derivative order {m} not implemented (max 6).")

        return sign * y

    # ------------------------------------------------------------------
    # Core bound
    # ------------------------------------------------------------------

    def bound(
        self,
        t:        torch.Tensor,
        delta:    torch.Tensor,
        m:        int,
        n_taylor: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        t       : Tensor of shape S — centre points
        delta   : Tensor of shape S — per-element perturbation radii (>= 0)
        m       : int in {1, 2, 3, 4}  — derivative order
        n_taylor: int or None       — Taylor terms; None -> 6 - m

        Returns
        -------
        Tensor of shape S satisfying
            result[i] >= |mu^(m)(t[i]) - mu^(m)(s[i])|
        for all s with |t[i] - s[i]| < delta[i].
        """
        if m not in self.M_RANGE:
            raise ValueError(f"m must be in {self.M_RANGE}, got {m}.")
        if delta.shape != t.shape:
            raise ValueError(f"delta shape {delta.shape} must match t shape {t.shape}.")
        if (delta < 0).any():
            raise ValueError("delta must be non-negative everywhere.")
        t = self._as_compute_dtype(t)
        delta = self._as_compute_dtype(delta)

        n = (n_taylor if n_taylor is not None
             else (self.n_taylor_default if self.n_taylor_default is not None
                   else self.MAX_DERIVATIVE - m))
        if not (1 <= n <= self.MAX_DERIVATIVE - m):
            raise ValueError(
                f"n_taylor={n} out of range for m={m}; must be in [1, {self.MAX_DERIVATIVE - m}]."
            )

        # Taylor sum: sum_{i=1}^{n}  delta^i / i!  *  |mu^(m+i)(t)|
        result  = torch.zeros_like(t)
        delta_i = delta.clone()          # delta^i, starts at delta^1
        for i in range(1, n + 1):
            result  = result + (delta_i / math.factorial(i)) * self.activation_m_prime_stable(t, m + i).abs()
            delta_i = delta_i * delta    # after loop: delta_i == delta^{n+1}

        # Lagrange remainder:
        # delta^{n+1} * (m+n+1)! / (n+1)! * 2^{m+n+2} * exp(-2*max(0,|t|-delta))
        tail_order = m + n + 1           # = 7 for all m when n = 6 - m
        rem_coeff  = (math.factorial(tail_order) / math.factorial(n + 1)
                      * 2 ** (tail_order + 1))
        exponent   = -2.0 * torch.clamp(t.abs() - delta, min=0.0)
        remainder  = delta_i * rem_coeff * torch.exp(exponent)

        return result + remainder


if __name__ == "__main__":

    bnd       = TanhBounds()
    r         = 100
    t         = torch.linspace(-r, r, 3000, dtype=bnd.dtype)
    for m in range(1, bnd.MAX_DERIVATIVE + 1):
        d1 = bnd.activation_m_prime(t, m)
        d2 = bnd.activation_m_prime_stable(t, m)
        print(m)
        print((d1 - d2).abs().max())
