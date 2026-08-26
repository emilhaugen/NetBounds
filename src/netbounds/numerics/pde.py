"""PDE Q1 moment kernel used to compute the current tables."""

from __future__ import annotations

import torch

from ._bounds import compute_fourth_order_bounds, compute_second_order_bounds, compute_third_order_bounds

from ._initial_support.quadrature import as_cell_vector, require_nonnegative

from ._pde_helpers import (
    _add_counts,
    _boundary_center_and_variations,
    _boundary_derivative_center,
    _boundary_derivative_sup,
    _column,
    _interval_variation,
    _product_interval,
    _product_variation,
)

def _spatial_eps_vector(
    y: torch.Tensor,
    eps_x: float | torch.Tensor,
    d: int,
) -> torch.Tensor:
    """Normalize scalar/isotropic or coordinatewise spatial cell radii."""

    eps = torch.as_tensor(eps_x, dtype=y.dtype, device=y.device)
    if eps.ndim == 0:
        eps = eps.repeat(d)
    if eps.shape != (d,):
        raise ValueError("spatial radius must be scalar or have shape (d,)")
    return eps


def _boundary_grouped_coeffs(
    y: torch.Tensor,
    eps_x: float | torch.Tensor,
    d: int,
) -> dict[str, object]:
    """Analytic coefficient centers/variations with grouped Delta B.

    This helper uses the grouped coefficient form
    ``B*(F_tt-c^2 Delta F) - 2c^2 sum_i B_i F_i - c^2 (Delta B) F`` so the
    shared coefficient ``B`` is only varied once for the time-minus-Laplacian
    network factor.
    """
    eps_spatial = _spatial_eps_vector(y, eps_x, d)
    terms = _boundary_center_and_variations(y, eps_spatial, d)
    delta_b = torch.zeros_like(terms["B"])
    eta_delta_b = torch.zeros_like(terms["B"])
    for b_ii, eta_b_ii in zip(terms["B_ii"], terms["eta_B_ii"]):
        delta_b = delta_b + b_ii
        eta_delta_b = eta_delta_b + eta_b_ii
    terms["DeltaB"] = delta_b
    terms["eta_DeltaB"] = require_nonnegative("eta_DeltaB", eta_delta_b)
    return terms

def _wave_operator_third_center_and_error(third, *, input_dim: int, d: int, t: int, c2: float, q: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Center/error bound for ∂_q(F_tt - c^2 ΔF), preserving center cancellation."""
    center = _column(third.base_third_derivatives, third.triples, (t, t, q))
    error = _column(third.third_derivative_bounds, third.triples, (t, t, q))
    for i in range(d):
        center = center - c2 * _column(third.base_third_derivatives, third.triples, (i, i, q))
        error = error + c2 * _column(third.third_derivative_bounds, third.triples, (i, i, q))
    return center, require_nonnegative("wave_operator_third_error", error)

def _boundary_p_first_coeffs(
    y: torch.Tensor,
    eps_x: float | torch.Tensor,
    d: int,
) -> dict[str, object]:
    """Centers and valid center-variation radii for ``P_i`` and ``P_{i,q}``."""

    x = y[:, :d]
    eps_spatial = _spatial_eps_vector(y, eps_x, d)
    lower = (x - eps_spatial).clamp_min(0.0)
    upper = (x + eps_spatial).clamp_max(1.0)
    s = x * (1.0 - x)
    d1 = 1.0 - 2.0 * x
    s_lower = torch.minimum(lower * (1.0 - lower), upper * (1.0 - upper))
    s_upper = torch.where(
        (lower <= 0.5) & (0.5 <= upper),
        torch.full_like(lower, 0.25),
        torch.maximum(lower * (1.0 - lower), upper * (1.0 - upper)),
    )
    d_lower = 1.0 - 2.0 * upper
    d_upper = 1.0 - 2.0 * lower
    P: list[torch.Tensor] = []
    eta_P: list[torch.Tensor] = []
    P_q: list[list[torch.Tensor]] = []
    eta_P_q: list[list[torch.Tensor]] = []
    one = torch.ones(x.shape[0], dtype=x.dtype, device=x.device)
    zero = torch.zeros_like(one)
    for i in range(d):
        others = [j for j in range(d) if j != i]
        center = s[:, others].prod(dim=-1) if others else one.clone()
        interval_lower = s_lower[:, others].prod(dim=-1) if others else one.clone()
        interval_upper = s_upper[:, others].prod(dim=-1) if others else one.clone()
        P.append(center)
        eta_P.append(_interval_variation(center, interval_lower, interval_upper))
        row = []
        eta_row = []
        for q in range(d + 1):
            if q >= d or q == i:
                row.append(zero)
                eta_row.append(zero)
                continue
            other_factors = [j for j in range(d) if j != i and j != q]
            center_q = d1[:, q]
            for j in other_factors:
                center_q = center_q * s[:, j]
            factor_lower = torch.cat(
                [d_lower[:, q : q + 1], s_lower[:, other_factors]], dim=1
            )
            factor_upper = torch.cat(
                [d_upper[:, q : q + 1], s_upper[:, other_factors]], dim=1
            )
            interval_q_lower, interval_q_upper = _product_interval(
                factor_lower, factor_upper
            )
            row.append(center_q)
            eta_row.append(
                _interval_variation(center_q, interval_q_lower, interval_q_upper)
            )
        P_q.append(row)
        eta_P_q.append(eta_row)
    return {"P": P, "eta_P": eta_P, "P_q": P_q, "eta_P_q": eta_P_q}

def _quad_col(vals: torch.Tensor, quartets: tuple[tuple[int, int, int, int], ...], idx: tuple[int, int, int, int]) -> torch.Tensor:
    return vals[:, quartets.index(tuple(sorted(idx)))]

def _quadratic_envelope_l2sq(
    hessian_sup: torch.Tensor,
    eps_vec: torch.Tensor,
) -> torch.Tensor:
    """Exact cell integral of a nonnegative quadratic Taylor envelope squared.

    For ``e(z)=1/2 sum_qr h_qr |z_q z_r|`` on the centered rectangle,
    return ``integral e(z)^2 dz``.  The entries of ``hessian_sup`` must be
    nonnegative pointwise Hessian suprema with shape ``(cells,n,n)``.
    """
    if hessian_sup.ndim != 3 or hessian_sup.shape[1] != hessian_sup.shape[2]:
        raise ValueError("hessian_sup must have shape (cells,n,n)")
    n = hessian_sup.shape[1]
    if eps_vec.shape != (n,):
        raise ValueError("eps_vec shape does not match hessian_sup")
    pairs = [(q, r) for q in range(n) for r in range(n)]
    moment = torch.empty((n * n, n * n), dtype=hessian_sup.dtype, device=hessian_sup.device)
    for a, (q, r) in enumerate(pairs):
        for b, (s, ell) in enumerate(pairs):
            powers = [0] * n
            for axis in (q, r, s, ell):
                powers[axis] += 1
            value = torch.ones((), dtype=hessian_sup.dtype, device=hessian_sup.device)
            for axis, power in enumerate(powers):
                if power:
                    value = value * eps_vec[axis].pow(power) / float(power + 1)
            moment[a, b] = value
    coeff = 0.5 * hessian_sup.reshape(hessian_sup.shape[0], n * n)
    mean_square = torch.einsum("mi,ij,mj->m", coeff, moment, coeff)
    cell_volume = torch.prod(2.0 * eps_vec)
    return require_nonnegative("quadratic_envelope_l2sq", cell_volume * mean_square)

def _affine_quadratic_envelope_cross_integral(
    residual_abs: torch.Tensor,
    gradient_abs: torch.Tensor,
    hessian_sup: torch.Tensor,
    eps_vec: torch.Tensor,
) -> torch.Tensor:
    """Integrate ``(|r0|+sum_i |g_i||z_i|) * e(z)`` exactly.

    Here ``e(z)=1/2 sum_qr h_qr |z_q z_r|`` bounds the Taylor remainder.
    This upper-bounds the absolute affine/remainder cross term without the
    additional Cauchy--Schwarz relaxation used by Minkowski.
    """
    cells, n, n2 = hessian_sup.shape
    if n != n2 or gradient_abs.shape != (cells, n) or residual_abs.shape != (cells,):
        raise ValueError("incompatible affine/quadratic envelope shapes")
    out_mean = torch.zeros(cells, dtype=hessian_sup.dtype, device=hessian_sup.device)
    for q in range(n):
        for r in range(n):
            coeff = 0.5 * hessian_sup[:, q, r]
            powers = [0] * n
            powers[q] += 1
            powers[r] += 1
            base_moment = torch.ones((), dtype=hessian_sup.dtype, device=hessian_sup.device)
            for axis, power in enumerate(powers):
                if power:
                    base_moment = base_moment * eps_vec[axis].pow(power) / float(power + 1)
            factor = residual_abs * base_moment
            for affine_axis in range(n):
                augmented = list(powers)
                augmented[affine_axis] += 1
                moment = torch.ones((), dtype=hessian_sup.dtype, device=hessian_sup.device)
                for axis, power in enumerate(augmented):
                    if power:
                        moment = moment * eps_vec[axis].pow(power) / float(power + 1)
                factor = factor + gradient_abs[:, affine_axis] * moment
            out_mean = out_mean + coeff * factor
    return require_nonnegative(
        "affine_quadratic_envelope_cross_integral",
        torch.prod(2.0 * eps_vec) * out_mean,
    )

def wave_hyper_taylor_rect_batch(
    *,
    model,
    y: torch.Tensor,
    eps_vec: torch.Tensor,
    c2: float,
    n_taylor: int | None,
    equation: str = "wave",
):
    """Operator-aware affine Q1 cell with a split Hessian remainder.

    ``equation='wave'`` preserves ``F_tt-c2*Delta F``.  The parabolic
    ``equation='heat'`` variant preserves ``F_t-c2*Delta F``; in that case
    ``c2`` is the thermal diffusivity.  The returned rigorous remainder is
    always the sum of independent ``BA`` and boundary-correction suprema.
    """
    if equation not in ("heat", "wave"):
        raise ValueError("equation must be 'heat' or 'wave'")
    m = y.shape[0]
    dtype, device = y.dtype, y.device
    d = model.d
    input_dim = model.input_dim
    t = input_dim - 1
    first = model.compute_first_order_bounds(y, eps_vec, n_taylor)
    second = compute_second_order_bounds(model, y, eps_vec, n_taylor, multi_indices="full_input")
    third = compute_third_order_bounds(model, y, eps_vec, n_taylor, multi_indices="full_input")
    # Only quartets needed by the spatial Laplacian contribution to A_qr,
    # plus the F_ttqr term for Wave.  Heat's F_tqr contribution is third order.
    quartets = []
    for q in range(input_dim):
        for r0 in range(input_dim):
            if equation == "wave":
                quartets.append(tuple(sorted((t, t, q, r0))))
            for i in range(d):
                quartets.append(tuple(sorted((i, i, q, r0))))
    fourth = compute_fourth_order_bounds(model, y, eps_vec, n_taylor, multi_indices=quartets)

    f = as_cell_vector("F", model(y), length=m, dtype=dtype, device=device)
    grad = first.base_gradient
    first_err = first.derivative_bounds
    second_err = second.second_derivative_bounds
    second_sup = second.base_hessian.abs() + second_err
    third_sup = third.base_third_derivatives.abs() + third.third_derivative_bounds

    delta_f = torch.zeros(m, dtype=dtype, device=device)
    for i in range(d):
        delta_f = delta_f + _column(second.base_hessian, second.pairs, (i, i))
    if equation == "wave":
        a = _column(second.base_hessian, second.pairs, (t, t)) - c2 * delta_f
    else:
        a = grad[:, t] - c2 * delta_f

    A_q_center: list[torch.Tensor] = []
    A_q_error: list[torch.Tensor] = []
    for q in range(input_dim):
        if equation == "wave":
            aq_center, aq_error = _wave_operator_third_center_and_error(
                third, input_dim=input_dim, d=d, t=t, c2=c2, q=q
            )
        else:
            aq_center = _column(second.base_hessian, second.pairs, (t, q))
            aq_error = _column(
                second.second_derivative_bounds, second.pairs, (t, q)
            )
            for i in range(d):
                aq_center = aq_center - c2 * _column(
                    third.base_third_derivatives, third.triples, (i, i, q)
                )
                aq_error = aq_error + c2 * _column(
                    third.third_derivative_bounds, third.triples, (i, i, q)
                )
            aq_error = require_nonnegative("heat_operator_gradient_error", aq_error)
        A_q_center.append(aq_center)
        A_q_error.append(aq_error)

    eps_x = eps_vec[:d]
    coeffs = _boundary_grouped_coeffs(y, eps_x, d)
    B = coeffs["B"]
    eta_B = coeffs["eta_B"]
    residual = B * a

    p = _boundary_p_first_coeffs(y, eps_x, d)
    x = y[:, :d]
    rcoef = 1.0 - 2.0 * x
    eta_r = 2.0 * eps_x
    H: list[torch.Tensor] = []
    eta_H: list[torch.Tensor] = []
    H_q: list[list[torch.Tensor]] = []
    eta_H_q: list[list[torch.Tensor]] = []
    for i in range(d):
        h_i = rcoef[:, i] * grad[:, i] - f
        H.append(h_i)
        eta_h_i = torch.zeros(m, dtype=dtype, device=device)
        hq_row = []
        etahq_row = []
        for q in range(input_dim):
            f_iq = _column(second.base_hessian, second.pairs, (i, q))
            h_iq = rcoef[:, i] * f_iq - grad[:, q]
            if q == i:
                h_iq = h_iq - 2.0 * grad[:, i]
            hq_row.append(h_iq)
            err_hiq = torch.zeros(m, dtype=dtype, device=device)
            for sidx in range(input_dim):
                f_iqs = _column(third.base_third_derivatives, third.triples, (i, q, sidx))
                center = rcoef[:, i] * f_iqs - _column(second.base_hessian, second.pairs, (q, sidx))
                if sidx == i:
                    center = center - 2.0 * f_iq
                if q == i:
                    center = center - 2.0 * _column(second.base_hessian, second.pairs, (i, sidx))
                err = rcoef[:, i].abs() * _column(third.third_derivative_bounds, third.triples, (i, q, sidx))
                err = err + eta_r[i] * _column(third_sup, third.triples, (i, q, sidx))
                err = err + _column(second_err, second.pairs, (q, sidx))
                if sidx == i:
                    err = err + 2.0 * _column(second_err, second.pairs, (i, q))
                if q == i:
                    err = err + 2.0 * _column(second_err, second.pairs, (i, sidx))
                err_hiq = err_hiq + eps_vec[sidx] * (center.abs() + err)
            etahq_row.append(require_nonnegative(f"eta_H_hyptay_{i}_{q}", err_hiq))
            err_hi = rcoef[:, i].abs() * _column(second_err, second.pairs, (i, q))
            err_hi = err_hi + eta_r[i] * _column(second_sup, second.pairs, (i, q)) + first_err[:, q]
            if q == i:
                err_hi = err_hi + 2.0 * first_err[:, i]
            eta_h_i = eta_h_i + eps_vec[q] * (h_iq.abs() + err_hi)
        H_q.append(hq_row)
        eta_H_q.append(etahq_row)
        eta_H.append(require_nonnegative(f"eta_H_hyptay_sum_{i}", eta_h_i))
        residual = residual - 2.0 * c2 * p["P"][i] * h_i

    affine_grad_term = torch.zeros(m, dtype=dtype, device=device)
    affine_gradient: list[torch.Tensor] = []
    boundary_error_sum = torch.zeros(m, dtype=dtype, device=device)
    for q in range(input_dim):
        if q < d:
            B_q = coeffs["B_i"][q]
        else:
            B_q = torch.zeros(m, dtype=dtype, device=device)
        hyper_center_q = B_q * a + B * A_q_center[q]
        boundary_center_q = torch.zeros(m, dtype=dtype, device=device)
        boundary_err_q = torch.zeros(m, dtype=dtype, device=device)
        for i in range(d):
            boundary_center_q = boundary_center_q + p["P_q"][i][q] * H[i] + p["P"][i] * H_q[i][q]
            boundary_err_q = boundary_err_q + _product_variation(p["P_q"][i][q], p["eta_P_q"][i][q], H[i], eta_H[i])
            boundary_err_q = boundary_err_q + _product_variation(p["P"][i], p["eta_P"][i], H_q[i][q], eta_H_q[i][q])
        full_center_q = hyper_center_q - 2.0 * c2 * boundary_center_q
        affine_gradient.append(full_center_q)
        affine_grad_term = affine_grad_term + (eps_vec[q].square() / 3.0) * full_center_q.square()
        boundary_error_sum = boundary_error_sum + eps_vec[q] * 2.0 * c2 * boundary_err_q

    # Second-order Taylor remainder for U=BA.
    B_sup = B.abs() + eta_B
    A_sup = a.abs() + sum(eps_vec[q] * (A_q_center[q].abs() + A_q_error[q]) for q in range(input_dim))
    A_q_sup = [A_q_center[q].abs() + A_q_error[q] for q in range(input_dim)]
    eta_A = require_nonnegative("eta_wave_operator_A", A_sup - a.abs())
    rho_hyper = torch.zeros(m, dtype=dtype, device=device)
    rho_hyper_centered_variation = torch.zeros(m, dtype=dtype, device=device)
    aq_r_center_abs_sum = torch.zeros(m, dtype=dtype, device=device)
    aq_r_error_sum = torch.zeros(m, dtype=dtype, device=device)
    U_qr_centers: list[list[torch.Tensor]] = []
    U_qr_errors: list[list[torch.Tensor]] = []
    U_qr_sups: list[list[torch.Tensor]] = []
    U_qr_centered_sups: list[list[torch.Tensor]] = []
    full_hessian_rho = torch.zeros(m, dtype=dtype, device=device)
    full_hessian_center_part = torch.zeros(m, dtype=dtype, device=device)
    full_hessian_error_part = torch.zeros(m, dtype=dtype, device=device)
    for q in range(input_dim):
        U_center_row: list[torch.Tensor] = []
        U_error_row: list[torch.Tensor] = []
        U_sup_row: list[torch.Tensor] = []
        U_centered_sup_row: list[torch.Tensor] = []
        for rj in range(input_dim):
            if equation == "wave":
                A_qr_center = _quad_col(
                    fourth.base_fourth_derivatives,
                    fourth.quartets,
                    (t, t, q, rj),
                )
                A_qr_error = _quad_col(
                    fourth.fourth_derivative_bounds,
                    fourth.quartets,
                    (t, t, q, rj),
                )
            else:
                A_qr_center = _column(
                    third.base_third_derivatives, third.triples, (t, q, rj)
                )
                A_qr_error = _column(
                    third.third_derivative_bounds, third.triples, (t, q, rj)
                )
            for i in range(d):
                A_qr_center = A_qr_center - c2 * _quad_col(fourth.base_fourth_derivatives, fourth.quartets, (i, i, q, rj))
                A_qr_error = A_qr_error + c2 * _quad_col(fourth.fourth_derivative_bounds, fourth.quartets, (i, i, q, rj))
            A_qr_sup = A_qr_center.abs() + A_qr_error
            aq_r_center_abs_sum = aq_r_center_abs_sum + 0.5 * eps_vec[q] * eps_vec[rj] * A_qr_center.abs()
            aq_r_error_sum = aq_r_error_sum + 0.5 * eps_vec[q] * eps_vec[rj] * A_qr_error
            if q < d and rj < d:
                counts_qr = _add_counts((0,) * d, (q, rj))
                lower_x = (y[:, :d] - eps_x).clamp_min(0.0)
                upper_x = (y[:, :d] + eps_x).clamp_max(1.0)
                B_qr_sup = _boundary_derivative_sup(lower_x, upper_x, counts_qr)
                B_qr_center = _boundary_derivative_center(y[:, :d], counts_qr)
                eta_B_qr = torch.zeros(m, dtype=dtype, device=device)
                for spatial_axis in range(d):
                    counts_qrs = _add_counts(counts_qr, (spatial_axis,))
                    eta_B_qr = eta_B_qr + eps_vec[spatial_axis] * _boundary_derivative_sup(
                        lower_x, upper_x, counts_qrs
                    )
                eta_B_qr = require_nonnegative("eta_B_qr", eta_B_qr)
            else:
                B_qr_sup = torch.zeros(m, dtype=dtype, device=device)
                B_qr_center = torch.zeros(m, dtype=dtype, device=device)
                eta_B_qr = torch.zeros(m, dtype=dtype, device=device)
            if q < d:
                B_q_center = coeffs["B_i"][q]
                eta_B_q = coeffs["eta_B_i"][q]
                B_q_sup = B_q_center.abs() + eta_B_q
            else:
                B_q_center = torch.zeros(m, dtype=dtype, device=device)
                eta_B_q = torch.zeros(m, dtype=dtype, device=device)
                B_q_sup = torch.zeros(m, dtype=dtype, device=device)
            if rj < d:
                B_r_center = coeffs["B_i"][rj]
                eta_B_r = coeffs["eta_B_i"][rj]
                B_r_sup = B_r_center.abs() + eta_B_r
            else:
                B_r_center = torch.zeros(m, dtype=dtype, device=device)
                eta_B_r = torch.zeros(m, dtype=dtype, device=device)
                B_r_sup = torch.zeros(m, dtype=dtype, device=device)
            U_qr_center = B_qr_center * a + B_q_center * A_q_center[rj] + B_r_center * A_q_center[q] + B * A_qr_center
            U_qr_sup = B_qr_sup * A_sup + B_q_sup * A_q_sup[rj] + B_r_sup * A_q_sup[q] + B_sup * A_qr_sup
            U_qr_variation = _product_variation(B_qr_center, eta_B_qr, a, eta_A)
            U_qr_variation = U_qr_variation + _product_variation(
                B_q_center, eta_B_q, A_q_center[rj], A_q_error[rj]
            )
            U_qr_variation = U_qr_variation + _product_variation(
                B_r_center, eta_B_r, A_q_center[q], A_q_error[q]
            )
            U_qr_variation = U_qr_variation + _product_variation(
                B, eta_B, A_qr_center, A_qr_error
            )
            U_qr_variation = require_nonnegative("U_qr_centered_variation", U_qr_variation)
            U_qr_error = (U_qr_sup - U_qr_center.abs()).clamp_min(0.0)
            U_center_row.append(U_qr_center)
            U_error_row.append(U_qr_error)
            U_sup_row.append(U_qr_sup)
            U_centered_sup_row.append(U_qr_center.abs() + U_qr_variation)
            weight_qr = 0.5 * eps_vec[q] * eps_vec[rj]
            rho_hyper = rho_hyper + weight_qr * U_qr_sup
            rho_hyper_centered_variation = rho_hyper_centered_variation + weight_qr * (
                U_qr_center.abs() + U_qr_variation
            )
        U_qr_centers.append(U_center_row)
        U_qr_errors.append(U_error_row)
        U_qr_sups.append(U_sup_row)
        U_qr_centered_sups.append(U_centered_sup_row)

    # Second-order Taylor remainder for the boundary correction V=-2*c2*S,
    # S=sum_i P_i H_i. This is exactly the residual part other than BA.
    s = x * (1.0 - x)
    d1 = 1.0 - 2.0 * x
    lower_x = (x - eps_x).clamp_min(0.0)
    upper_x = (x + eps_x).clamp_max(1.0)
    s_sup = torch.where((lower_x <= 0.5) & (0.5 <= upper_x), torch.full_like(lower_x, 0.25), torch.maximum((lower_x * (1.0 - lower_x)).abs(), (upper_x * (1.0 - upper_x)).abs()))
    d_sup = torch.maximum((1.0 - 2.0 * lower_x).abs(), (1.0 - 2.0 * upper_x).abs())
    H_sup = [H[i].abs() + eta_H[i] for i in range(d)]
    H_q_sup = [[H_q[i][q].abs() + eta_H_q[i][q] for q in range(input_dim)] for i in range(d)]
    rho_boundary_second = torch.zeros(m, dtype=dtype, device=device)
    boundary_hessian_center_part = torch.zeros(m, dtype=dtype, device=device)
    S_qr_sups: list[list[torch.Tensor]] = []
    for q in range(input_dim):
        S_sup_row: list[torch.Tensor] = []
        for rj in range(input_dim):
            S_qr_sup = torch.zeros(m, dtype=dtype, device=device)
            S_qr_center = torch.zeros(m, dtype=dtype, device=device)
            for i in range(d):
                if q >= d or rj >= d or q == i or rj == i:
                    P_qr_center = torch.zeros(m, dtype=dtype, device=device)
                    P_qr_sup = torch.zeros(m, dtype=dtype, device=device)
                elif q == rj:
                    P_qr_center = torch.full((m,), -2.0, dtype=dtype, device=device)
                    P_qr_sup = torch.full((m,), 2.0, dtype=dtype, device=device)
                    for j in range(d):
                        if j == i or j == q:
                            continue
                        P_qr_center = P_qr_center * s[:, j]
                        P_qr_sup = P_qr_sup * s_sup[:, j]
                else:
                    P_qr_center = d1[:, q] * d1[:, rj]
                    P_qr_sup = d_sup[:, q] * d_sup[:, rj]
                    for j in range(d):
                        if j == i or j == q or j == rj:
                            continue
                        P_qr_center = P_qr_center * s[:, j]
                        P_qr_sup = P_qr_sup * s_sup[:, j]

                F_iqr_center = _column(third.base_third_derivatives, third.triples, (i, q, rj))
                F_iqr_sup = _column(third_sup, third.triples, (i, q, rj))
                H_qr_center = rcoef[:, i] * F_iqr_center - _column(second.base_hessian, second.pairs, (q, rj))
                H_qr_sup = (rcoef[:, i].abs() + eta_r[i]) * F_iqr_sup + _column(second_sup, second.pairs, (q, rj))
                if q == i:
                    H_qr_center = H_qr_center - 2.0 * _column(second.base_hessian, second.pairs, (i, rj))
                    H_qr_sup = H_qr_sup + 2.0 * _column(second_sup, second.pairs, (i, rj))
                if rj == i:
                    H_qr_center = H_qr_center - 2.0 * _column(second.base_hessian, second.pairs, (i, q))
                    H_qr_sup = H_qr_sup + 2.0 * _column(second_sup, second.pairs, (i, q))

                term_center = P_qr_center * H[i] + p["P_q"][i][q] * H_q[i][rj] + p["P_q"][i][rj] * H_q[i][q] + p["P"][i] * H_qr_center
                S_qr_center = S_qr_center + term_center
                S_qr_sup = S_qr_sup + P_qr_sup * H_sup[i]
                S_qr_sup = S_qr_sup + (p["P_q"][i][q].abs() + p["eta_P_q"][i][q]) * H_q_sup[i][rj]
                S_qr_sup = S_qr_sup + (p["P_q"][i][rj].abs() + p["eta_P_q"][i][rj]) * H_q_sup[i][q]
                S_qr_sup = S_qr_sup + (p["P"][i].abs() + p["eta_P"][i]) * H_qr_sup
            S_qr_error = (S_qr_sup - S_qr_center.abs()).clamp_min(0.0)
            S_sup_row.append(S_qr_sup)
            V_qr_center = -2.0 * c2 * S_qr_center
            V_qr_error = 2.0 * c2 * S_qr_error
            full_center_qr = U_qr_centers[q][rj] + V_qr_center
            full_error_qr = U_qr_errors[q][rj] + V_qr_error
            weight_qr = 0.5 * eps_vec[q] * eps_vec[rj]
            boundary_hessian_center_part = boundary_hessian_center_part + weight_qr * 2.0 * c2 * S_qr_center.abs()
            rho_boundary_second = rho_boundary_second + weight_qr * (2.0 * c2 * (S_qr_center.abs() + S_qr_error))
            full_hessian_center_part = full_hessian_center_part + weight_qr * full_center_qr.abs()
            full_hessian_error_part = full_hessian_error_part + weight_qr * full_error_qr
            full_hessian_rho = full_hessian_rho + weight_qr * (full_center_qr.abs() + full_error_qr)
        S_qr_sups.append(S_sup_row)

    rho = rho_hyper + rho_boundary_second  # PROVABLE: sum of independent supremum bounds on U and V parts
    rho_centered_hyper = rho_hyper_centered_variation + rho_boundary_second
    cell_volume = torch.prod(2.0 * eps_vec)
    hessian_sup = torch.stack(
        [torch.stack(row, dim=1) for row in U_qr_sups], dim=1
    ) + 2.0 * c2 * torch.stack(
        [torch.stack(row, dim=1) for row in S_qr_sups], dim=1
    )
    moment_remainder_l2sq = _quadratic_envelope_l2sq(hessian_sup, eps_vec)
    centered_hessian_sup = torch.stack(
        [torch.stack(row, dim=1) for row in U_qr_centered_sups], dim=1
    ) + 2.0 * c2 * torch.stack(
        [torch.stack(row, dim=1) for row in S_qr_sups], dim=1
    )
    centered_moment_remainder_l2sq = _quadratic_envelope_l2sq(
        centered_hessian_sup, eps_vec
    )
    affine_gradient_tensor = torch.stack(affine_gradient, dim=1)
    moment_cross_integral = _affine_quadratic_envelope_cross_integral(
        residual.abs(), affine_gradient_tensor.abs(), hessian_sup, eps_vec
    )
    centered_moment_cross_integral = _affine_quadratic_envelope_cross_integral(
        residual.abs(), affine_gradient_tensor.abs(), centered_hessian_sup, eps_vec
    )
    affine_l2sq = cell_volume * (residual.square() + affine_grad_term)
    affine_norm = torch.sqrt(require_nonnegative("hyper_taylor_affine_l2sq", affine_l2sq))
    bound_l2sq = (affine_norm + torch.sqrt(cell_volume) * rho).square()
    centered_hyper_bound_l2sq = (
        affine_norm + torch.sqrt(cell_volume) * rho_centered_hyper
    ).square()
    moment_bound_l2sq = (affine_norm + torch.sqrt(moment_remainder_l2sq)).square()
    centered_moment_bound_l2sq = (
        affine_norm + torch.sqrt(centered_moment_remainder_l2sq)
    ).square()
    moment_cross_bound_l2sq = require_nonnegative(
        "hyper_taylor_moment_cross_bound_l2sq",
        affine_l2sq + 2.0 * moment_cross_integral + moment_remainder_l2sq,
    )
    centered_moment_cross_bound_l2sq = require_nonnegative(
        "hyper_taylor_centered_moment_cross_bound_l2sq",
        affine_l2sq
        + 2.0 * centered_moment_cross_integral
        + centered_moment_remainder_l2sq,
    )
    moment_best_bound_l2sq = torch.minimum(moment_bound_l2sq, moment_cross_bound_l2sq)
    centered_moment_best_bound_l2sq = torch.minimum(
        centered_moment_bound_l2sq, centered_moment_cross_bound_l2sq
    )
    return residual, rho, {
        "hyper_taylor_rho": rho,
        "hyper_taylor_rho_centered_hyper": rho_centered_hyper,
        "hyper_taylor_rho_hyper_centered_variation": rho_hyper_centered_variation,
        "hyper_taylor_full_hessian_center_part": full_hessian_center_part,
        "hyper_taylor_full_hessian_error_part": full_hessian_error_part,
        "hyper_taylor_rho_hyper": rho_hyper,
        "hyper_taylor_rho_boundary": rho_boundary_second,
        "hyper_taylor_boundary_hessian_center_part": boundary_hessian_center_part,
        "hyper_taylor_A_qr_center_part": aq_r_center_abs_sum,
        "hyper_taylor_A_qr_error_part": aq_r_error_sum,
        "hyper_taylor_affine_l2sq": affine_l2sq,
        "hyper_taylor_bound_l2sq": bound_l2sq,
        "hyper_taylor_centered_hyper_bound_l2sq": centered_hyper_bound_l2sq,
        "hyper_taylor_moment_remainder_l2sq": moment_remainder_l2sq,
        "hyper_taylor_moment_bound_l2sq": moment_bound_l2sq,
        "hyper_taylor_moment_cross_integral": moment_cross_integral,
        "hyper_taylor_moment_cross_bound_l2sq": moment_cross_bound_l2sq,
        "hyper_taylor_moment_best_bound_l2sq": moment_best_bound_l2sq,
        "hyper_taylor_centered_moment_remainder_l2sq": centered_moment_remainder_l2sq,
        "hyper_taylor_centered_moment_bound_l2sq": centered_moment_bound_l2sq,
        "hyper_taylor_centered_moment_cross_integral": centered_moment_cross_integral,
        "hyper_taylor_centered_moment_cross_bound_l2sq": centered_moment_cross_bound_l2sq,
        "hyper_taylor_centered_moment_best_bound_l2sq": centered_moment_best_bound_l2sq,
        "hyper_taylor_abs_residual": residual.abs(),
    }

def heat_hyper_taylor_rect_batch(
    *,
    model,
    y: torch.Tensor,
    eps_vec: torch.Tensor,
    alpha: float,
    n_taylor: int | None,
):
    """Heat analogue preserving ``F_t-alpha*Delta F`` before absolute bounds."""
    return wave_hyper_taylor_rect_batch(
        model=model,
        y=y,
        eps_vec=eps_vec,
        c2=alpha,
        n_taylor=n_taylor,
        equation="heat",
    )
