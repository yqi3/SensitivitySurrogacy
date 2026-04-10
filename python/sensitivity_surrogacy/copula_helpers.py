from __future__ import annotations

from numbers import Real

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq


def tau_plackett(theta):
    if not _is_scalar_finite_number(theta) or theta <= 0:
        raise ValueError("`theta` must be a finite positive scalar for the Plackett copula.")

    if abs(theta - 1) < 1e-8:
        return 0.0

    return 1 + 2 / (theta - 1) - (2 * theta * np.log(theta)) / (theta - 1) ** 2


def plackett_param_from_tau(tau_target, tol=1e-6):
    """
    Convert Kendall's tau to a Plackett copula parameter.

    This helper function maps a user-specified Kendall's tau value to the
    corresponding parameter of the Plackett copula. It is intended to simplify
    sensitivity analysis with `longterm_copula`, since Kendall's tau provides
    a more interpretable dependence measure than the raw copula parameter.

    The function solves the nonlinear relationship between Kendall's tau and
    the Plackett copula parameter via numerical root finding.

    Parameters
    ----------
    tau_target : float
        Target Kendall's tau value. Must lie strictly between -1 and 1.

    tol : float, default=1e-6
        Tolerance for numerical root finding.

    Returns
    -------
    float
        Plackett copula parameter corresponding to `tau_target`.

        When `tau_target` is sufficiently close to zero, the function returns 1,
        which corresponds to the independence case.

    Examples
    --------
    >>> tau0 = 0.3
    >>> theta = plackett_param_from_tau(tau0)
    >>> theta

    See Also
    --------
    frank_param_from_tau
    longterm_copula
    """
    if not _is_scalar_finite_number(tau_target):
        raise ValueError("`tau_target` must be a finite numeric scalar.")

    if tau_target <= -1 or tau_target >= 1:
        raise ValueError(
            "For the Plackett copula, `tau_target` must lie strictly between -1 and 1."
        )

    if not _is_scalar_finite_number(tol) or tol <= 0:
        raise ValueError("`tol` must be a positive numeric scalar.")

    if abs(tau_target) < tol:
        return 1.0

    def objective(theta):
        return tau_plackett(theta) - tau_target

    if tau_target < 0:
        return brentq(objective, 1e-8, 1 - 1e-8, xtol=tol)
    else:
        return brentq(objective, 1 + 1e-8, 1e8, xtol=tol)


def frank_param_from_tau(tau_target, tol=1e-6, independence_buffer=1e-6):
    """
    Convert Kendall's tau to a Frank copula parameter.

    This helper function maps a user-specified Kendall's tau value to the
    corresponding parameter of the Frank copula. It is intended to simplify
    sensitivity analysis with `longterm_copula`, since Kendall's tau provides
    a more interpretable dependence measure than the raw copula parameter.

    The function numerically inverts the relationship between Kendall's tau
    and the Frank copula parameter. For tau values sufficiently close to zero,
    the function returns a small nonzero value rather than exactly zero,
    since the Frank copula parameterization becomes numerically unstable
    at zero in downstream estimation routines.

    Parameters
    ----------
    tau_target : float
        Target Kendall's tau value. Must lie strictly between -1 and 1.

    tol : float, default=1e-6
        Tolerance used for zero-detection and root finding.

    independence_buffer : float, default=1e-6
        Small nonzero value returned when tau is close to zero. This avoids
        numerical instability at the independence case.

    Returns
    -------
    float
        Frank copula parameter corresponding to `tau_target`.

    Examples
    --------
    >>> tau0 = 0.3
    >>> theta = frank_param_from_tau(tau0)
    >>> theta

    See Also
    --------
    plackett_param_from_tau
    longterm_copula
    """
    if not _is_scalar_finite_number(tau_target):
        raise ValueError("`tau_target` must be a finite numeric scalar.")

    if tau_target <= -1 or tau_target >= 1:
        raise ValueError(
            "For the Frank copula, `tau_target` must lie strictly between -1 and 1."
        )

    if not _is_scalar_finite_number(tol) or tol <= 0:
        raise ValueError("`tol` must be a positive numeric scalar.")

    if (
        not _is_scalar_finite_number(independence_buffer)
        or independence_buffer <= 0
    ):
        raise ValueError("`independence_buffer` must be a positive numeric scalar.")

    if abs(tau_target) < tol:
        return float(independence_buffer)

    def debye(theta):
        if theta == 0:
            return 1.0

        def integrand(t):
            return t / (np.exp(t) - 1)

        value, _ = quad(integrand, 0, theta)
        return value / theta

    def objective(theta):
        return tau_target - (1 - (4 / theta) * (1 - debye(theta)))

    return brentq(objective, -500, 500, xtol=tol)


def _is_scalar_finite_number(x) -> bool:
    return isinstance(x, Real) and np.isfinite(x)