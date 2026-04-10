from __future__ import annotations

from numbers import Real, Integral
from typing import Sequence

import numpy as np
import pandas as pd


def _validate_longterm_partial_id_inputs(
    data,
    S_vars,
    X_vars,
    Y_var,
    type,
    type_prop,
    prop_lb,
    prop_ub,
    alpha,
    cross_fit_fold,
    nuisance_cv_fold,
    grf_honesty,
    grf_tune_parameters,
    grf_num_threads,
    xgb_cv_rounds,
    xgb_eta,
    xgb_max_depth,
    xgb_threads,
):
    if not isinstance(data, pd.DataFrame):
        raise TypeError("`data` must be a data.frame.")

    if not _is_nonempty_string_sequence(S_vars):
        raise TypeError("`S_vars` must be a non-empty character vector of column names.")

    if not _is_nonempty_string_sequence(X_vars):
        raise TypeError("`X_vars` must be a non-empty character vector of column names.")

    if not isinstance(Y_var, str):
        raise TypeError("`Y_var` must be a single character string naming the outcome column.")

    required_cols = list(dict.fromkeys(["treatment", "observe", *S_vars, *X_vars, Y_var]))
    missing_cols = [col for col in required_cols if col not in data.columns]
    if len(missing_cols) > 0:
        raise ValueError(
            "The following required columns are missing from `data`: "
            + ", ".join(missing_cols)
        )

    if len(set(S_vars).intersection(X_vars)) > 0:
        raise ValueError("`S_vars` and `X_vars` must not overlap.")

    if not data["treatment"].isin([0, 1]).all():
        raise ValueError("`treatment` must contain only 0/1 values.")

    if not data["observe"].isin([0, 1]).all():
        raise ValueError("`observe` must contain only 0/1 values.")

    valid_type = ["glmnet", "grf", "xgboost"]
    if type not in valid_type:
        raise ValueError("`type` must be one of: " + ", ".join(valid_type))

    if type_prop not in valid_type:
        raise ValueError("`type_prop` must be one of: " + ", ".join(valid_type))

    if (
        not _is_scalar_finite_number(prop_lb)
        or not _is_scalar_finite_number(prop_ub)
        or prop_lb <= 0
        or prop_ub >= 1
        or prop_lb >= prop_ub
    ):
        raise ValueError("`prop_lb` and `prop_ub` must satisfy 0 < prop_lb < prop_ub < 1.")

    if (
        not _is_scalar_finite_number(alpha)
        or alpha <= 0
        or alpha >= 1
    ):
        raise ValueError("`alpha` must be a scalar in (0, 1).")

    if not _is_integer_scalar(cross_fit_fold) or cross_fit_fold < 2:
        raise ValueError("`cross_fit_fold` must be an integer >= 2.")

    if not _is_integer_scalar(nuisance_cv_fold) or nuisance_cv_fold < 2:
        raise ValueError("`nuisance_cv_fold` must be an integer >= 2.")

    if len(data) < cross_fit_fold:
        raise ValueError("`cross_fit_fold` cannot exceed the number of rows in `data`.")

    if (data["observe"] == 0).sum() == 0 or (data["observe"] == 1).sum() == 0:
        raise ValueError(
            "`data` must contain both experimental (`observe = 0`) and observational (`observe = 1`) samples."
        )

    exp_data = data.loc[data["observe"] == 0, :]
    if exp_data["treatment"].nunique() < 2:
        raise ValueError(
            "The experimental sample (`observe = 0`) must contain both treatment arms 0 and 1."
        )

    # Additional learner-parameter checks
    if not isinstance(grf_honesty, bool):
        raise TypeError("`grf_honesty` must be a boolean.")

    if not isinstance(grf_tune_parameters, str):
        raise TypeError("`grf_tune_parameters` must be a character string.")

    if not _is_integer_scalar(grf_num_threads) or grf_num_threads < 1:
        raise ValueError("`grf_num_threads` must be an integer >= 1.")

    if not _is_integer_scalar(xgb_cv_rounds) or xgb_cv_rounds < 1:
        raise ValueError("`xgb_cv_rounds` must be an integer >= 1.")

    if not _is_scalar_finite_number(xgb_eta) or xgb_eta <= 0:
        raise ValueError("`xgb_eta` must be a positive finite scalar.")

    if not _is_integer_scalar(xgb_max_depth) or xgb_max_depth < 1:
        raise ValueError("`xgb_max_depth` must be an integer >= 1.")

    if not _is_integer_scalar(xgb_threads) or xgb_threads < 1:
        raise ValueError("`xgb_threads` must be an integer >= 1.")

    return True


def _validate_longterm_copula_inputs(
    data,
    S_vars,
    X_vars,
    Y_var,
    type,
    type_prop,
    cop,
    param,
    prop_lb,
    prop_ub,
    alpha,
    integralMaxEval,
    cross_fit_fold,
    nuisance_cv_fold,
):
    if not isinstance(data, pd.DataFrame):
        raise TypeError("`data` must be a data.frame.")

    if not _is_nonempty_string_sequence(S_vars):
        raise TypeError("`S_vars` must be a non-empty character vector of column names.")

    if not _is_nonempty_string_sequence(X_vars):
        raise TypeError("`X_vars` must be a non-empty character vector of column names.")

    if not isinstance(Y_var, str):
        raise TypeError("`Y_var` must be a single column name.")

    required_cols = list(dict.fromkeys(["treatment", "observe", *S_vars, *X_vars, Y_var]))
    missing_cols = [col for col in required_cols if col not in data.columns]
    if len(missing_cols) > 0:
        raise ValueError("Missing required columns in `data`: " + ", ".join(missing_cols))

    if len(set(S_vars).intersection(X_vars)) > 0:
        raise ValueError("`S_vars` and `X_vars` must not overlap.")

    if not data["treatment"].isin([0, 1]).all():
        raise ValueError("`treatment` must contain only 0/1 values.")

    if not data["observe"].isin([0, 1]).all():
        raise ValueError("`observe` must contain only 0/1 values.")

    valid_type = ["glmnet", "grf", "xgboost"]
    if type not in valid_type:
        raise ValueError("`type` must be one of: " + ", ".join(valid_type))

    if type_prop not in valid_type:
        raise ValueError("`type_prop` must be one of: " + ", ".join(valid_type))

    valid_cop = ["Frank", "Plackett"]
    if cop not in valid_cop:
        raise ValueError("`cop` must be either 'Frank' or 'Plackett'.")

    if cop == "Frank":
        if (
            not _is_scalar_finite_number(param)
            or abs(param) < 1e-10
        ):
            raise ValueError(
                "For `cop = 'Frank'`, `param` must be finite and not too close to 0."
            )

    if cop == "Plackett":
        if (
            not _is_scalar_finite_number(param)
            or param <= 0
        ):
            raise ValueError(
                "For `cop = 'Plackett'`, `param` must be a finite positive scalar."
            )

    if (
        not _is_scalar_finite_number(prop_lb)
        or not _is_scalar_finite_number(prop_ub)
        or prop_lb <= 0
        or prop_ub >= 1
        or prop_lb >= prop_ub
    ):
        raise ValueError("`prop_lb` and `prop_ub` must satisfy 0 < prop_lb < prop_ub < 1.")

    if (
        not _is_scalar_finite_number(alpha)
        or alpha <= 0
        or alpha >= 1
    ):
        raise ValueError("`alpha` must be a scalar in (0, 1).")

    if not _is_integer_scalar(cross_fit_fold) or cross_fit_fold < 2:
        raise ValueError("`cross_fit_fold` must be an integer >= 2.")

    if not _is_integer_scalar(nuisance_cv_fold) or nuisance_cv_fold < 2:
        raise ValueError("`nuisance_cv_fold` must be an integer >= 2.")

    if not _is_integer_scalar(integralMaxEval) or integralMaxEval < 2:
        raise ValueError("`integralMaxEval` must be an integer >= 2.")

    if len(data) < cross_fit_fold:
        raise ValueError("`cross_fit_fold` cannot exceed the number of rows in `data`.")

    if (data["observe"] == 0).sum() == 0 or (data["observe"] == 1).sum() == 0:
        raise ValueError(
            "`data` must contain both observational and experimental samples."
        )

    return True


def _is_nonempty_string_sequence(x) -> bool:
    if isinstance(x, str):
        return False
    if not isinstance(x, Sequence):
        return False
    if len(x) < 1:
        return False
    return all(isinstance(elem, str) for elem in x)


def _is_scalar_finite_number(x) -> bool:
    return isinstance(x, Real) and np.isfinite(x)


def _is_integer_scalar(x) -> bool:
    return isinstance(x, Integral) or (
        _is_scalar_finite_number(x) and float(x).is_integer()
    )