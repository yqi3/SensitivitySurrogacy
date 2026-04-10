from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import norm
import sensitivity_surrogacy.nuisance_copula as _nc

from .validation import _validate_longterm_copula_inputs
from .nuisance_shared import (
    varrho_x,
    varrho_s_x,
    phi_x,
    phi_s_x,
    _as_2d_numpy,
    _prepare_glmnet_matrix,
    _predict_propensity_batch,
    _balanced_random_folds,
)
from .nuisance_copula import (
    h_s_x_y,
    mu_s_x_copula,
    d_s_x,
    bar_mu_x_copula,
)


def longterm_copula(
    data,
    S_vars,
    X_vars,
    Y_var,
    type,
    type_prop,
    cop,
    param,
    prop_lb=0.01,
    prop_ub=0.99,
    alpha=0.05,
    integralMaxEval=1000,
    cross_fit_fold=5,
    nuisance_cv_fold=5,
    grf_honesty=True,
    grf_tune_parameters="all",
    grf_num_threads=1,
    xgb_cv_rounds=100,
    xgb_eta=0.1,
    xgb_max_depth=2,
    xgb_threads=1,
):
    """
    Estimate long-term treatment effects for a copula-based sensitivity analysis.

    Parameters
    ----------
    data : pandas.DataFrame
        Analysis dataset.
    S_vars : list[str]
        Names of surrogate outcome variables.
    X_vars : list[str]
        Names of pre-treatment covariates.
    Y_var : str
        Name of the long-term outcome variable.
    type : str
        Method used to estimate WSI nuisance components.
    type_prop : str
        Method used to estimate propensity-related nuisance components.
    cop : str
        Copula family. Supported: "Frank", "Plackett".
    param : float
        Copula parameter.
    prop_lb : float, default=0.01
        Lower trimming threshold for propensity scores.
    prop_ub : float, default=0.99
        Upper trimming threshold for propensity scores.
    alpha : float, default=0.05
        Significance level for confidence intervals.
    integralMaxEval : int, default=1000
        Number of evaluation points for numerical integration.
    cross_fit_fold : int, default=5
        Number of folds for cross-fitting.
    nuisance_cv_fold : int, default=5
        Number of folds for CV inside nuisance estimation.
    grf_honesty : bool, default=True
        Honesty option for grf-style learners.
    grf_tune_parameters : str, default="all"
        grf tuning mode.
    grf_num_threads : int, default=1
        Number of threads for grf-style learners.
    xgb_cv_rounds : int, default=100
        Maximum xgboost CV rounds.
    xgb_eta : float, default=0.1
        xgboost learning rate.
    xgb_max_depth : int, default=2
        xgboost max tree depth.
    xgb_threads : int, default=1
        Number of xgboost threads.

    Returns
    -------
    dict
        Dictionary with keys:
        - "hat_tau"
        - "se"
        - "ci"
    """
    _validate_longterm_copula_inputs(
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
    )

    fold_ids = _balanced_random_folds(len(data), cross_fit_fold)
    folds = {
        fold: np.where(fold_ids == fold)[0]
        for fold in range(1, cross_fit_fold + 1)
    }

    data_nuisance_out = [
        compute_nuisance_on_fold_copula(
            i=i,
            data=data,
            folds=folds,
            S_vars=S_vars,
            X_vars=X_vars,
            Y_var=Y_var,
            type=type,
            type_prop=type_prop,
            cop=cop,
            param=param,
            prop_lb=prop_lb,
            prop_ub=prop_ub,
            integralMaxEval=integralMaxEval,
            cross_fit_fold=cross_fit_fold,
            nuisance_cv_fold=nuisance_cv_fold,
            grf_honesty=grf_honesty,
            grf_tune_parameters=grf_tune_parameters,
            grf_num_threads=grf_num_threads,
            xgb_cv_rounds=xgb_cv_rounds,
            xgb_eta=xgb_eta,
            xgb_max_depth=xgb_max_depth,
            xgb_threads=xgb_threads,
        )
        for i in range(1, cross_fit_fold + 1)
    ]

    data_nuisance = pd.concat(data_nuisance_out, axis=0, ignore_index=True)

    hat_tau = compute_tau_copula(data_nuisance)

    se_ci = compute_ci_copula(hat_tau, data_nuisance, alpha)
    se = se_ci["se"]
    ci = se_ci["ci"]

    return {
        "hat_tau": hat_tau,
        "se": se,
        "ci": ci
    }


def compute_nuisance_on_fold_copula(
    i,
    data,
    folds,
    S_vars,
    X_vars,
    Y_var,
    type,
    type_prop,
    cop,
    param,
    prop_lb,
    prop_ub,
    integralMaxEval,
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
    print(
        f"------Estimating fold {i}/{cross_fit_fold} for {cop} copula with parameter = {param}------"
    )

    test_idx = folds[i]
    train_idx = np.concatenate(
        [folds[j] for j in range(1, cross_fit_fold + 1) if j != i]
    )

    data_train = data.iloc[train_idx, :].copy()
    data_test = data.iloc[test_idx, :].copy()

    nuisance = train_nuisance_copula(
        data_train=data_train,
        S_vars=S_vars,
        X_vars=X_vars,
        Y_var=Y_var,
        type=type,
        type_prop=type_prop,
        cop=cop,
        param=param,
        prop_lb=prop_lb,
        prop_ub=prop_ub,
        integralMaxEval=integralMaxEval,
        nuisance_cv_fold=nuisance_cv_fold,
        grf_honesty=grf_honesty,
        grf_tune_parameters=grf_tune_parameters,
        grf_num_threads=grf_num_threads,
        xgb_cv_rounds=xgb_cv_rounds,
        xgb_eta=xgb_eta,
        xgb_max_depth=xgb_max_depth,
        xgb_threads=xgb_threads,
    )

    print("Computing nuisance terms...")

    data_nuisance = data_test.loc[:, [Y_var, "treatment", "observe"]].copy()
    x = data_test.loc[:, list(X_vars)].copy()
    x_copy = x.copy()
    s = data_test.loc[:, list(S_vars)].copy()
    s_x = data_test.loc[:, list(S_vars) + list(X_vars)].copy()
    y = data_test.loc[:, Y_var].to_numpy()

    _nc.counter = 0
    _nc.total = data_test.shape[0]

    print("h_1_s_x_y")
    data_nuisance["h_1_s_x_y"] = [
        h_s_x_y(
            W=1,
            S=s.iloc[r, :].to_numpy(),
            X=x.iloc[r, :].to_numpy(),
            Y=y[r],
            varrho_s_x_out=nuisance["varrho_s_x"],
            data_train=data_train,
            S_vars=S_vars,
            X_vars=X_vars,
            Y_var=Y_var,
            type_prop=type_prop,
            nuisance_cv_fold=nuisance_cv_fold,
            prop_ub=prop_ub,
            prop_lb=prop_lb,
            integralMaxEval=integralMaxEval,
            cop=cop,
            param=param,
        )
        for r in range(len(data_test))
    ]

    _nc.counter = 0
    print("h_0_s_x_y")
    data_nuisance["h_0_s_x_y"] = [
        h_s_x_y(
            W=0,
            S=s.iloc[r, :].to_numpy(),
            X=x.iloc[r, :].to_numpy(),
            Y=y[r],
            varrho_s_x_out=nuisance["varrho_s_x"],
            data_train=data_train,
            S_vars=S_vars,
            X_vars=X_vars,
            Y_var=Y_var,
            type_prop=type_prop,
            nuisance_cv_fold=nuisance_cv_fold,
            prop_ub=prop_ub,
            prop_lb=prop_lb,
            integralMaxEval=integralMaxEval,
            cop=cop,
            param=param,
        )
        for r in range(len(data_test))
    ]

    _nc.counter = 0
    print("mu_1_s_x")
    data_nuisance["mu_1_s_x"] = [
        mu_s_x_copula(
            W=1,
            S=s.iloc[r, :].to_numpy(),
            X=x.iloc[r, :].to_numpy(),
            varrho_s_x_out=nuisance["varrho_s_x"],
            data_train=data_train,
            S_vars=S_vars,
            X_vars=X_vars,
            Y_var=Y_var,
            type_prop=type_prop,
            prop_ub=prop_ub,
            prop_lb=prop_lb,
            integralMaxEval=integralMaxEval,
            cop=cop,
            param=param,
        )
        for r in range(len(data_test))
    ]

    _nc.counter = 0
    print("mu_0_s_x")
    data_nuisance["mu_0_s_x"] = [
        mu_s_x_copula(
            W=0,
            S=s.iloc[r, :].to_numpy(),
            X=x.iloc[r, :].to_numpy(),
            varrho_s_x_out=nuisance["varrho_s_x"],
            data_train=data_train,
            S_vars=S_vars,
            X_vars=X_vars,
            Y_var=Y_var,
            type_prop=type_prop,
            prop_ub=prop_ub,
            prop_lb=prop_lb,
            integralMaxEval=integralMaxEval,
            cop=cop,
            param=param,
        )
        for r in range(len(data_test))
    ]

    _nc.counter = 0
    print("d_s_x")
    data_nuisance["d_s_x"] = [
        d_s_x(
            S=s.iloc[r, :].to_numpy(),
            X=x.iloc[r, :].to_numpy(),
            varrho_s_x_out=nuisance["varrho_s_x"],
            data_train=data_train,
            S_vars=S_vars,
            X_vars=X_vars,
            Y_var=Y_var,
            type_prop=type_prop,
            prop_ub=prop_ub,
            prop_lb=prop_lb,
            integralMaxEval=integralMaxEval,
            cop=cop,
            param=param,
        )
        for r in range(len(data_test))
    ]

    data_nuisance["bar_mu_1_x"] = _predict_regression_or_constant(
        nuisance["bar_mu_1_x"],
        x,
        type,
    )
    data_nuisance["bar_mu_0_x"] = _predict_regression_or_constant(
        nuisance["bar_mu_0_x"],
        x,
        type,
    )

    data_nuisance["varrho_x"] = _predict_propensity_batch(
        nuisance["varrho_x"], x_copy, type_prop, prop_lb, prop_ub
    )
    data_nuisance["varrho_s_x"] = _predict_propensity_batch(
        nuisance["varrho_s_x"], s_x, type_prop, prop_lb, prop_ub
    )
    data_nuisance["phi_x"] = _predict_propensity_batch(
        nuisance["phi_x"], x_copy, type_prop, prop_lb, prop_ub
    )
    data_nuisance["phi_s_x"] = _predict_propensity_batch(
        nuisance["phi_s_x"], s_x, type_prop, prop_lb, prop_ub
    )

    data_nuisance["phi"] = 1.0 - float(np.mean(data_train["observe"]))

    return data_nuisance


def train_nuisance_copula(
    data_train,
    S_vars,
    X_vars,
    Y_var,
    type,
    type_prop,
    cop,
    param,
    prop_lb,
    prop_ub,
    integralMaxEval,
    nuisance_cv_fold,
    grf_honesty,
    grf_tune_parameters,
    grf_num_threads,
    xgb_cv_rounds,
    xgb_eta,
    xgb_max_depth,
    xgb_threads,
):
    print("Training propensity score models...")

    varrho_x_out = varrho_x(
        data_train,
        X_vars,
        type_prop,
        nuisance_cv_fold,
        grf_honesty,
        grf_tune_parameters,
        grf_num_threads,
        xgb_cv_rounds,
        xgb_eta,
        xgb_max_depth,
        xgb_threads,
    )

    varrho_s_x_out = varrho_s_x(
        data_train,
        X_vars,
        S_vars,
        type_prop,
        nuisance_cv_fold,
        grf_honesty,
        grf_tune_parameters,
        grf_num_threads,
        xgb_cv_rounds,
        xgb_eta,
        xgb_max_depth,
        xgb_threads,
    )

    phi_x_out = phi_x(
        data_train,
        X_vars,
        type_prop,
        nuisance_cv_fold,
        grf_honesty,
        grf_tune_parameters,
        grf_num_threads,
        xgb_cv_rounds,
        xgb_eta,
        xgb_max_depth,
        xgb_threads,
    )

    phi_s_x_out = phi_s_x(
        data_train,
        X_vars,
        S_vars,
        type_prop,
        nuisance_cv_fold,
        grf_honesty,
        grf_tune_parameters,
        grf_num_threads,
        xgb_cv_rounds,
        xgb_eta,
        xgb_max_depth,
        xgb_threads,
    )

    print("Training bar_mu models...")
    print("bar_mu_1_x")
    bar_mu_1_x_out = bar_mu_x_copula(
        nuisance_type=type,
        W=1,
        data_train=data_train,
        S_vars=S_vars,
        X_vars=X_vars,
        Y_var=Y_var,
        nuisance_cv_fold=nuisance_cv_fold,
        type_prop=type_prop,
        prop_ub=prop_ub,
        prop_lb=prop_lb,
        integralMaxEval=integralMaxEval,
        cop=cop,
        param=param,
        grf_honesty=grf_honesty,
        grf_tune_parameters=grf_tune_parameters,
        grf_num_threads=grf_num_threads,
        xgb_cv_rounds=xgb_cv_rounds,
        xgb_eta=xgb_eta,
        xgb_max_depth=xgb_max_depth,
        xgb_threads=xgb_threads,
    )

    print("bar_mu_0_x")
    bar_mu_0_x_out = bar_mu_x_copula(
        nuisance_type=type,
        W=0,
        data_train=data_train,
        S_vars=S_vars,
        X_vars=X_vars,
        Y_var=Y_var,
        nuisance_cv_fold=nuisance_cv_fold,
        type_prop=type_prop,
        prop_ub=prop_ub,
        prop_lb=prop_lb,
        integralMaxEval=integralMaxEval,
        cop=cop,
        param=param,
        grf_honesty=grf_honesty,
        grf_tune_parameters=grf_tune_parameters,
        grf_num_threads=grf_num_threads,
        xgb_cv_rounds=xgb_cv_rounds,
        xgb_eta=xgb_eta,
        xgb_max_depth=xgb_max_depth,
        xgb_threads=xgb_threads,
    )

    return {
        "bar_mu_1_x": bar_mu_1_x_out,
        "bar_mu_0_x": bar_mu_0_x_out,
        "varrho_x": varrho_x_out,
        "varrho_s_x": varrho_s_x_out,
        "phi_x": phi_x_out,
        "phi_s_x": phi_s_x_out,
    }


def compute_tau_copula(data_nuisance):
    m = data_nuisance.copy()

    m["m_0_a"] = ((1 - m["observe"]) / m["phi"]) * (
        (m["treatment"] / m["varrho_x"]) * (m["mu_1_s_x"] - m["bar_mu_1_x"])
        - ((1 - m["treatment"]) / (1 - m["varrho_x"])) * (m["mu_0_s_x"] - m["bar_mu_0_x"])
    )

    m["m_0_b"] = ((1 - m["observe"]) / m["phi"]) * (
        m["bar_mu_1_x"] - m["bar_mu_0_x"]
    )

    m["m_0_c"] = (m["observe"] / m["phi"]) * (m["phi_s_x"] / (1 - m["phi_s_x"])) * (
        (m["varrho_s_x"] / m["varrho_x"]) * (m["h_1_s_x_y"] - m["mu_1_s_x"])
        - ((1 - m["varrho_s_x"]) / (1 - m["varrho_x"])) * (m["h_0_s_x_y"] - m["mu_0_s_x"])
    )

    m["m_0_d"] = ((1 - m["observe"]) / m["phi"]) * (1 / m["varrho_x"]) * (
        (m["d_s_x"] - m["mu_1_s_x"]) * (m["treatment"] - m["varrho_s_x"])
    )

    m["m_0_e"] = ((1 - m["observe"]) / m["phi"]) * (1 / (1 - m["varrho_x"])) * (
        (m["d_s_x"] - m["mu_0_s_x"]) * (m["treatment"] - m["varrho_s_x"])
    )

    m["m_0"] = m["m_0_a"] + m["m_0_b"] + m["m_0_c"] + m["m_0_d"] + m["m_0_e"]

    multiplier = float(np.mean(1 - data_nuisance["observe"]) / data_nuisance["phi"].iloc[0])
    return float(np.mean(m["m_0"]) / multiplier)


def compute_m_table_copula(tau_0, data_nuisance):
    m = data_nuisance.copy()

    m["m_0_a"] = ((1 - m["observe"]) / m["phi"]) * (
        (m["treatment"] / m["varrho_x"]) * (m["mu_1_s_x"] - m["bar_mu_1_x"])
        - ((1 - m["treatment"]) / (1 - m["varrho_x"])) * (m["mu_0_s_x"] - m["bar_mu_0_x"])
    )

    m["m_0_b"] = ((1 - m["observe"]) / m["phi"]) * (
        m["bar_mu_1_x"] - m["bar_mu_0_x"] - tau_0
    )

    m["m_0_c"] = (m["observe"] / m["phi"]) * (m["phi_s_x"] / (1 - m["phi_s_x"])) * (
        (m["varrho_s_x"] / m["varrho_x"]) * (m["h_1_s_x_y"] - m["mu_1_s_x"])
        - ((1 - m["varrho_s_x"]) / (1 - m["varrho_x"])) * (m["h_0_s_x_y"] - m["mu_0_s_x"])
    )

    m["m_0_d"] = ((1 - m["observe"]) / m["phi"]) * (1 / m["varrho_x"]) * (
        (m["d_s_x"] - m["mu_1_s_x"]) * (m["treatment"] - m["varrho_s_x"])
    )

    m["m_0_e"] = ((1 - m["observe"]) / m["phi"]) * (1 / (1 - m["varrho_x"])) * (
        (m["d_s_x"] - m["mu_0_s_x"]) * (m["treatment"] - m["varrho_s_x"])
    )

    m["m_0"] = m["m_0_a"] + m["m_0_b"] + m["m_0_c"] + m["m_0_d"] + m["m_0_e"]

    return m


def compute_ci_copula(hat_tau, data_nuisance, alpha):
    influence = compute_m_table_copula(hat_tau, data_nuisance)["m_0"].to_numpy(dtype=float)
    hat_V = float(np.var(influence, ddof=1))

    z = float(norm.ppf(1 - alpha / 2))
    se = float(np.sqrt(hat_V / len(influence)))
    ci_l = hat_tau - z * se
    ci_u = hat_tau + z * se

    return {
        "se": se,
        "ci": np.array([ci_l, ci_u], dtype=float),
    }


def _predict_regression_or_constant(model_or_scalar, new_data, nuisance_type):
    if np.isscalar(model_or_scalar):
        return np.repeat(float(model_or_scalar), len(new_data))

    if nuisance_type == "glmnet":
        x_mat = _prepare_glmnet_matrix(new_data)
        pred = np.asarray(model_or_scalar.predict(x_mat, lamb=[model_or_scalar.lambda_max_])).reshape(-1)
        return pred

    if nuisance_type == "grf":
        pred = np.asarray(model_or_scalar.predict(_as_2d_numpy(new_data))).reshape(-1)
        return pred

    if nuisance_type == "xgboost":
        if isinstance(new_data, pd.DataFrame):
            dnew = xgb.DMatrix(_as_2d_numpy(new_data), feature_names=list(new_data.columns))
        else:
            dnew = xgb.DMatrix(_as_2d_numpy(new_data))
        pred = np.asarray(model_or_scalar.predict(dnew)).reshape(-1)
        return pred

    raise ValueError("Enter a valid nuisance parameter estimation type")