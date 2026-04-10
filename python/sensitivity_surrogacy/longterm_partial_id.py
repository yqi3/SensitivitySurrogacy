from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import norm

from .validation import _validate_longterm_partial_id_inputs
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
from .nuisance_partial_id import (
    mu_s_x_partial_id,
    bar_mu_x_partial_id,
    q_s_x,
    H_y_s,
)


def longterm_partial_id(
    data,
    S_vars,
    X_vars,
    Y_var,
    type,
    type_prop,
    prop_lb=0.01,
    prop_ub=0.99,
    alpha=0.05,
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
    Estimate worst-case (partial-identification) bounds for long-term treatment effects.

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
        Method used to estimate outcome nuisance components.
    type_prop : str
        Method used to estimate propensity-related nuisance components.
    prop_lb : float, default=0.01
        Lower trimming threshold for propensity scores.
    prop_ub : float, default=0.99
        Upper trimming threshold for propensity scores.
    alpha : float, default=0.05
        Significance level for confidence intervals.
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
        - "hat_tau_upper"
        - "se_upper"
        - "ci_upper"
        - "hat_tau_lower"
        - "se_lower"
        - "ci_lower"
    """
    _validate_longterm_partial_id_inputs(
        data=data,
        S_vars=S_vars,
        X_vars=X_vars,
        Y_var=Y_var,
        type=type,
        type_prop=type_prop,
        prop_lb=prop_lb,
        prop_ub=prop_ub,
        alpha=alpha,
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

    fold_ids = _balanced_random_folds(len(data), cross_fit_fold)
    folds = {
        fold: np.where(fold_ids == fold)[0]
        for fold in range(1, cross_fit_fold + 1)
    }

    data_nuisance_out = [
        compute_nuisance_on_fold_partial_id(
            i=i,
            data=data,
            folds=folds,
            S_vars=S_vars,
            X_vars=X_vars,
            Y_var=Y_var,
            type=type,
            type_prop=type_prop,
            prop_lb=prop_lb,
            prop_ub=prop_ub,
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

    hat_tau_upper = compute_tau_partial_id(data_nuisance, Y_var, "upper")
    hat_tau_lower = compute_tau_partial_id(data_nuisance, Y_var, "lower")

    se_ci = compute_bound_ci_partial_id(hat_tau_upper, data_nuisance, Y_var, "upper", alpha)
    se_upper = se_ci["se"]
    ci_upper = se_ci["ci"]

    se_ci = compute_bound_ci_partial_id(hat_tau_lower, data_nuisance, Y_var, "lower", alpha)
    se_lower = se_ci["se"]
    ci_lower = se_ci["ci"]

    return {
        "hat_tau_upper": hat_tau_upper,
        "se_upper": se_upper,
        "ci_upper": ci_upper,
        "hat_tau_lower": hat_tau_lower,
        "se_lower": se_lower,
        "ci_lower": ci_lower,
    }


def compute_nuisance_on_fold_partial_id(
    i,
    data,
    folds,
    S_vars,
    X_vars,
    Y_var,
    type,
    type_prop,
    prop_lb,
    prop_ub,
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
    print(f"------Estimating fold {i}/{cross_fit_fold}------")

    test_idx = folds[i]
    train_idx = np.concatenate(
        [folds[j] for j in range(1, cross_fit_fold + 1) if j != i]
    )

    data_train = data.iloc[train_idx, :].copy()
    data_test = data.iloc[test_idx, :].copy()

    nuisance = train_nuisance_partial_id(
        data_train=data_train,
        S_vars=S_vars,
        X_vars=X_vars,
        Y_var=Y_var,
        type=type,
        type_prop=type_prop,
        prop_lb=prop_lb,
        prop_ub=prop_ub,
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

    data_nuisance["mu_1U_s_x"] = [
        mu_s_x_partial_id(
            mu_type="1U",
            S=s.iloc[r, :].to_numpy(),
            X=x.iloc[r, :].to_numpy(),
            varrho_s_x_out=nuisance["varrho_s_x"],
            data_train=data_train,
            S_vars=S_vars,
            X_vars=X_vars,
            Y_var=Y_var,
            type_prop=type_prop,
            nuisance_cv_fold=nuisance_cv_fold,
            prop_ub=prop_ub,
            prop_lb=prop_lb,
        )
        for r in range(len(data_test))
    ]

    data_nuisance["mu_1L_s_x"] = [
        mu_s_x_partial_id(
            mu_type="1L",
            S=s.iloc[r, :].to_numpy(),
            X=x.iloc[r, :].to_numpy(),
            varrho_s_x_out=nuisance["varrho_s_x"],
            data_train=data_train,
            S_vars=S_vars,
            X_vars=X_vars,
            Y_var=Y_var,
            type_prop=type_prop,
            nuisance_cv_fold=nuisance_cv_fold,
            prop_ub=prop_ub,
            prop_lb=prop_lb,
        )
        for r in range(len(data_test))
    ]

    data_nuisance["mu_0U_s_x"] = [
        mu_s_x_partial_id(
            mu_type="0U",
            S=s.iloc[r, :].to_numpy(),
            X=x.iloc[r, :].to_numpy(),
            varrho_s_x_out=nuisance["varrho_s_x"],
            data_train=data_train,
            S_vars=S_vars,
            X_vars=X_vars,
            Y_var=Y_var,
            type_prop=type_prop,
            nuisance_cv_fold=nuisance_cv_fold,
            prop_ub=prop_ub,
            prop_lb=prop_lb,
        )
        for r in range(len(data_test))
    ]

    data_nuisance["mu_0L_s_x"] = [
        mu_s_x_partial_id(
            mu_type="0L",
            S=s.iloc[r, :].to_numpy(),
            X=x.iloc[r, :].to_numpy(),
            varrho_s_x_out=nuisance["varrho_s_x"],
            data_train=data_train,
            S_vars=S_vars,
            X_vars=X_vars,
            Y_var=Y_var,
            type_prop=type_prop,
            nuisance_cv_fold=nuisance_cv_fold,
            prop_ub=prop_ub,
            prop_lb=prop_lb,
        )
        for r in range(len(data_test))
    ]

    data_nuisance["q_U_s_x"] = [
        q_s_x(
            q_type="U",
            data_train=data_train,
            S_vars=S_vars,
            X_vars=X_vars,
            Y_var=Y_var,
            type_prop=type_prop,
            prop_ub=prop_ub,
            prop_lb=prop_lb,
            S=s.iloc[r, :].to_numpy(),
            X=x.iloc[r, :].to_numpy(),
            varrho_s_x_out=nuisance["varrho_s_x"],
        )
        for r in range(len(data_test))
    ]

    data_nuisance["q_L_s_x"] = [
        q_s_x(
            q_type="L",
            data_train=data_train,
            S_vars=S_vars,
            X_vars=X_vars,
            Y_var=Y_var,
            type_prop=type_prop,
            prop_ub=prop_ub,
            prop_lb=prop_lb,
            S=s.iloc[r, :].to_numpy(),
            X=x.iloc[r, :].to_numpy(),
            varrho_s_x_out=nuisance["varrho_s_x"],
        )
        for r in range(len(data_test))
    ]

    data_nuisance["bar_mu_1U_x"] = _predict_regression_model(
        nuisance["bar_mu_1U_x"], x, type
    )
    data_nuisance["bar_mu_1L_x"] = _predict_regression_model(
        nuisance["bar_mu_1L_x"], x, type
    )
    data_nuisance["bar_mu_0U_x"] = _predict_regression_model(
        nuisance["bar_mu_0U_x"], x, type
    )
    data_nuisance["bar_mu_0L_x"] = _predict_regression_model(
        nuisance["bar_mu_0L_x"], x, type
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


def train_nuisance_partial_id(
    data_train,
    S_vars,
    X_vars,
    Y_var,
    type,
    type_prop,
    prop_lb,
    prop_ub,
    nuisance_cv_fold,
    grf_honesty,
    grf_tune_parameters,
    grf_num_threads,
    xgb_cv_rounds,
    xgb_eta,
    xgb_max_depth,
    xgb_threads,
):
    print("Training outcome models...")

    bar_mu_1U_x_out = bar_mu_x_partial_id(
        mu_type="1U",
        data_train=data_train,
        treatment_val=1,
        S_vars=S_vars,
        X_vars=X_vars,
        Y_var=Y_var,
        type=type,
        type_prop=type_prop,
        prop_lb=prop_lb,
        prop_ub=prop_ub,
        nuisance_cv_fold=nuisance_cv_fold,
        grf_honesty=grf_honesty,
        grf_tune_parameters=grf_tune_parameters,
        grf_num_threads=grf_num_threads,
        xgb_cv_rounds=xgb_cv_rounds,
        xgb_eta=xgb_eta,
        xgb_max_depth=xgb_max_depth,
        xgb_threads=xgb_threads,
    )
    bar_mu_1L_x_out = bar_mu_x_partial_id(
        mu_type="1L",
        data_train=data_train,
        treatment_val=1,
        S_vars=S_vars,
        X_vars=X_vars,
        Y_var=Y_var,
        type=type,
        type_prop=type_prop,
        prop_lb=prop_lb,
        prop_ub=prop_ub,
        nuisance_cv_fold=nuisance_cv_fold,
        grf_honesty=grf_honesty,
        grf_tune_parameters=grf_tune_parameters,
        grf_num_threads=grf_num_threads,
        xgb_cv_rounds=xgb_cv_rounds,
        xgb_eta=xgb_eta,
        xgb_max_depth=xgb_max_depth,
        xgb_threads=xgb_threads,
    )
    bar_mu_0U_x_out = bar_mu_x_partial_id(
        mu_type="0U",
        data_train=data_train,
        treatment_val=0,
        S_vars=S_vars,
        X_vars=X_vars,
        Y_var=Y_var,
        type=type,
        type_prop=type_prop,
        prop_lb=prop_lb,
        prop_ub=prop_ub,
        nuisance_cv_fold=nuisance_cv_fold,
        grf_honesty=grf_honesty,
        grf_tune_parameters=grf_tune_parameters,
        grf_num_threads=grf_num_threads,
        xgb_cv_rounds=xgb_cv_rounds,
        xgb_eta=xgb_eta,
        xgb_max_depth=xgb_max_depth,
        xgb_threads=xgb_threads,
    )
    bar_mu_0L_x_out = bar_mu_x_partial_id(
        mu_type="0L",
        data_train=data_train,
        treatment_val=0,
        S_vars=S_vars,
        X_vars=X_vars,
        Y_var=Y_var,
        type=type,
        type_prop=type_prop,
        prop_lb=prop_lb,
        prop_ub=prop_ub,
        nuisance_cv_fold=nuisance_cv_fold,
        grf_honesty=grf_honesty,
        grf_tune_parameters=grf_tune_parameters,
        grf_num_threads=grf_num_threads,
        xgb_cv_rounds=xgb_cv_rounds,
        xgb_eta=xgb_eta,
        xgb_max_depth=xgb_max_depth,
        xgb_threads=xgb_threads,
    )

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

    return {
        "bar_mu_1U_x": bar_mu_1U_x_out,
        "bar_mu_1L_x": bar_mu_1L_x_out,
        "bar_mu_0U_x": bar_mu_0U_x_out,
        "bar_mu_0L_x": bar_mu_0L_x_out,
        "varrho_x": varrho_x_out,
        "varrho_s_x": varrho_s_x_out,
        "phi_x": phi_x_out,
        "phi_s_x": phi_s_x_out,
    }


def compute_tau_partial_id(data_nuisance, Y_var, tau_type):
    m = data_nuisance.copy()
    m["Y"] = m[Y_var]

    if tau_type == "upper":
        m["m_0_a"] = ((1 - m["observe"]) / m["phi"]) * (
            (m["treatment"] / m["varrho_x"]) * (m["mu_1U_s_x"] - m["bar_mu_1U_x"])
            - ((1 - m["treatment"]) / (1 - m["varrho_x"])) * (m["mu_0L_s_x"] - m["bar_mu_0L_x"])
        )
        m["m_0_b"] = ((1 - m["observe"]) / m["phi"]) * (
            m["bar_mu_1U_x"] - m["bar_mu_0L_x"]
        )
        m["m_0_c"] = (m["observe"] / m["phi"]) * (m["phi_s_x"] / (1 - m["phi_s_x"])) * (
            (m["varrho_s_x"] / m["varrho_x"]) *
            (H_y_s("U", m["Y"], m["q_U_s_x"], m["varrho_s_x"]) - m["mu_1U_s_x"])
            - ((1 - m["varrho_s_x"]) / (1 - m["varrho_x"])) *
            (H_y_s("L", m["Y"], m["q_U_s_x"], (1 - m["varrho_s_x"])) - m["mu_0L_s_x"])
        )
        m["m_0_d"] = ((1 - m["observe"]) / m["phi"]) * (1 / m["varrho_x"]) * (
            (m["q_U_s_x"] - m["mu_1U_s_x"]) * (m["treatment"] - m["varrho_s_x"])
        )
        m["m_0_e"] = ((1 - m["observe"]) / m["phi"]) * (1 / (1 - m["varrho_x"])) * (
            (m["q_U_s_x"] - m["mu_0L_s_x"]) * (m["treatment"] - m["varrho_s_x"])
        )
    else:
        m["m_0_a"] = ((1 - m["observe"]) / m["phi"]) * (
            (m["treatment"] / m["varrho_x"]) * (m["mu_1L_s_x"] - m["bar_mu_1L_x"])
            - ((1 - m["treatment"]) / (1 - m["varrho_x"])) * (m["mu_0U_s_x"] - m["bar_mu_0U_x"])
        )
        m["m_0_b"] = ((1 - m["observe"]) / m["phi"]) * (
            m["bar_mu_1L_x"] - m["bar_mu_0U_x"]
        )
        m["m_0_c"] = (m["observe"] / m["phi"]) * (m["phi_s_x"] / (1 - m["phi_s_x"])) * (
            (m["varrho_s_x"] / m["varrho_x"]) *
            (H_y_s("L", m["Y"], m["q_L_s_x"], m["varrho_s_x"]) - m["mu_1L_s_x"])
            - ((1 - m["varrho_s_x"]) / (1 - m["varrho_x"])) *
            (H_y_s("U", m["Y"], m["q_L_s_x"], (1 - m["varrho_s_x"])) - m["mu_0U_s_x"])
        )
        m["m_0_d"] = ((1 - m["observe"]) / m["phi"]) * (1 / m["varrho_x"]) * (
            (m["q_L_s_x"] - m["mu_1L_s_x"]) * (m["treatment"] - m["varrho_s_x"])
        )
        m["m_0_e"] = ((1 - m["observe"]) / m["phi"]) * (1 / (1 - m["varrho_x"])) * (
            (m["q_L_s_x"] - m["mu_0U_s_x"]) * (m["treatment"] - m["varrho_s_x"])
        )

    m["m_0"] = m["m_0_a"] + m["m_0_b"] + m["m_0_c"] + m["m_0_d"] + m["m_0_e"]

    multiplier = float(np.mean(1 - data_nuisance["observe"]) / data_nuisance["phi"].iloc[0])
    return float(np.mean(m["m_0"]) / multiplier)


def compute_m_table_partial_id(tau_0, data_nuisance, Y_var, tau_type):
    m = data_nuisance.copy()
    m["Y"] = m[Y_var]

    if tau_type == "upper":
        m["m_0_a"] = ((1 - m["observe"]) / m["phi"]) * (
            (m["treatment"] / m["varrho_x"]) * (m["mu_1U_s_x"] - m["bar_mu_1U_x"])
            - ((1 - m["treatment"]) / (1 - m["varrho_x"])) * (m["mu_0L_s_x"] - m["bar_mu_0L_x"])
        )
        m["m_0_b"] = ((1 - m["observe"]) / m["phi"]) * (
            m["bar_mu_1U_x"] - m["bar_mu_0L_x"] - tau_0
        )
        m["m_0_c"] = (m["observe"] / m["phi"]) * (m["phi_s_x"] / (1 - m["phi_s_x"])) * (
            (m["varrho_s_x"] / m["varrho_x"]) *
            (H_y_s("U", m["Y"], m["q_U_s_x"], m["varrho_s_x"]) - m["mu_1U_s_x"])
            - ((1 - m["varrho_s_x"]) / (1 - m["varrho_x"])) *
            (H_y_s("L", m["Y"], m["q_U_s_x"], (1 - m["varrho_s_x"])) - m["mu_0L_s_x"])
        )
        m["m_0_d"] = ((1 - m["observe"]) / m["phi"]) * (1 / m["varrho_x"]) * (
            (m["q_U_s_x"] - m["mu_1U_s_x"]) * (m["treatment"] - m["varrho_s_x"])
        )
        m["m_0_e"] = ((1 - m["observe"]) / m["phi"]) * (1 / (1 - m["varrho_x"])) * (
            (m["q_U_s_x"] - m["mu_0L_s_x"]) * (m["treatment"] - m["varrho_s_x"])
        )
    else:
        m["m_0_a"] = ((1 - m["observe"]) / m["phi"]) * (
            (m["treatment"] / m["varrho_x"]) * (m["mu_1L_s_x"] - m["bar_mu_1L_x"])
            - ((1 - m["treatment"]) / (1 - m["varrho_x"])) * (m["mu_0U_s_x"] - m["bar_mu_0U_x"])
        )
        m["m_0_b"] = ((1 - m["observe"]) / m["phi"]) * (
            m["bar_mu_1L_x"] - m["bar_mu_0U_x"] - tau_0
        )
        m["m_0_c"] = (m["observe"] / m["phi"]) * (m["phi_s_x"] / (1 - m["phi_s_x"])) * (
            (m["varrho_s_x"] / m["varrho_x"]) *
            (H_y_s("L", m["Y"], m["q_L_s_x"], m["varrho_s_x"]) - m["mu_1L_s_x"])
            - ((1 - m["varrho_s_x"]) / (1 - m["varrho_x"])) *
            (H_y_s("U", m["Y"], m["q_L_s_x"], (1 - m["varrho_s_x"])) - m["mu_0U_s_x"])
        )
        m["m_0_d"] = ((1 - m["observe"]) / m["phi"]) * (1 / m["varrho_x"]) * (
            (m["q_L_s_x"] - m["mu_1L_s_x"]) * (m["treatment"] - m["varrho_s_x"])
        )
        m["m_0_e"] = ((1 - m["observe"]) / m["phi"]) * (1 / (1 - m["varrho_x"])) * (
            (m["q_L_s_x"] - m["mu_0U_s_x"]) * (m["treatment"] - m["varrho_s_x"])
        )

    m["m_0"] = m["m_0_a"] + m["m_0_b"] + m["m_0_c"] + m["m_0_d"] + m["m_0_e"]

    return m


def compute_bound_ci_partial_id(hat_tau, data_nuisance, Y_var, tau_type, alpha):
    influence = compute_m_table_partial_id(
        hat_tau, data_nuisance, Y_var, tau_type
    )["m_0"].to_numpy(dtype=float)

    hat_V = float(np.var(influence, ddof=1))
    z = float(norm.ppf(1 - alpha / 2))
    se = float(np.sqrt(hat_V / len(influence)))
    ci_l = hat_tau - z * se
    ci_u = hat_tau + z * se

    return {
        "se": se,
        "ci": np.array([ci_l, ci_u], dtype=float),
    }


def _predict_regression_model(model, new_data, nuisance_type):
    if nuisance_type == "glmnet":
        x_mat = _prepare_glmnet_matrix(new_data)
        pred = np.asarray(model.predict(x_mat, lamb=[model.lambda_max_])).reshape(-1)
        return pred

    if nuisance_type == "grf":
        pred = np.asarray(model.predict(_as_2d_numpy(new_data))).reshape(-1)
        return pred

    if nuisance_type == "xgboost":
        if isinstance(new_data, pd.DataFrame):
            dnew = xgb.DMatrix(_as_2d_numpy(new_data), feature_names=list(new_data.columns))
        else:
            dnew = xgb.DMatrix(_as_2d_numpy(new_data))
        pred = np.asarray(model.predict(dnew)).reshape(-1)
        return pred

    raise ValueError("Enter a valid nuisance parameter estimation type")