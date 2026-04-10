from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from econml.grf import RegressionForest
from quantile_forest import RandomForestQuantileRegressor
from glmnet import ElasticNet as GlmnetElasticNet
_rng = np.random.default_rng(42)

from .nuisance_shared import (
    varrho_s_x,
    _as_2d_numpy,
    _prepare_glmnet_matrix,
    _flatten_row,
    _build_new_data_row,
    _predict_propensity,
    _predict_propensity_batch,
    _balanced_random_folds,
)


def cond_AVaR(eta, S, X, neg_sign=False, data_train=None, S_vars=None, X_vars=None, Y_var=None):
    df = data_train.loc[data_train["observe"] == 1, :].copy()

    y_raw = df.loc[:, Y_var].to_numpy(dtype=float)
    if neg_sign:
        y = -y_raw
    else:
        y = y_raw

    # Stage 1: estimate conditional eta-quantile using quantile forest
    s_x = df.loc[:, list(S_vars) + list(X_vars)]

    q_forest = RandomForestQuantileRegressor(
        # n_estimators=500,
        max_samples=0.5,
        max_features="sqrt",
        bootstrap=True,
        min_samples_leaf=5,
        random_state=_rng.integers(2**31),
    )
    q_forest.fit(_as_2d_numpy(s_x), y)

    rq_pred = q_forest.predict(_as_2d_numpy(s_x), quantiles=[float(eta)], oob_score=True)
    rq_pred = np.asarray(rq_pred)
    if rq_pred.ndim == 2:
        rq_approx = rq_pred[:, 0]
    else:
        rq_approx = rq_pred.reshape(-1)

    # Stage 2: sieve with generated outcome variable
    psi = (1.0 / (1.0 - eta)) * (
        y * (y >= rq_approx) - rq_approx * (y >= (rq_approx - (1.0 - eta)))
    )

    sieve_model = _sieve_sgd_preprocess(X=_as_2d_numpy(s_x), basis_type="cosine")
    sieve_model = _sieve_sgd_solver(sieve_model=sieve_model, X=_as_2d_numpy(s_x), Y=psi)
    new_data = _build_new_data_row(S, X, S_vars, X_vars)
    est = _sieve_sgd_predict(sieve_model, X=_as_2d_numpy(new_data))

    return float(np.asarray(est).reshape(-1)[0])


def mu_s_x_partial_id(
    mu_type,
    S,
    X,
    varrho_s_x_out,
    eta=None,
    data_train=None,
    S_vars=None,
    X_vars=None,
    Y_var=None,
    type_prop=None,
    nuisance_cv_fold=None,
    prop_ub=None,
    prop_lb=None,
):
    if mu_type in ("1U", "0U"):
        neg_sign = False
    else:
        neg_sign = True

    if varrho_s_x_out is not None:
        new_data = _build_new_data_row(S, X, S_vars, X_vars)

        rho_hat = _predict_propensity(
            varrho_s_x_out=varrho_s_x_out,
            new_data=new_data,
            type_prop=type_prop,
            prop_lb=prop_lb,
            prop_ub=prop_ub,
        )

        if mu_type in ("1U", "1L"):
            eta = 1.0 - rho_hat
        else:
            eta = rho_hat
    else:
        # Leave-one-out within the training data for estimating E[H | S_i, X_i]
        s_target = _flatten_row(S)
        x_target = _flatten_row(X)

        def _row_matches(row):
            s_match = np.all(row[list(S_vars)].to_numpy() == s_target)
            x_match = np.all(row[list(X_vars)].to_numpy() == x_target)
            return s_match and x_match

        row_keep = ~data_train.apply(_row_matches, axis=1)
        data_train = data_train.loc[row_keep, :].copy()

    return cond_AVaR(
        float(eta),
        _flatten_row(S),
        _flatten_row(X),
        neg_sign=neg_sign,
        data_train=data_train,
        S_vars=S_vars,
        X_vars=X_vars,
        Y_var=Y_var,
    )


def bar_mu_x_partial_id(
    mu_type,
    data_train,
    treatment_val,
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
    # Cross-fitting within the training data
    df = data_train.loc[data_train["observe"] == 0, :].copy()
    df_0 = df.loc[df["treatment"] == 0, :].copy()
    df_1 = df.loc[df["treatment"] == 1, :].copy()

    if len(df_0) < nuisance_cv_fold or len(df_1) < nuisance_cv_fold:
        warnings.warn(
            "Some treatment arms in the observational sample have fewer observations than "
            "`nuisance_cv_fold`; cross-fitting may be unstable."
        )

    df_0["fold"] = _balanced_random_folds(len(df_0), nuisance_cv_fold)
    df_1["fold"] = _balanced_random_folds(len(df_1), nuisance_cv_fold)
    df = pd.concat([df_0, df_1], axis=0, ignore_index=True)

    etas = np.full(len(df), np.nan)

    for fold_id in range(1, nuisance_cv_fold + 1):
        test_idx = np.where(df["fold"].to_numpy() == fold_id)[0]
        train_idx = np.setdiff1d(np.arange(len(df)), test_idx)

        test_s_x = df.iloc[test_idx][list(S_vars) + list(X_vars)]
        train_s_x_y = df.iloc[train_idx].copy()

        varrho_s_x_out = varrho_s_x(
            train_s_x_y,
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

        rho_hat = _predict_propensity_batch(
            varrho_s_x_out=varrho_s_x_out,
            new_data=test_s_x,
            type_prop=type_prop,
            prop_lb=prop_lb,
            prop_ub=prop_ub,
        )

        if mu_type in ("1U", "1L"):
            etas[test_idx] = 1.0 - rho_hat
        elif mu_type in ("0U", "0L"):
            etas[test_idx] = rho_hat

    df["eta"] = etas
    df = df.loc[df["treatment"] == treatment_val, :].copy()

    s = df.loc[:, list(S_vars)]
    x = df.loc[:, list(X_vars)]
    etas = df["eta"].to_numpy()

    mu_x_pred = np.array([
        mu_s_x_partial_id(
            mu_type=mu_type,
            S=s.iloc[i, :].to_numpy(),
            X=x.iloc[i, :].to_numpy(),
            eta=etas[i],
            varrho_s_x_out=None,
            data_train=data_train,
            S_vars=S_vars,
            X_vars=X_vars,
            Y_var=Y_var,
            type_prop=type_prop,
            nuisance_cv_fold=nuisance_cv_fold,
            prop_ub=prop_ub,
            prop_lb=prop_lb,
        )
        for i in range(len(df))
    ])

    if type == "glmnet":
        x_mat = _prepare_glmnet_matrix(x)
        bar_mu_x_model = GlmnetElasticNet(
            alpha=1,              # lasso
            n_lambda=100,
            n_splits=nuisance_cv_fold,
            scoring="mean_squared_error",
            n_jobs=1,
            random_state=int(_rng.integers(2**31)),
        )
        bar_mu_x_model.fit(x_mat, np.asarray(mu_x_pred, dtype=float))

    elif type == "grf":
        bar_mu_x_model = RegressionForest(
            n_estimators=1000,
            honest=grf_honesty,
            inference=False,
            max_samples=0.5,            # GRF subsamples ~50% without replacement
            min_samples_leaf=5,         # closer to GRF default
            max_features="sqrt",
            n_jobs=grf_num_threads,
            random_state=_rng.integers(2**31),
        )
        bar_mu_x_model.fit(_as_2d_numpy(x), np.asarray(mu_x_pred, dtype=float))

    elif type == "xgboost":
        x_mat = _as_2d_numpy(x).astype(float)
        mu_vec = np.asarray(mu_x_pred, dtype=float)

        dtrain = xgb.DMatrix(x_mat, label=mu_vec, feature_names=list(x.columns))

        cv_fit = xgb.cv(
            params={
                "objective": "reg:squarederror",
                "max_depth": xgb_max_depth,
                "eta": xgb_eta,
                "nthread": xgb_threads,
            },
            dtrain=dtrain,
            num_boost_round=xgb_cv_rounds,
            nfold=nuisance_cv_fold,
            verbose_eval=False,
            seed=int(_rng.integers(2**31)),
        )

        if "test-rmse-mean" in cv_fit.columns:
            best_nrounds = int(np.argmin(cv_fit["test-rmse-mean"].to_numpy()) + 1)
        else:
            best_nrounds = xgb_cv_rounds

        if not np.isfinite(best_nrounds) or best_nrounds < 1:
            best_nrounds = xgb_cv_rounds

        bar_mu_x_model = xgb.train(
            params={
                "objective": "reg:squarederror",
                "max_depth": xgb_max_depth,
                "eta": xgb_eta,
                "nthread": xgb_threads,
            },
            dtrain=dtrain,
            num_boost_round=best_nrounds,
            verbose_eval=False,
        )

    else:
        raise ValueError("Enter a valid nuisance parameter estimation type")

    return bar_mu_x_model


def q_s_x(
    q_type,
    data_train,
    S_vars,
    X_vars,
    Y_var,
    type_prop,
    prop_ub,
    prop_lb,
    S,
    X,
    varrho_s_x_out,
):
    # predict rho(S,X) using the observational sample
    df = data_train.loc[data_train["observe"] == 1, :].copy()
    s_x = df.loc[:, list(S_vars) + list(X_vars)]
    y = df.loc[:, Y_var].to_numpy(dtype=float)

    new_data = _build_new_data_row(S, X, S_vars, X_vars)
    rho_hat = _predict_propensity(
        varrho_s_x_out=varrho_s_x_out,
        new_data=new_data,
        type_prop=type_prop,
        prop_lb=prop_lb,
        prop_ub=prop_ub,
    )

    if q_type == "U":
        eta = float(1.0 - rho_hat)
    else:
        eta = float(rho_hat)

    q_forest = RandomForestQuantileRegressor(
        n_estimators=1000,
        max_samples=0.5,
        max_features="sqrt",
        bootstrap=True,
        min_samples_leaf=5,
        random_state=_rng.integers(2**31),
    )
    q_forest.fit(_as_2d_numpy(s_x), y)

    pred = q_forest.predict(_as_2d_numpy(new_data), quantiles=[eta])
    pred = np.asarray(pred)

    if pred.ndim == 2:
        return float(pred[0, 0])
    return float(pred.reshape(-1)[0])


def H_y_s(H_type, Y, S, eta):
    if H_type == "U":
        return S + (1.0 / eta) * ((Y > S) * (Y - S))
    else:
        return S - (1.0 / eta) * ((S > Y) * (S - Y))
    

def _sieve_sgd_preprocess(X, basis_type="cosine", n_basis=5):
    X = _as_2d_numpy(X).astype(float)

    if basis_type != "cosine":
        raise ValueError("Currently only `basis_type='cosine'` is supported.")

    x_min = np.min(X, axis=0)
    x_max = np.max(X, axis=0)

    scale = x_max - x_min
    scale[scale <= 1e-12] = 1.0

    return {
        "basis_type": basis_type,
        "n_basis": int(n_basis),
        "x_min": x_min,
        "scale": scale,
        "model": None,
    }


import math
from itertools import combinations, permutations
from typing import Dict, List, Optional, Tuple


def _normalize_X_quantile(
    X: np.ndarray,
    norm_para: Optional[np.ndarray] = None,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
) -> Dict[str, np.ndarray]:
    """
    Python translation of R normalize_X.

    Parameters
    ----------
    X : array-like, shape (n, p)
    norm_para : None or ndarray of shape (2, p)
        Row 0 = lower values, row 1 = upper values.
    lower_q, upper_q : float
        Quantiles used when norm_para is None.

    Returns
    -------
    dict with keys:
        - "X": normalized/clipped array in [0, 1]
        - "norm_para": ndarray shape (2, p)
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    Xn = X.copy()

    if norm_para is None:
        lowers = np.quantile(Xn, lower_q, axis=0)
        uppers = np.quantile(Xn, upper_q, axis=0)

        denom = uppers - lowers
        denom = np.where(np.abs(denom) < 1e-12, 1.0, denom)

        Xn = (Xn - lowers) / denom
        Xn = np.clip(Xn, 0.0, 1.0)

        norm_para = np.vstack([lowers, uppers])
    else:
        norm_para = np.asarray(norm_para, dtype=float)
        lowers = norm_para[0, :]
        uppers = norm_para[1, :]

        denom = uppers - lowers
        denom = np.where(np.abs(denom) < 1e-12, 1.0, denom)

        Xn = (Xn - lowers) / denom
        Xn = np.clip(Xn, 0.0, 1.0)

    return {"X": Xn, "norm_para": norm_para}


def _generate_factors(n: int, dimlimit: int) -> List[List[int]]:
    """
    Python translation of Rcpp Generate_factors(n, dimlimit).

    Returns factor combinations of n using integers >= 2, with nondecreasing
    order inside each combination, and length <= dimlimit.
    """
    results: List[List[int]] = []

    def rec(first: int, each_prod: int, curr: List[int]) -> None:
        if first > n or each_prod > n:
            return
        if each_prod == n:
            if len(curr) <= dimlimit:
                results.append(curr.copy())
            return

        for i in range(first, n):
            if i * each_prod > n:
                break
            if n % i == 0:
                curr.append(i)
                rec(i, i * each_prod, curr)
                curr.pop()

    rec(2, 1, [])
    return results


def _sjt_permutations(seq):
    """
    Steinhaus-Johnson-Trotter permutation generation.
    Matches the ordering produced by R's combinat::permn().
    """
    seq = list(seq)
    n = len(seq)
    if n == 0:
        return [()]
    if n == 1:
        return [tuple(seq)]

    # Recursively generate permutations of seq[:-1]
    last = seq[-1]
    sub_perms = _sjt_permutations(seq[:-1])
    result = []

    for i, perm in enumerate(sub_perms):
        perm = list(perm)
        # Even-indexed sub-permutations: insert right to left
        # Odd-indexed sub-permutations: insert left to right
        if i % 2 == 0:
            positions = range(len(perm), -1, -1)
        else:
            positions = range(0, len(perm) + 1)
        for pos in positions:
            new_perm = perm[:pos] + [last] + perm[pos:]
            result.append(tuple(new_perm))

    return result


def _unique_permutations(seq):
    """
    Unique permutations in the same order as R's unique(combinat::permn(seq)).

    R's combinat::permn uses Steinhaus-Johnson-Trotter ordering, and
    unique() preserves first-occurrence order. We replicate both.
    """
    # Sort the input so that the SJT recursion starts from a consistent base,
    # matching R's combinat::permn which operates on the elements as given.
    all_perms = _sjt_permutations(list(seq))

    # Deduplicate preserving first-occurrence order (same as R's unique())
    seen = set()
    result = []
    for p in all_perms:
        if p not in seen:
            seen.add(p)
            result.append(p)
    return result



def _create_index_matrix(
    xdim: int,
    basisN: Optional[int] = None,
    maxj: Optional[int] = None,
    interaction_order: int = 5,
) -> np.ndarray:
    """
    Python translation of R create_index_matrix.

    Returns matrix with first column = row product,
    remaining columns = per-dimension basis indices.
    """
    if (basisN is None) == (maxj is None):
        raise ValueError("Exactly one of `basisN` or `maxj` must be provided.")

    interaction_order = min(interaction_order, xdim)

    rows: List[List[int]] = [([1] * xdim)]

    def append_rows_for_product(product_v: int) -> None:
        nonlocal rows

        factors_v = _generate_factors(product_v, interaction_order)
        greater_than_1: List[Tuple[int, ...]] = []

        if len(factors_v) > 0:
            for fac in factors_v:
                greater_than_1.extend(_unique_permutations(fac))

        # add trivial factorization: just [product_v]
        greater_than_1.append((product_v,))

        for fac_tuple in greater_than_1:
            m = len(fac_tuple)
            for pos in combinations(range(xdim), m):
                row = [1] * xdim
                for idx, p in enumerate(pos):
                    row[p] = fac_tuple[idx]
                rows.append(row)

    if maxj is not None:
        for product_v in range(2, maxj + 1):
            append_rows_for_product(product_v)
    else:
        product_v = 2
        while len(rows) < basisN:
            append_rows_for_product(product_v)
            product_v += 1
        rows = rows[:basisN]

    index_core = np.asarray(rows, dtype=int)
    row_prod = np.prod(index_core, axis=1).reshape(-1, 1)
    index_matrix = np.hstack([row_prod, index_core])

    return index_matrix


def _design_M_cosine(
    X: np.ndarray,
    basisN: int,
    index_matrix: np.ndarray,
) -> np.ndarray:
    """
    Python translation of Design_M_C(..., type='cosine', ...).

    Parameters
    ----------
    X : ndarray shape (n, p), assumed already normalized to [0, 1]
    basisN : int
    index_matrix : ndarray whose first col is row product and remaining cols
        are basis instructions by dimension

    Returns
    -------
    Phi : ndarray shape (n, basisN)
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    n, p = X.shape
    instr = np.asarray(index_matrix[:, 1:], dtype=int)
    instr = instr[:basisN, :]
    Phi = np.ones((n, basisN), dtype=float)

    for j in range(basisN):
        for k in range(p):
            idx = instr[j, k]
            if idx > 1:
                Phi[:, j] *= np.cos((idx - 1) * math.pi * X[:, k])

    return Phi


def _clean_up_sieve_result(sieve_model: Dict) -> Dict:
    """
    Minimal Python analogue of clean_up_result.
    Chooses best model by rolling CV and stores it under `best_model`.
    """
    rolling_cvs = [inf["rolling_cv"] for inf in sieve_model["inf_list"]]
    best_idx = int(np.argmin(rolling_cvs))
    sieve_model["best_model_index"] = best_idx
    sieve_model["best_model"] = sieve_model["inf_list"][best_idx]
    return sieve_model


def _sieve_sgd_preprocess(
    X,
    s: Tuple[float, ...] = (2.0,),
    r0: Tuple[float, ...] = (2.0,),
    J: Tuple[float, ...] = (1.0,),
    basis_type: str = "cosine",
    interaction_order: Tuple[int, ...] = (3,),
    omega: Tuple[float, ...] = (0.51,),
    norm_feature: bool = True,
    norm_para: Optional[np.ndarray] = None,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
) -> Dict:
    """
    Minimal faithful version of R sieve.sgd.preprocess.

    For now we allow tuples but will usually use only the default single value
    in each hyperparameter dimension.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    if norm_feature:
        norm_list = _normalize_X_quantile(
            X, norm_para=norm_para, lower_q=lower_q, upper_q=upper_q
        )
        norm_para = norm_list["norm_para"]

    s = tuple(s)
    r0 = tuple(r0)
    J = tuple(J)
    interaction_order = tuple(interaction_order)
    omega = tuple(omega)

    hyper_grid = []
    for i_s in range(len(s)):
        for i_r0 in range(len(r0)):
            for i_J in range(len(J)):
                for i_io in range(len(interaction_order)):
                    for i_om in range(len(omega)):
                        hyper_grid.append(
                            {
                                "s_idx": i_s,
                                "r0_idx": i_r0,
                                "J_idx": i_J,
                                "interaction_order_idx": i_io,
                                "omega_idx": i_om,
                            }
                        )

    min_s = min(s)
    max_J = max(J)
    max_interaction = max(interaction_order)
    s_size = X.shape[0]
    max_basisN = math.ceil(max_J * (s_size ** (1.0 / (2.0 * min_s + 1.0))))

    index_matrix_full = _create_index_matrix(
        xdim=X.shape[1],
        basisN=max_basisN,
        interaction_order=max_interaction,
    )
    index_row_prod = index_matrix_full[:, 0].astype(float)
    index_core = index_matrix_full[:, 1:].astype(int)

    inf_list = []
    for grid_entry in hyper_grid:
        inf_list.append(
            {
                "hyper_para_index": grid_entry,
                "rolling_cv": 0.0,
                "beta_f_int": np.zeros(1, dtype=float),
                "beta_f": np.zeros(1, dtype=float),
            }
        )

    return {
        "s_size_sofar": 0,
        "type": basis_type,
        "hyper_para_list": {
            "s": s,
            "r0": r0,
            "J": J,
            "interaction_order": interaction_order,
            "omega": omega,
        },
        "index_matrix": index_core,
        "index_row_prod": index_row_prod,
        "inf_list": inf_list,
        "norm_para": norm_para,
    }


def _sieve_sgd_solver(
    sieve_model: Dict,
    X,
    Y,
    cv_weight_rate: float = 1.0,
) -> Dict:
    """
    Much closer Python translation of R sieve.sgd.solver.

    This keeps the online SGD logic and rolling CV.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float).reshape(-1)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    norm_para = sieve_model["norm_para"]
    if norm_para is None:
        raise ValueError("No normalization parameters found in sieve model.")

    norm_list = _normalize_X_quantile(X, norm_para=norm_para)
    Xn = norm_list["X"]

    s_list = sieve_model["hyper_para_list"]["s"]
    r0_list = sieve_model["hyper_para_list"]["r0"]
    J_list = sieve_model["hyper_para_list"]["J"]
    omega_list = sieve_model["hyper_para_list"]["omega"]
    index_matrix = sieve_model["index_matrix"]
    index_row_prod = sieve_model["index_row_prod"]
    basis_type = sieve_model["type"]
    inf_list = sieve_model["inf_list"]
    s_size_sofar = sieve_model["s_size_sofar"]

    if basis_type != "cosine":
        raise NotImplementedError("This first faithful version only supports cosine.")

    for i in range(Xn.shape[0]):
        i_sofar = i + 1 + s_size_sofar
        newx = Xn[i : i + 1, :]
        newy = float(Y[i])

        # regenerate basis count as in R
        min_s = min(s_list)
        max_J_coeff = max(J_list)
        max_J_now = math.ceil(max_J_coeff * (i_sofar ** (1.0 / (2.0 * min_s + 1.0))))

        Phi = _design_M_cosine(newx, basisN=max_J_now, index_matrix=np.column_stack([index_row_prod, index_matrix]))

        for m, info in enumerate(inf_list):
            hp = info["hyper_para_index"]

            s_val = s_list[hp["s_idx"]]
            r0_val = r0_list[hp["r0_idx"]]
            J_coeff = J_list[hp["J_idx"]]
            omega_val = omega_list[hp["omega_idx"]]

            J_m = math.ceil(J_coeff * (i_sofar ** (1.0 / (2.0 * s_val + 1.0))))

            beta_f = info["beta_f"]
            beta_f_int = info["beta_f_int"]

            if len(beta_f) < J_m:
                beta_f = np.concatenate([beta_f, np.zeros(J_m - len(beta_f))])
            if len(beta_f_int) < J_m:
                beta_f_int = np.concatenate([beta_f_int, np.zeros(J_m - len(beta_f_int))])

            phi_m = Phi[0, :J_m]

            fnewx = float(np.dot(beta_f[:J_m], phi_m))
            info["rolling_cv"] += (i_sofar ** cv_weight_rate) * ((newy - fnewx) ** 2)

            rn_m = r0_val * (i_sofar ** (-1.0 / (2.0 * s_val + 1.0)))

            fnewx_int = float(np.dot(beta_f_int[:J_m], phi_m))
            res = newy - fnewx_int

            weight_vec = (index_row_prod[:J_m]) ** (-2.0 * omega_val)
            beta_f_int[:J_m] = beta_f_int[:J_m] + rn_m * res * weight_vec * phi_m

            beta_f[:J_m] = ((i_sofar - 1.0) / i_sofar) * beta_f[:J_m] + beta_f_int[:J_m] / i_sofar

            info["beta_f"] = beta_f
            info["beta_f_int"] = beta_f_int

    sieve_model["s_size_sofar"] = s_size_sofar + Xn.shape[0]
    sieve_model["inf_list"] = inf_list
    sieve_model = _clean_up_sieve_result(sieve_model)
    return sieve_model


def _sieve_sgd_predict(
    sieve_model: Dict,
    X,
) -> np.ndarray:
    """
    Python translation of sieve.sgd.predict using best_model.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    norm_para = sieve_model["norm_para"]
    if norm_para is None:
        raise ValueError("No normalization parameters found in sieve model.")

    norm_list = _normalize_X_quantile(X, norm_para=norm_para)
    Xn = norm_list["X"]

    best_model = sieve_model["best_model"]
    beta = np.asarray(best_model["beta_f"], dtype=float)
    J_m = len(beta)

    Phi = _design_M_cosine(
        Xn,
        basisN=J_m,
        index_matrix=np.column_stack([sieve_model["index_row_prod"], sieve_model["index_matrix"]]),
    )

    pred = Phi @ beta
    return np.asarray(pred, dtype=float)