from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from glmnet import ElasticNet as GlmnetElasticNet
from sklearn.linear_model import LinearRegression
from econml.grf import RegressionForest
from quantile_forest import RandomForestQuantileRegressor
import sensitivity_surrogacy.nuisance_copula as _nc

from .nuisance_shared import (
    varrho_s_x,
    _as_2d_numpy,
    _flatten_row,
    _build_new_data_row,
    _predict_propensity,
    _predict_propensity_batch,
    _balanced_random_folds,
)
_rng = np.random.default_rng(42)


def cond_quantile_forest(u, S, X, data_train, S_vars, X_vars, Y_var):
    df = data_train.loc[data_train["observe"] == 1, :]
    s_x = df.loc[:, list(S_vars) + list(X_vars)]
    y = np.asarray(df[Y_var]).reshape(-1)

    rq_model = RandomForestQuantileRegressor(
        n_estimators=1000,
        max_samples=0.5,
        max_features="sqrt",
        bootstrap=True,
        min_samples_leaf=5,
        random_state=_rng.integers(2**31),
    )
    rq_model.fit(_as_2d_numpy(s_x), y)

    new_values = np.concatenate([_flatten_row(S), _flatten_row(X)])
    new_data = pd.DataFrame([new_values], columns=list(S_vars) + list(X_vars))

    preds = rq_model.predict(_as_2d_numpy(new_data), quantiles=list(np.asarray(u, dtype=float)))
    return np.asarray(preds).reshape(-1)


def approx_integral(y_vals, upperLimit, lowerLimit):
    n_points = len(y_vals)
    h = (upperLimit - lowerLimit) / (n_points - 1)
    integral = (h / 2.0) * (
        y_vals[0] + 2.0 * np.sum(y_vals[1:(n_points - 1)]) + y_vals[n_points - 1]
    )
    return integral


def h_s_x_y(
    W,
    S,
    X,
    Y,
    varrho_s_x_out,
    data_train,
    S_vars,
    X_vars,
    Y_var,
    type_prop,
    nuisance_cv_fold,
    prop_ub,
    prop_lb,
    integralMaxEval,
    cop,
    param,
):

    if pd.isna(Y):
        return 0.0

    if cop == "Frank":
        def g_fn(u):
            return np.exp(-u * param) - 1

        def cond_copula(u, v):
            numerator = g_fn(u) * g_fn(v) + g_fn(v)
            denominator = g_fn(u) * g_fn(v) + g_fn(1)
            cond_cdf = numerator / denominator
            return cond_cdf

        def d_sigma(w, u, v):
            A = g_fn(u) * g_fn(v) + g_fn(v)
            B = g_fn(u) * g_fn(v) + g_fn(1)
            dA_du = (-param * np.exp(-u * param)) * g_fn(v)
            dB_du = (-param * np.exp(-u * param)) * g_fn(v)
            numerator = B * dA_du - A * dB_du
            denominator = B ** 2
            return ((-1.0) / (w - v)) * numerator / denominator

    elif cop == "Plackett":
        def cond_copula(u, v):
            numerator = -1 + u + v - u * param + v * param
            inner_term = (1 + (param - 1) * (u + v)) ** 2 - 4 * param * (param - 1) * u * v
            denominator = 2 * np.sqrt(inner_term)
            return 0.5 + numerator / denominator

        def d_sigma(w, u, v):
            A = u * (1 - param) + v * (1 + param) - 1
            R = 1 + (param - 1) * (u + v)
            D = R ** 2 - 4 * param * (param - 1) * u * v
            B = np.sqrt(D)
            numerator = (1 - param) * B - A * (
                1 / (2 * np.sqrt(D)) * (2 * R * (param - 1) - 4 * param * (param - 1) * v)
            )
            denominator = 2 * B ** 2
            return ((-1.0) / (w - v)) * numerator / denominator

    else:
        raise ValueError("h_s_x_y: Case currently not supported.")

    def sigma(w, u, v):
        return (1.0 / (w - v)) * (w - cond_copula(u, v))

    new_data = _build_new_data_row(S, X, S_vars, X_vars)
    eta = 1.0 - _predict_propensity(
        varrho_s_x_out=varrho_s_x_out,
        new_data=new_data,
        type_prop=type_prop,
        prop_lb=prop_lb,
        prop_ub=prop_ub,
    )

    u_vals = np.linspace(0.01, 0.99, integralMaxEval)
    quantile_vals = cond_quantile_forest(u_vals, S, X, data_train, S_vars, X_vars, Y_var)

    if cop == "Frank":
        if (param < 0 and W == 1) or (param >= 0 and W == 0):
            sum_vals = (1 - u_vals) * quantile_vals + (Y - quantile_vals) * (Y > quantile_vals)
        else:
            sum_vals = u_vals * quantile_vals - (quantile_vals - Y) * (Y < quantile_vals)
    elif cop == "Plackett":
        if (param < 1 and W == 1) or (param >= 1 and W == 0):
            sum_vals = (1 - u_vals) * quantile_vals + (Y - quantile_vals) * (Y > quantile_vals)
        else:
            sum_vals = u_vals * quantile_vals - (quantile_vals - Y) * (Y < quantile_vals)
    else:
        raise ValueError("h_s_x_y: Case currently not supported.")

    prod_vals = sum_vals * np.array([d_sigma(W, u, eta) for u in u_vals])

    _nc.counter += 1
    if _nc.counter % 100 == 0:
        print(f"h_s_x_y progress: {100 * _nc.counter / _nc.total}%")

    integral_part = approx_integral(y_vals=prod_vals, lowerLimit=0.01, upperLimit=0.99)

    if cop == "Frank":
        if param < 0:
            if W == 1:
                return sigma(W, 1 - W, eta) * Y + integral_part
            else:
                return sigma(W, 1 - W, eta) * Y - integral_part
        else:
            if W == 1:
                return sigma(W, W, eta) * Y - integral_part
            else:
                return sigma(W, W, eta) * Y + integral_part

    elif cop == "Plackett":
        if param < 1:
            if W == 1:
                return sigma(W, 1 - W, eta) * Y + integral_part
            else:
                return sigma(W, 1 - W, eta) * Y - integral_part
        else:
            if W == 1:
                return sigma(W, W, eta) * Y - integral_part
            else:
                return sigma(W, W, eta) * Y + integral_part

    else:
        raise ValueError("h_s_x_y: Case currently not supported.")


def mu_s_x_copula(
    W,
    S,
    X,
    varrho_s_x_out,
    eta=None,
    data_train=None,
    S_vars=None,
    X_vars=None,
    Y_var=None,
    type_prop=None,
    prop_ub=None,
    prop_lb=None,
    integralMaxEval=None,
    cop=None,
    param=None,
):

    if varrho_s_x_out is not None:
        new_data = _build_new_data_row(S, X, S_vars, X_vars)
        eta = 1.0 - _predict_propensity(
            varrho_s_x_out=varrho_s_x_out,
            new_data=new_data,
            type_prop=type_prop,
            prop_lb=prop_lb,
            prop_ub=prop_ub,
        )

    if cop == "Frank":
        def g_fn(u):
            return np.exp(-u * param) - 1

        def cond_copula(u, v):
            numerator = g_fn(u) * g_fn(v) + g_fn(v)
            denominator = g_fn(u) * g_fn(v) + g_fn(1)
            cond_cdf = numerator / denominator
            return cond_cdf

    elif cop == "Plackett":
        def cond_copula(u, v):
            numerator = -1 + u + v - u * param + v * param
            inner_term = (1 + (param - 1) * (u + v)) ** 2 - 4 * param * (param - 1) * u * v
            denominator = 2 * np.sqrt(inner_term)
            return 0.5 + numerator / denominator

    else:
        raise ValueError("mu_s_x_copula: Case currently not supported.")

    def sigma(w, u, v):
        return (1.0 / (w - v)) * (w - cond_copula(u, v))

    u_vals = np.linspace(0.01, 0.99, integralMaxEval)
    quantile_vals = cond_quantile_forest(u_vals, S, X, data_train, S_vars, X_vars, Y_var)
    prod_vals = quantile_vals * np.array([sigma(W, u, eta) for u in u_vals])

    _nc.counter += 1
    if _nc.counter % 100 == 0:
        print(f"mu_s_x_copula progress: {100 * _nc.counter / _nc.total}%")

    return approx_integral(y_vals=prod_vals, upperLimit=0.99, lowerLimit=0.01)


def bar_mu_x_copula(
    nuisance_type,
    W,
    data_train,
    S_vars,
    X_vars,
    Y_var,
    nuisance_cv_fold,
    type_prop,
    prop_ub,
    prop_lb,
    integralMaxEval,
    cop,
    param,
    grf_honesty,
    grf_tune_parameters,
    grf_num_threads,
    xgb_cv_rounds,
    xgb_eta,
    xgb_max_depth,
    xgb_threads,
):

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

        etas[test_idx] = 1.0 - _predict_propensity_batch(
            varrho_s_x_out=varrho_s_x_out,
            new_data=test_s_x,
            type_prop=type_prop,
            prop_lb=prop_lb,
            prop_ub=prop_ub,
        )

    s = df.loc[:, list(S_vars)]
    x = df.loc[:, list(X_vars)]

    _nc.counter = 0
    _nc.total = len(df)

    mu = np.array([
        mu_s_x_copula(
            W=W,
            S=s.iloc[i, :].to_numpy(),
            X=x.iloc[i, :].to_numpy(),
            eta=etas[i],
            varrho_s_x_out=None,
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
        for i in range(len(df))
    ])

    x_mat = _as_2d_numpy(x)
    mu_vec = np.asarray(mu, dtype=float)

    row_ok = np.isfinite(mu_vec) & np.all(np.isfinite(x_mat), axis=1)
    x_mat = x_mat[row_ok, :]
    mu_vec = mu_vec[row_ok]

    if nuisance_type == "glmnet":
        if len(mu_vec) == 0:
            bar_mu_x_model = float(np.nanmean(mu))
        elif _safe_sd(mu_vec) <= 1e-8:
            bar_mu_x_model = float(np.mean(mu_vec))
        else:
            if x_mat.shape[1] == 1:
                x_mat = np.column_stack([np.zeros((x_mat.shape[0], 1)), x_mat])

            try:
                bar_mu_x_model = GlmnetElasticNet(
                    alpha=1,
                    n_lambda=100,
                    n_splits=nuisance_cv_fold,
                    scoring="mean_squared_error",
                    n_jobs=1,
                    random_state=int(_rng.integers(2**31)),
                )
                bar_mu_x_model.fit(x_mat, mu_vec)
            except Exception:
                try:
                    from sklearn.linear_model import LinearRegression
                    bar_mu_x_model = LinearRegression().fit(x_mat, mu_vec)
                except Exception:
                    bar_mu_x_model = float(np.mean(mu_vec))

    elif nuisance_type == "grf":
        fallback_mean = _fallback_mean(mu_vec, mu)
        n_obs = x_mat.shape[0]
        p_dim = x_mat.shape[1]
        y_is_constant = len(mu_vec) <= 1 or not np.isfinite(_safe_sd(mu_vec)) or _safe_sd(mu_vec) <= 1e-8
        any_x_var = p_dim > 0 and np.any([_x_var_ok(x_mat[:, j]) for j in range(p_dim)])
        min_n_required = 40 if bool(grf_honesty) else 20

        if len(mu_vec) == 0:
            bar_mu_x_model = fallback_mean
        elif p_dim == 0:
            bar_mu_x_model = fallback_mean
        elif y_is_constant:
            bar_mu_x_model = float(np.mean(mu_vec))
        elif not any_x_var:
            bar_mu_x_model = float(np.mean(mu_vec))
        elif n_obs < min_n_required:
            bar_mu_x_model = float(np.mean(mu_vec))
        else:
            try:
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
                bar_mu_x_model.fit(x_mat.astype(float), mu_vec.astype(float))
            except Exception:
                bar_mu_x_model = float(np.mean(mu_vec))

    elif nuisance_type == "xgboost":
        fallback_mean = _fallback_mean(mu_vec, mu)
        n_obs = x_mat.shape[0]
        p_dim = x_mat.shape[1]
        y_is_constant = len(mu_vec) <= 1 or not np.isfinite(_safe_sd(mu_vec)) or _safe_sd(mu_vec) <= 1e-8
        any_x_var = p_dim > 0 and np.any([_x_var_ok(x_mat[:, j]) for j in range(p_dim)])
        min_n_required = max(nuisance_cv_fold, 20)

        if len(mu_vec) == 0:
            bar_mu_x_model = fallback_mean
        elif p_dim == 0:
            bar_mu_x_model = fallback_mean
        elif y_is_constant:
            bar_mu_x_model = float(np.mean(mu_vec))
        elif not any_x_var:
            bar_mu_x_model = float(np.mean(mu_vec))
        elif n_obs < min_n_required:
            bar_mu_x_model = float(np.mean(mu_vec))
        else:
            try:
                dtrain = xgb.DMatrix(x_mat, label=mu_vec)
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
            except Exception:
                bar_mu_x_model = float(np.mean(mu_vec))
    else:
        raise ValueError("Enter a valid nuisance parameter estimation type")

    return bar_mu_x_model


def d_s_x(
    S,
    X,
    varrho_s_x_out,
    data_train,
    S_vars,
    X_vars,
    Y_var,
    type_prop,
    prop_ub,
    prop_lb,
    integralMaxEval,
    cop,
    param,
):

    new_data = _build_new_data_row(S, X, S_vars, X_vars)
    eta = 1.0 - _predict_propensity(
        varrho_s_x_out=varrho_s_x_out,
        new_data=new_data,
        type_prop=type_prop,
        prop_lb=prop_lb,
        prop_ub=prop_ub,
    )

    if cop == "Frank":
        def cond_copula_dens(u, v):
            g_u = np.exp(-u * param) - 1
            g_v = np.exp(-v * param) - 1
            g_prime_v = -param * np.exp(-v * param)

            numerator = g_u * g_v + g_v
            d_numerator_dv = g_prime_v * g_u + g_prime_v

            denominator = g_u * g_v + (np.exp(-param) - 1)
            d_denominator_dv = g_prime_v * g_u

            derivative = (
                d_numerator_dv * denominator - numerator * d_denominator_dv
            ) / (denominator ** 2)
            return derivative

    elif cop == "Plackett":
        def cond_copula_dens(u, v):
            A = u * (1 - param) + v * (1 + param) - 1
            R = 1 + (param - 1) * (u + v)
            D = R ** 2 - 4 * param * (param - 1) * u * v
            B = np.sqrt(D)
            numerator = (1 + param) * B - A * (
                1 / (2 * np.sqrt(D)) * (2 * R * (param - 1) - 4 * param * (param - 1) * u)
            )
            denominator = 2 * B ** 2
            return numerator / denominator
    else:
        raise ValueError("d_s_x: Case currently not supported.")

    u_vals = np.linspace(0.01, 0.99, integralMaxEval)
    quantile_vals = cond_quantile_forest(u_vals, S, X, data_train, S_vars, X_vars, Y_var)
    prod_vals = quantile_vals * np.array([cond_copula_dens(u, eta) for u in u_vals])

    _nc.counter += 1
    if _nc.counter % 100 == 0:
        print(f"d_s_x progress: {100 * _nc.counter / _nc.total}%")

    return approx_integral(y_vals=prod_vals, upperLimit=0.99, lowerLimit=0.01)


def _safe_sd(x):
    x = np.asarray(x, dtype=float)
    if len(x) <= 1:
        return np.nan
    return float(np.std(x, ddof=1))


def _x_var_ok(z):
    z = np.asarray(z, dtype=float)
    return len(z) > 1 and np.isfinite(_safe_sd(z)) and _safe_sd(z) > 1e-8


def _fallback_mean(mu_vec, mu):
    fallback_mean = np.nanmean(mu_vec) if len(mu_vec) > 0 else np.nan
    if not np.isfinite(fallback_mean):
        fallback_mean = np.nanmean(mu)
    if not np.isfinite(fallback_mean):
        fallback_mean = 0.0
    return float(fallback_mean)
