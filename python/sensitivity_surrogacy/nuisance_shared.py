from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from econml.grf import RegressionForest
from glmnet import LogitNet
_rng = np.random.default_rng(42)


def varrho_x(
    data_train,
    X_vars,
    type_prop,
    nuisance_cv_fold,
    grf_honesty,
    grf_tune_parameters,  # not used
    grf_num_threads,
    xgb_cv_rounds,
    xgb_eta,
    xgb_max_depth,
    xgb_threads,
):
    # Conditional treatment propensity
    df = data_train.loc[data_train["observe"] == 0, :]
    x = df.loc[:, X_vars]
    w = df.loc[:, "treatment"]

    if len(np.unique(pd.to_numeric(w))) < 2:
        raise ValueError("varrho_x: `treatment` must contain both 0 and 1.")

    if type_prop == "glmnet":
        x_mat = _prepare_glmnet_matrix(x)
        varrho_x_model = _fit_glmnet_binomial(
            x_mat=x_mat,
            y=np.asarray(w, dtype=int),
            nuisance_cv_fold=nuisance_cv_fold,
        )
    elif type_prop == "grf":
        varrho_x_model = _fit_grf_regression(
            x_mat=_as_2d_numpy(x),
            y=np.asarray(w, dtype=float),
            grf_honesty=grf_honesty,
            grf_num_threads=grf_num_threads,
        )
    elif type_prop == "xgboost":
        varrho_x_model = _fit_xgboost_binomial(
            x_df=x,
            y=np.asarray(w, dtype=float),
            nuisance_cv_fold=nuisance_cv_fold,
            xgb_cv_rounds=xgb_cv_rounds,
            xgb_eta=xgb_eta,
            xgb_max_depth=xgb_max_depth,
            xgb_threads=xgb_threads,
        )
    else:
        raise ValueError("Enter a valid nuisance parameter estimation type")

    return varrho_x_model


def varrho_s_x(
    data_train,
    X_vars,
    S_vars,
    type_prop,
    nuisance_cv_fold,
    grf_honesty,
    grf_tune_parameters,  # not used
    grf_num_threads,
    xgb_cv_rounds,
    xgb_eta,
    xgb_max_depth,
    xgb_threads,
):
    # Conditional treatment propensity
    df = data_train.loc[data_train["observe"] == 0, :]
    s_x = df.loc[:, list(S_vars) + list(X_vars)]
    w = df.loc[:, "treatment"]

    if len(np.unique(pd.to_numeric(w))) < 2:
        raise ValueError("varrho_s_x: `treatment` must contain both 0 and 1.")

    if type_prop == "glmnet":
        s_x_mat = _as_2d_numpy(s_x)
        varrho_s_x_model = _fit_glmnet_binomial(
            x_mat=s_x_mat,
            y=np.asarray(w, dtype=int),
            nuisance_cv_fold=nuisance_cv_fold,
        )
    elif type_prop == "grf":
        varrho_s_x_model = _fit_grf_regression(
            x_mat=_as_2d_numpy(s_x),
            y=np.asarray(w, dtype=float),
            grf_honesty=grf_honesty,
            grf_num_threads=grf_num_threads,
        )
    elif type_prop == "xgboost":
        varrho_s_x_model = _fit_xgboost_binomial(
            x_df=s_x,
            y=np.asarray(w, dtype=float),
            nuisance_cv_fold=nuisance_cv_fold,
            xgb_cv_rounds=xgb_cv_rounds,
            xgb_eta=xgb_eta,
            xgb_max_depth=xgb_max_depth,
            xgb_threads=xgb_threads,
        )
    else:
        raise ValueError("Enter a valid nuisance parameter estimation type")

    return varrho_s_x_model


def phi_x(
    data_train,
    X_vars,
    type_prop,
    nuisance_cv_fold,
    grf_honesty,
    grf_tune_parameters,  # not used
    grf_num_threads,
    xgb_cv_rounds,
    xgb_eta,
    xgb_max_depth,
    xgb_threads,
):
    # Conditional observational sample propensity
    x = data_train.loc[:, X_vars]
    g = 1 - data_train.loc[:, "observe"]  # dependent variable is 1[P_i=E]

    if len(np.unique(pd.to_numeric(g))) < 2:
        raise ValueError("phi_x: `observe` must contain both 0 and 1.")

    if type_prop == "glmnet":
        x_mat = _prepare_glmnet_matrix(x)
        phi_x_model = _fit_glmnet_binomial(
            x_mat=x_mat,
            y=np.asarray(g, dtype=int),
            nuisance_cv_fold=nuisance_cv_fold,
        )
    elif type_prop == "grf":
        phi_x_model = _fit_grf_regression(
            x_mat=_as_2d_numpy(x),
            y=np.asarray(g, dtype=float),
            grf_honesty=grf_honesty,
            grf_num_threads=grf_num_threads,
        )
    elif type_prop == "xgboost":
        phi_x_model = _fit_xgboost_binomial(
            x_df=x,
            y=np.asarray(g, dtype=float),
            nuisance_cv_fold=nuisance_cv_fold,
            xgb_cv_rounds=xgb_cv_rounds,
            xgb_eta=xgb_eta,
            xgb_max_depth=xgb_max_depth,
            xgb_threads=xgb_threads,
        )
    else:
        raise ValueError("Enter a valid nuisance parameter estimation type")

    return phi_x_model


def phi_s_x(
    data_train,
    X_vars,
    S_vars,
    type_prop,
    nuisance_cv_fold,
    grf_honesty,
    grf_tune_parameters,  # not used
    grf_num_threads,
    xgb_cv_rounds,
    xgb_eta,
    xgb_max_depth,
    xgb_threads,
):
    # Conditional observational sample propensity
    s_x = data_train.loc[:, list(S_vars) + list(X_vars)]
    g = 1 - data_train.loc[:, "observe"]  # dependent variable is 1[P_i=E]

    if len(np.unique(pd.to_numeric(g))) < 2:
        raise ValueError("phi_s_x: `observe` must contain both 0 and 1.")

    if type_prop == "glmnet":
        s_x_mat = _as_2d_numpy(s_x)
        phi_s_x_model = _fit_glmnet_binomial(
            x_mat=s_x_mat,
            y=np.asarray(g, dtype=int),
            nuisance_cv_fold=nuisance_cv_fold,
        )
    elif type_prop == "grf":
        phi_s_x_model = _fit_grf_regression(
            x_mat=_as_2d_numpy(s_x),
            y=np.asarray(g, dtype=float),
            grf_honesty=grf_honesty,
            grf_num_threads=grf_num_threads,
        )
    elif type_prop == "xgboost":
        phi_s_x_model = _fit_xgboost_binomial(
            x_df=s_x,
            y=np.asarray(g, dtype=float),
            nuisance_cv_fold=nuisance_cv_fold,
            xgb_cv_rounds=xgb_cv_rounds,
            xgb_eta=xgb_eta,
            xgb_max_depth=xgb_max_depth,
            xgb_threads=xgb_threads,
        )
    else:
        raise ValueError("Enter a valid nuisance parameter estimation type")

    return phi_s_x_model


def _fit_glmnet_binomial(x_mat, y, nuisance_cv_fold):
    model = LogitNet(
        alpha=1,              # lasso (same as R default)
        n_lambda=100,         # same as R default
        n_splits=nuisance_cv_fold,
        scoring="log_loss",
        n_jobs=1,
        random_state=int(_rng.integers(2**31)),
    )
    model.fit(x_mat, y)
    return model


def _fit_grf_regression(x_mat, y, grf_honesty, grf_num_threads):
    model = RegressionForest(
        n_estimators=500,
        honest=grf_honesty,
        max_features="sqrt",
        inference=False,
        max_samples=0.5,            # GRF subsamples ~50% without replacement
        min_samples_leaf=5,         # closer to GRF default
        n_jobs=grf_num_threads,
        random_state=int(_rng.integers(2**31)),
    )
    model.fit(x_mat, y)
    return model


def _fit_xgboost_binomial(
    x_df,
    y,
    nuisance_cv_fold,
    xgb_cv_rounds,
    xgb_eta,
    xgb_max_depth,
    xgb_threads,
):
    x_mat = _as_2d_numpy(x_df)

    dtrain = xgb.DMatrix(
        x_mat,
        label=y,
        feature_names=list(x_df.columns),
    )

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": xgb_max_depth,
        "eta": xgb_eta,
        "nthread": xgb_threads,
    }

    cv = xgb.cv(
        params=params,
        dtrain=dtrain,
        num_boost_round=xgb_cv_rounds,
        nfold=nuisance_cv_fold,
        verbose_eval=False,
        seed=int(_rng.integers(2**31)),
    )

    best_nrounds = int(np.argmin(cv["test-logloss-mean"].to_numpy()) + 1)

    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=best_nrounds,
        verbose_eval=False,
    )

    return model


def _as_2d_numpy(df_or_array):
    if isinstance(df_or_array, pd.DataFrame):
        arr = df_or_array.to_numpy()
    elif isinstance(df_or_array, pd.Series):
        arr = df_or_array.to_numpy().reshape(-1, 1)
    else:
        arr = np.asarray(df_or_array)

    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    return arr


def _prepare_glmnet_matrix(x):
    x_mat = _as_2d_numpy(x)
    if x_mat.shape[1] == 1:
        zero_col = np.zeros((x_mat.shape[0], 1))
        x_mat = np.column_stack([zero_col, x_mat])
    return x_mat


def _build_new_data_row(S, X, S_vars, X_vars):
    new_values = np.concatenate([_flatten_row(S), _flatten_row(X)])
    return pd.DataFrame([new_values], columns=list(S_vars) + list(X_vars))


def _flatten_row(z):
    arr = np.asarray(z)
    return arr.reshape(-1)


def _clip_prob(p, prop_lb, prop_ub):
    return np.maximum(np.minimum(np.asarray(p, dtype=float), prop_ub), prop_lb)


def _predict_propensity(varrho_s_x_out, new_data, type_prop, prop_lb, prop_ub):
    pred = _predict_propensity_batch(
        varrho_s_x_out=varrho_s_x_out,
        new_data=new_data,
        type_prop=type_prop,
        prop_lb=prop_lb,
        prop_ub=prop_ub,
    )
    return float(np.asarray(pred).reshape(-1)[0])


def _predict_propensity_batch(varrho_s_x_out, new_data, type_prop, prop_lb, prop_ub):
    if type_prop == "glmnet":
        x_mat = _prepare_glmnet_matrix(new_data)
        pred = varrho_s_x_out.predict_proba(x_mat, lamb=[varrho_s_x_out.lambda_max_])[:, 1]
    elif type_prop == "grf":
        pred = np.asarray(varrho_s_x_out.predict(_as_2d_numpy(new_data))).reshape(-1)
    elif type_prop == "xgboost":
        if isinstance(new_data, pd.DataFrame):
            dnew = xgb.DMatrix(_as_2d_numpy(new_data), feature_names=list(new_data.columns))
        else:
            dnew = xgb.DMatrix(_as_2d_numpy(new_data))
        pred = np.asarray(varrho_s_x_out.predict(dnew)).reshape(-1)
    else:
        raise ValueError("Enter a valid nuisance parameter estimation type")

    return _clip_prob(pred, prop_lb=prop_lb, prop_ub=prop_ub)


def _balanced_random_folds(n, n_folds):
    reps = np.tile(np.arange(1, n_folds + 1), int(np.ceil(n / n_folds)))[:n]
    _rng.shuffle(reps)
    return reps
