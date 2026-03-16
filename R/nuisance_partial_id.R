cond_AVaR <- function(eta, S, X, neg_sign=FALSE, data_train, S_vars, X_vars, Y_var){
  df <- dplyr::filter(data_train, observe == 1)
  y <- dplyr::select(df, all_of(Y_var))*((-1)*neg_sign) + dplyr::select(df, all_of(Y_var))*(1-neg_sign)
  y <- as.vector(y[[1]])

  # Stage 1: estimate conditional eta-quantile using quantile forest (Athey et al., 2019)
  s_x <- as.matrix(dplyr::select(df, all_of(c(S_vars, X_vars))))
  q.forest <- grf::quantile_forest(s_x, y, quantiles=eta)
  rq_approx <- predict(q.forest)$predictions

  # Stage 2: sieve with a generated outcome variable corresponding to equation (3) in Olma (2021)
  psi <- unlist((1/(1-eta)) * (y*(y>=rq_approx) - rq_approx*(y>=rq_approx - (1-eta))))

  capture.output({
    suppressMessages({
      sieve_model <- Sieve::sieve.sgd.preprocess(X = s_x, type = "cosine")
      sieve_model <- Sieve::sieve.sgd.solver(sieve.model = sieve_model, X = s_x, Y = psi)
      new_values <- c(S, X)
      named_values <- setNames(new_values, c(S_vars, X_vars))
      new_data <- as.data.frame(t(named_values))
      est <- Sieve::sieve.sgd.predict(sieve_model, X = new_data)$best_model$prdy
    })
  })
  return(est)
}

mu_s_x_partial_id <- function(mu_type, S, X, varrho_s_x_out, eta = NULL, data_train, S_vars, X_vars, Y_var, type_prop, nuisance_cv_fold, prop_ub, prop_lb) {
  if (mu_type %in% c("1U", "0U")) {
    neg_sign <- FALSE
  } else {
    neg_sign <- TRUE
  }
  if (!is.null(varrho_s_x_out)) { # varrho_s_x_out is from the nuisance model for the test fold
    # predict eta=1-rho(X,S) on the given data point
    new_values <- c(unlist(S), unlist(X))
    named_values <- setNames(new_values, c(S_vars, X_vars))
    if (mu_type %in% c("1U", "1L")) {
      if (type_prop == "glmnet") {
        new_data <- as.matrix(t(named_values))
        eta <- 1-pmax(pmin(predict(varrho_s_x_out, newx = new_data, s = varrho_s_x_out$lambda.min, type = "response"), prop_ub), prop_lb)
      } else if (type_prop == "grf") {
        new_data <- as.matrix(t(named_values))
        eta <- 1-pmax(pmin(predict(varrho_s_x_out, newdata = new_data)$predictions, prop_ub), prop_lb)
      } else if (type_prop == "xgboost") {
        new_data <- as.matrix(t(named_values))
        eta <- 1-pmax(pmin(predict(varrho_s_x_out, newdata = new_data), prop_ub), prop_lb)
      }
    } else {
      if (type_prop == "glmnet") {
        new_data <- as.matrix(t(named_values))
        eta <- pmax(pmin(predict(varrho_s_x_out, newx = new_data, s = varrho_s_x_out$lambda.min, type = "response"), prop_ub), prop_lb)
      } else if (type_prop == "grf") {
        new_data <- as.matrix(t(named_values))
        eta <- pmax(pmin(predict(varrho_s_x_out, newdata = new_data)$predictions, prop_ub), prop_lb)
      } else if (type_prop == "xgboost") {
        new_data <- as.matrix(t(named_values))
        eta <- pmax(pmin(predict(varrho_s_x_out, newdata = new_data), prop_ub), prop_lb)
      }
    }
  } else {
    # Leave-one-out within the training data for estimating E[H | S_i, X_i]
    # Here, we perform leave-one-out instead of cross-fitting with a few data folds to ensure there are enough observations for conditional quantile estimation using quantile forests (see cond_AVaR function)
    data_train <- data_train[!apply(data_train, 1, function(row) all(row[S_vars] == unlist(S)) & all(row[X_vars] == unlist(X))),] # training data without the given (X,S) arguments
  }
  return(cond_AVaR(as.numeric(eta), unlist(S), unlist(X), neg_sign = neg_sign, data_train = data_train, S_vars = S_vars, X_vars = X_vars, Y_var = Y_var))
}

bar_mu_x_partial_id <- function(mu_type, data_train, treatment_val, S_vars, X_vars, Y_var,
                     type, type_prop, prop_lb, prop_ub,
                     nuisance_cv_fold, grf_honesty, grf_tune_parameters, grf_num_threads,
                     xgb_cv_rounds, xgb_eta, xgb_max_depth, xgb_threads) {
  # Cross-fitting within the training data
  df <- dplyr::filter(data_train, observe == 0)
  df_0 <- df[df$treatment == 0,]
  df_1 <- df[df$treatment == 1,]
  if (nrow(df_0) < nuisance_cv_fold || nrow(df_1) < nuisance_cv_fold) {
    warning("Some treatment arms in the observational sample have fewer observations than `nuisance_cv_fold`; cross-fitting may be unstable.")
  }
  df_0$fold <- sample(rep(1:nuisance_cv_fold, length.out = nrow(df_0)))
  df_1$fold <- sample(rep(1:nuisance_cv_fold, length.out = nrow(df_1)))
  df <- rbind(df_0, df_1)
  folds <- split(1:nrow(df), df$fold) # balanced data split
  etas <- rep(NA, nrow(df))

  # Perform cross-fitting for rho(S_i, X_i)
  for (j in seq_along(folds)) {
    test_idx <- folds[[j]]
    test_s_x <- as.matrix(df[test_idx, names(df) %in% c(S_vars, X_vars)])
    train_idx <- setdiff(seq_len(nrow(df)), test_idx)
    train_s_x_y <- df[train_idx,]
    varrho_s_x_out <- varrho_s_x(train_s_x_y, X_vars, S_vars, type_prop, nuisance_cv_fold,
                                 grf_honesty, grf_tune_parameters, grf_num_threads,
                                 xgb_cv_rounds, xgb_eta, xgb_max_depth, xgb_threads)
    if (mu_type %in% c("1U", "1L")) {
      if (type_prop == "glmnet") {
        etas[test_idx] <- 1-pmax(pmin(predict(varrho_s_x_out, newx = test_s_x, s = varrho_s_x_out$lambda.min, type = "response"), prop_ub), prop_lb)
      } else if (type_prop == "grf") {
        etas[test_idx] <- 1-pmax(pmin(predict(varrho_s_x_out, newdata = test_s_x)$predictions, prop_ub), prop_lb)
      } else if (type_prop == "xgboost") {
        etas[test_idx] <- 1-pmax(pmin(predict(varrho_s_x_out, newdata = test_s_x[, varrho_s_x_out$feature_names]), prop_ub), prop_lb)
      }
    } else if (mu_type %in% c("0U", "0L")) {
      if (type_prop == "glmnet") {
        etas[test_idx] <- pmax(pmin(predict(varrho_s_x_out, newx = test_s_x, s = varrho_s_x_out$lambda.min, type = "response"), prop_ub), prop_lb)
      } else if (type_prop == "grf") {
        etas[test_idx] <- pmax(pmin(predict(varrho_s_x_out, newdata = test_s_x)$predictions, prop_ub), prop_lb)
      } else if (type_prop == "xgboost") {
        etas[test_idx] <- pmax(pmin(predict(varrho_s_x_out, newdata = test_s_x[, varrho_s_x_out$feature_names]), prop_ub), prop_lb)
      }
    }
  }
  df$eta <- etas
  df <- dplyr::filter(df, treatment == treatment_val)
  s <- dplyr::select(df, all_of(S_vars))
  x <- dplyr::select(df, all_of(X_vars))
  etas <- df$eta
  mu_x_pred <- mapply(mu_s_x_partial_id, S = split(as.matrix(s), row(s)),
                         X = split(as.matrix(x), row(x)),
                         eta = split(etas, seq_along(etas)),
                         MoreArgs = list(mu_type = mu_type,
                                         varrho_s_x_out = NULL,
                                         data_train = data_train,
                                         S_vars = S_vars, X_vars = X_vars,
                                         Y_var = Y_var, type_prop = type_prop,
                                         nuisance_cv_fold = nuisance_cv_fold,
                                         prop_ub = prop_ub, prop_lb = prop_lb))

  if (type == "glmnet"){
    if (ncol(x)==1) {
      x <- cbind(0,as.matrix(x))
    }
    bar_mu_x_model <- glmnet::cv.glmnet(x = as.matrix(x), y = as.matrix(mu_x_pred), nfolds = nuisance_cv_fold)
  } else if(type == "grf"){
    bar_mu_x_model <- grf::regression_forest(X = as.matrix(x), Y = as.matrix(mu_x_pred), num.threads = grf_num_threads,
                               ci.group.size = 1, honesty = grf_honesty, tune.parameters = grf_tune_parameters)
  } else if(type == "xgboost"){
    x_mat <- as.matrix(x)
    storage.mode(x_mat) <- "double"
    mu_vec <- as.numeric(mu_x_pred)
    cv_fit <- xgboost::xgb.cv(
        data = x_mat,
        label = mu_vec,
        objective = "reg:squarederror",
        max_depth = xgb_max_depth,
        eta = xgb_eta,
        nthread = xgb_threads,
        nrounds = xgb_cv_rounds,
        verbose = FALSE,
        nfold = nuisance_cv_fold
        )
    best_nrounds <- cv_fit$best_iteration
    if (is.null(best_nrounds) || !is.finite(best_nrounds) || best_nrounds < 1L) {
      best_nrounds <- which.min(cv_fit$evaluation_log$test_rmse_mean)
    }
    if (!is.finite(best_nrounds) || best_nrounds < 1L) {
      best_nrounds <- xgb_cv_rounds
    }
    bar_mu_x_model <- xgboost::xgboost(
      data = x_mat,
      label = mu_vec,
      objective = "reg:squarederror",
      max_depth = xgb_max_depth,
      eta = xgb_eta,
      nthread = xgb_threads,
      nrounds = best_nrounds,
      verbose = FALSE
    )
  }
  return(bar_mu_x_model)
}

q_s_x <- function(q_type, data_train, S_vars, X_vars, Y_var, type_prop, prop_ub, prop_lb, S, X, varrho_s_x_out){
  # predict rho(S,X) using the observational sample
  df <- dplyr::filter(data_train, observe == 1)
  s_x <- dplyr::select(df, all_of(c(S_vars, X_vars)))
  s <- dplyr::select(df, all_of(S_vars))
  x <- dplyr::select(df, all_of(X_vars))
  y <- dplyr::select(df, all_of(Y_var))
  y <- as.vector(y[[1]])
  if (q_type == "U") {
    if (type_prop == "glmnet") {
      eta <- as.numeric(1-pmax(pmin(predict(varrho_s_x_out, newx = t(as.matrix(c(S,X))), s = varrho_s_x_out$lambda.min, type = "response"), prop_ub), prop_lb))
    } else if (type_prop == "grf") {
      eta <- as.numeric(1-pmax(pmin(predict(varrho_s_x_out, newdata = t(as.matrix(c(S,X))))$predictions, prop_ub), prop_lb))
    } else if (type_prop == "xgboost") {
      eta <- as.numeric(1-pmax(pmin(predict(varrho_s_x_out, newdata = t(as.matrix(c(S,X)))), prop_ub), prop_lb))
    }
  } else {
    if (type_prop == "glmnet") {
      eta <- as.numeric(pmax(pmin(predict(varrho_s_x_out, newx = t(as.matrix(c(S,X))), s = varrho_s_x_out$lambda.min, type = "response"), prop_ub), prop_lb))
    } else if (type_prop == "grf") {
      eta <- as.numeric(pmax(pmin(predict(varrho_s_x_out, newdata = t(as.matrix(c(S,X))))$predictions, prop_ub), prop_lb))
    } else if (type_prop == "xgboost") {
      eta <- as.numeric(pmax(pmin(predict(varrho_s_x_out, newdata = t(as.matrix(c(S,X)))), prop_ub), prop_lb))
    }
  }

  # compute quantile using the observational sample
  q.forest <- grf::quantile_forest(s_x, y, quantiles=eta)
  new_values <- c(unlist(S), unlist(X))
  named_values <- setNames(new_values, c(S_vars, X_vars))
  new_data <- as.data.frame(t(named_values))
  return(predict(q.forest, new_data)$predictions)
}

H_y_s <- function(H_type, Y, S, eta) {
  if (H_type == "U") {
    return(S + (1/eta) * ((Y>S)*(Y-S)))
  } else {
    return(S - (1/eta) * ((S>Y)*(S-Y)))
  }
}
