cond_quantile_forest <- function(u, S, X, data_train, S_vars, X_vars, Y_var){
  df <- dplyr::filter(data_train, observe == 1)
  s_x <- as.matrix(dplyr::select(df, all_of(c(S_vars, X_vars))))
  y <- unlist(as.vector(df[[Y_var]]))
  rq_model <- grf::quantile_forest(s_x, y, quantiles=u) # u is a vector for integral approximation
  new_values <- c(unlist(S), unlist(X))
  named_values <- setNames(new_values, c(S_vars, X_vars))
  new_data <- as.data.frame(t(named_values))
  return(predict(rq_model, new_data)$predictions)
}

approx_integral <- function(y_vals, upperLimit, lowerLimit) {
  n_points <- length(y_vals)
  h <- (upperLimit - lowerLimit) / (n_points - 1) # step size
  integral <- (h / 2) * (y_vals[1] + 2 * sum(y_vals[2:(n_points - 1)]) + y_vals[n_points]) # trapezoidal rule
  return(integral)
}

h_s_x_y <- function(W, S, X, Y, varrho_s_x_out, data_train, S_vars, X_vars, Y_var, type_prop,
                    nuisance_cv_fold, prop_ub, prop_lb, integralMaxEval, cop, param) {
  if (is.na(Y)) {
    # If data point comes from the experimental sample, output zero.
    # This check is for computing data_nuisance in longterm_copula.R
    # This step is only for preventing error, and the output value does not matter.
    # In the end, we only use data points from the observational sample for evaluating h (see compute_tau in longterm_copula.R).
    return(0)
  }
  if (cop == "Frank") {
    g_fn <- function(u) {
      return(exp(-u*param) - 1)
    }
    cond_copula <- function(u, v) {
      numerator <- g_fn(u)*g_fn(v) + g_fn(v)
      denominator <- g_fn(u)*g_fn(v) + g_fn(1)
      cond_cdf <- numerator / denominator
      return(cond_cdf)
    }
    d_sigma <- function(w, u, v) {
      A <- g_fn(u)*g_fn(v) + g_fn(v)
      B <- g_fn(u)*g_fn(v) + g_fn(1)
      dA_du <- (-param * exp(-u * param)) * g_fn(v)
      dB_du <- (-param * exp(-u * param)) * g_fn(v)
      numerator <- B * dA_du - A * dB_du
      denominator <- B^2
      return(((-1)/(w-v)) * numerator / denominator)
    }
  } else if (cop == "Plackett") {
    cond_copula <- function(u, v) {
      numerator <- -1 + u + v - u * param + v * param
      inner_term <- (1 + (param - 1) * (u + v))^2 - 4 * param * (param - 1) * u * v
      denominator <- 2 * sqrt(inner_term)
      return(0.5 + numerator / denominator)
    }
    d_sigma <- function(w, u, v) {
      A <- u*(1-param)+v*(1+param)-1
      R <- 1+(param-1)*(u+v)
      D <- R^2 - 4*param*(param-1)*u*v
      B <- sqrt(D)
      numerator <- (1-param)*B - A*( 1/(2*sqrt(D)) * (2*R*(param-1) - 4*param*(param-1)*v))
      denominator <- 2*B^2
      return(((-1)/(w-v)) * numerator / denominator)
    }
  }

  sigma <- function(w, u, v) {
    return((1/(w-v)) * (w - cond_copula(u, v)))
  }

  # predict eta=1-rho(X,S) on the given data point
  new_values <- c(unlist(S), unlist(X))
  named_values <- setNames(new_values, c(S_vars, X_vars))
  new_data <- as.matrix(t(named_values))

  if (type_prop == "glmnet") {
    eta <- 1-pmax(pmin(predict(varrho_s_x_out, newx = new_data, s = varrho_s_x_out$lambda.min, type = "response"), prop_ub), prop_lb)
  } else if (type_prop == "grf") {
    eta <- 1-pmax(pmin(predict(varrho_s_x_out, newdata = new_data)$predictions, prop_ub), prop_lb)
  } else if (type_prop == "xgboost") {
    eta <- 1-pmax(pmin(predict(varrho_s_x_out, newdata = new_data), prop_ub), prop_lb)
  }

  # Integral approximation
  u_vals <- seq(0.01, 0.99, length.out = integralMaxEval)
  quantile_vals <- cond_quantile_forest(u_vals, S, X, data_train, S_vars, X_vars, Y_var)
  if (cop == "Frank") {
    if ((param < 0 & W == 1) || (param >= 0 & W == 0)) { # V stochastically decreasing in U and W=1, or V stochastically increasing in U and W=0
      sum_vals <- (1-u_vals)*quantile_vals + (Y-quantile_vals)*(Y>quantile_vals)
    } else { # stochastically increasing and W=1, or decreasing and W=0
      sum_vals <- u_vals*quantile_vals - (quantile_vals-Y)*(Y<quantile_vals)
    }
  } else if (cop == "Plackett") {
    if ((param < 1 & W == 1) || (param >= 1 & W == 0)) { # V stochastically decreasing in U and W=1, or V stochastically increasing in U and W=0
      sum_vals <- (1-u_vals)*quantile_vals + (Y-quantile_vals)*(Y>quantile_vals)
    } else { # stochastically increasing and W=1, or decreasing and W=0
      sum_vals <- u_vals*quantile_vals - (quantile_vals-Y)*(Y<quantile_vals)
    }
  } else {
    stop("h_s_x_y: Case currently not supported.")
  }
  prod_vals <- sum_vals*sapply(u_vals, function(u) d_sigma(W, u, eta))

  counter <<- counter+1
  if (counter %% 100 == 0) {
    print(paste0("h_s_x_y progress: ", 100*counter/total, "%"))  # show progress
  }

  if (cop == "Frank") { # 4 cases
    if (param < 0) {
      if (W == 1) { # (1) V stochastically decreasing in U and W=1
        return(sigma(W, 1-W, eta)*Y + approx_integral(y_vals = prod_vals, lowerLimit = 0.01, upperLimit = 0.99))
      } else { # (2) V stochastically decreasing in U and W=0
        return(sigma(W, 1-W, eta)*Y - approx_integral(y_vals = prod_vals, lowerLimit = 0.01, upperLimit = 0.99))
      }
    } else {
      if (W == 1) { # (3) V stochastically increasing in U and W=1
        return(sigma(W, W, eta)*Y - approx_integral(y_vals = prod_vals, lowerLimit = 0.01, upperLimit = 0.99))
      } else { # (4) V stochastically increasing in U and W=0
        return(sigma(W, W, eta)*Y + approx_integral(y_vals = prod_vals, lowerLimit = 0.01, upperLimit = 0.99))
      }
    }
  } else if (cop == "Plackett") {
    if (param < 1) {
      if (W == 1) { # (1) V stochastically decreasing in U and W=1
        return(sigma(W, 1-W, eta)*Y + approx_integral(y_vals = prod_vals, lowerLimit = 0.01, upperLimit = 0.99))
      } else { # (2) V stochastically decreasing in U and W=0
        return(sigma(W, 1-W, eta)*Y - approx_integral(y_vals = prod_vals, lowerLimit = 0.01, upperLimit = 0.99))
      }
    } else {
      if (W == 1) { # (3) V stochastically increasing in U and W=1
        return(sigma(W, W, eta)*Y - approx_integral(y_vals = prod_vals, lowerLimit = 0.01, upperLimit = 0.99))
      } else { # (4) V stochastically increasing in U and W=0
        return(sigma(W, W, eta)*Y + approx_integral(y_vals = prod_vals, lowerLimit = 0.01, upperLimit = 0.99))
      }
    }
  } else {
    stop("h_s_x_y: Case currently not supported.")
  }
}

mu_s_x_copula <- function(W, S, X, varrho_s_x_out, eta = NULL, data_train, S_vars, X_vars, Y_var, type_prop, prop_ub, prop_lb, integralMaxEval, cop, param) {
  if (!is.null(varrho_s_x_out)) {  # varrho_s_x_out is from the nuisance model for the test fold
    # predict eta=1-rho(X,S) on the given data point
    new_values <- c(unlist(S), unlist(X))
    named_values <- setNames(new_values, c(S_vars, X_vars))
    new_data <- as.matrix(t(named_values))

    if (type_prop == "glmnet") {
      eta <- 1-pmax(pmin(predict(varrho_s_x_out, newx = new_data, s = varrho_s_x_out$lambda.min, type = "response"), prop_ub), prop_lb)
    } else if (type_prop == "grf") {
      eta <- 1-pmax(pmin(predict(varrho_s_x_out, newdata = new_data)$predictions, prop_ub), prop_lb)
    } else if (type_prop == "xgboost") {
      eta <- 1-pmax(pmin(predict(varrho_s_x_out, newdata = new_data), prop_ub), prop_lb)
    }
  }

  if (cop == "Frank") {
    g_fn <- function(u) {
      return(exp(-u*param) - 1)
    }
    cond_copula <- function(u, v) {
      numerator <- g_fn(u)*g_fn(v) + g_fn(v)
      denominator <- g_fn(u)*g_fn(v) + g_fn(1)
      cond_cdf <- numerator / denominator
      return(cond_cdf)
    }
  } else if (cop == "Plackett") {
    cond_copula <- function(u, v) {
      numerator <- -1 + u + v - u * param + v * param
      inner_term <- (1 + (param - 1) * (u + v))^2 - 4 * param * (param - 1) * u * v
      denominator <- 2 * sqrt(inner_term)
      return(0.5 + numerator / denominator)
    }
  }
  sigma <- function(w, u, v) {
    return((1/(w-v)) * (w - cond_copula(u, v)))
  }

  # Integral approximation
  u_vals <- seq(0.01, 0.99, length.out = integralMaxEval)
  quantile_vals <- cond_quantile_forest(u_vals, S, X, data_train, S_vars, X_vars, Y_var)
  prod_vals <- quantile_vals*sapply(u_vals, function(u) sigma(W, u, eta))

  counter <<- counter+1
  if (counter %% 100 == 0) {
    print(paste0("mu_s_x_copula progress: ", 100*counter/total, "%"))  # show progress
    curr_time <<- Sys.time()
  }

  return(approx_integral(y_vals = prod_vals, upperLimit = 0.99, lowerLimit = 0.01))
}

bar_mu_x_copula <- function(nuisance_type, W, data_train, S_vars, X_vars, Y_var, nuisance_cv_fold,
                     type_prop, prop_ub, prop_lb, integralMaxEval, cop, param,
                     grf_honesty, grf_tune_parameters, grf_num_threads,
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
    if (type_prop == "glmnet") {
      etas[test_idx] <- 1-pmax(pmin(predict(varrho_s_x_out, newx = test_s_x, s = varrho_s_x_out$lambda.min, type = "response"), prop_ub), prop_lb)
    } else if (type_prop == "grf") {
      etas[test_idx] <- 1-pmax(pmin(predict(varrho_s_x_out, newdata = test_s_x)$predictions, prop_ub), prop_lb)
    } else if (type_prop == "xgboost") {
      etas[test_idx] <- 1-pmax(pmin(predict(varrho_s_x_out, newdata = test_s_x[, varrho_s_x_out$feature_names]), prop_ub), prop_lb)
    }
  }

  s <- dplyr::select(df, all_of(S_vars))
  x <- dplyr::select(df, all_of(X_vars))
  counter <<- 0
  total <<- nrow(df)
  curr_time <<- Sys.time()
  mu <- mapply(mu_s_x_copula, S = split(as.matrix(s), row(s)),
               X = split(as.matrix(x), row(x)),
               eta = split(etas, seq_along(etas)),
               MoreArgs = list(W = W,
                               varrho_s_x_out = NULL,
                               data_train = data_train,
                               S_vars = S_vars, X_vars = X_vars,
                               Y_var = Y_var, type_prop = type_prop,
                               prop_ub = prop_ub, prop_lb = prop_lb,
                               integralMaxEval = integralMaxEval,
                               cop = cop, param = param))
  if (nuisance_type == "glmnet") {
    # ensure column matrix shape
    x_mat <- as.matrix(x)
    mu_vec <- as.numeric(mu)

    # keep only finite rows in both x and r
    row_ok <- is.finite(mu_vec) & apply(x_mat, 1, function(z) all(is.finite(z)))
    x_mat <- x_mat[row_ok, , drop = FALSE]
    mu_vec <- mu_vec[row_ok]

    # guard empty or degenerate after filtering
    if (length(mu_vec) == 0L) {
      bar_mu_x_model <- mean(mu, na.rm = TRUE)  # everything was non-finite; safest fallback
    } else if (sd(mu_vec) <= 1e-8) {
      bar_mu_x_model <- mean(mu_vec)            # constant (or nearly) -> constant model
    } else {
      # glmnet dislikes single-column without intercept sometimes; pad if needed
      if (ncol(x_mat) == 1L) x_mat <- cbind(zeros = 0, x_mat)

      # try CV glmnet, fall back to plain glmnet; final fallback = constant mean
      bar_mu_x_model <- tryCatch({
        glmnet::cv.glmnet(x = x_mat, y = mu_vec, nfolds = nuisance_cv_fold)
      }, error = function(e) {
        tryCatch({
          glmnet(x = x_mat, y = mu_vec)
        }, error = function(e2) {
          mean(mu_vec)
        })
      })
    }
  } else if (nuisance_type == "grf") {
    x_mat <- as.matrix(x)
    storage.mode(x_mat) <- "double"
    mu_vec <- as.numeric(mu)
    row_ok <- is.finite(mu_vec) & apply(x_mat, 1, function(z) all(is.finite(z)))
    x_mat <- x_mat[row_ok, , drop = FALSE]
    mu_vec <- mu_vec[row_ok]
    fallback_mean <- mean(mu_vec, na.rm = TRUE)
    if (!is.finite(fallback_mean)) fallback_mean <- mean(mu, na.rm = TRUE)
    if (!is.finite(fallback_mean)) fallback_mean <- 0
    x_var_ok <- function(z) {
      length(z) > 1L && is.finite(sd(z)) && sd(z) > 1e-8
    }
    n_obs <- nrow(x_mat)
    p_dim <- ncol(x_mat)
    y_is_constant <- length(mu_vec) <= 1L || !is.finite(sd(mu_vec)) || sd(mu_vec) <= 1e-8
    any_x_var <- p_dim > 0L && any(apply(x_mat, 2, x_var_ok))
    min_n_required <- if (isTRUE(grf_honesty)) 40L else 20L
    if (length(mu_vec) == 0L) {
      bar_mu_x_model <- fallback_mean
    } else if (p_dim == 0L) {
      bar_mu_x_model <- fallback_mean
    } else if (y_is_constant) {
      bar_mu_x_model <- mean(mu_vec)
    } else if (!any_x_var) {
      bar_mu_x_model <- mean(mu_vec)
    } else if (n_obs < min_n_required) {
      bar_mu_x_model <- mean(mu_vec)
    } else {
      bar_mu_x_model <- tryCatch({
        grf::regression_forest(
          X = x_mat,
          Y = mu_vec,
          num.threads = grf_num_threads,
          ci.group.size = 1,
          honesty = grf_honesty,
          tune.parameters = grf_tune_parameters
        )
      }, error = function(e) {
        mean(mu_vec)
      })
    }
  } else if (nuisance_type == "xgboost") {
    x_mat <- as.matrix(x)
    storage.mode(x_mat) <- "double"
    mu_vec <- as.numeric(mu)
    row_ok <- is.finite(mu_vec) & apply(x_mat, 1, function(z) all(is.finite(z)))
    x_mat <- x_mat[row_ok, , drop = FALSE]
    mu_vec <- mu_vec[row_ok]
    fallback_mean <- mean(mu_vec, na.rm = TRUE)
    if (!is.finite(fallback_mean)) fallback_mean <- mean(mu, na.rm = TRUE)
    if (!is.finite(fallback_mean)) fallback_mean <- 0
    x_var_ok <- function(z) {
      length(z) > 1L && is.finite(sd(z)) && sd(z) > 1e-8
    }
    n_obs <- nrow(x_mat)
    p_dim <- ncol(x_mat)
    y_is_constant <- length(mu_vec) <= 1L || !is.finite(sd(mu_vec)) || sd(mu_vec) <= 1e-8
    any_x_var <- p_dim > 0L && any(apply(x_mat, 2, x_var_ok))
    min_n_required <- max(nuisance_cv_fold, 20L)
    if (length(mu_vec) == 0L) {
      bar_mu_x_model <- fallback_mean
    } else if (p_dim == 0L) {
      bar_mu_x_model <- fallback_mean
    } else if (y_is_constant) {
      bar_mu_x_model <- mean(mu_vec)
    } else if (!any_x_var) {
      bar_mu_x_model <- mean(mu_vec)
    } else if (n_obs < min_n_required) {
      bar_mu_x_model <- mean(mu_vec)
    } else {
      bar_mu_x_model <- tryCatch({
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
        xgboost::xgboost(
          data = x_mat,
          label = mu_vec,
          objective = "reg:squarederror",
          max_depth = xgb_max_depth,
          eta = xgb_eta,
          nthread = xgb_threads,
          nrounds = best_nrounds,
          verbose = FALSE
        )
      }, error = function(e) {
        mean(mu_vec)
      })
    }
  }
  return(bar_mu_x_model)
}

d_s_x <- function(S, X, varrho_s_x_out, data_train, S_vars, X_vars, Y_var, type_prop,
                  prop_ub, prop_lb, integralMaxEval, cop, param) {
  # predict eta=1-rho(X,S) on the given data point
  new_values <- c(unlist(S), unlist(X))
  named_values <- setNames(new_values, c(S_vars, X_vars))

  if (type_prop == "glmnet") {
    new_data <- as.matrix(t(named_values))
    eta <- 1-pmax(pmin(predict(varrho_s_x_out, newx = new_data, s = varrho_s_x_out$lambda.min, type = "response"), prop_ub), prop_lb)
  } else if (type_prop == "grf") {
    new_data <- as.matrix(t(named_values))
    eta <- 1-pmax(pmin(predict(varrho_s_x_out, newdata = new_data)$predictions, prop_ub), prop_lb)
  } else if (type_prop == "xgboost"){
    new_data <- as.matrix(t(named_values))
    eta <- 1-pmax(pmin(predict(varrho_s_x_out, newdata = new_data), prop_ub), prop_lb)
  }

  # Compute c(v|u) = c(u,v)
  if (cop == "Frank") {
    cond_copula_dens <- function(u, v) {
      g_u <- exp(-u * param) - 1
      g_v <- exp(-v * param) - 1
      g_prime_v <- -param * exp(-v * param)
      # Compute the numerator and its derivative
      numerator <- g_u * g_v + g_v
      d_numerator_dv <- g_prime_v * g_u + g_prime_v
      # Compute the denominator and its derivative
      denominator <- g_u * g_v + (exp(-param) - 1)
      d_denominator_dv <- g_prime_v * g_u
      # Apply the quotient rule for differentiation
      derivative <- (d_numerator_dv * denominator - numerator * d_denominator_dv) / (denominator^2)
      return(derivative)
    }
  } else if (cop == "Plackett") {
    cond_copula_dens <- function(u, v) {
      A <- u*(1-param)+v*(1+param)-1
      R <- 1+(param-1)*(u+v)
      D <- R^2 - 4*param*(param-1)*u*v
      B <- sqrt(D)
      numerator <- (1+param)*B - A*( 1/(2*sqrt(D)) * (2*R*(param-1) - 4*param*(param-1)*u))
      denominator <- 2*B^2
      return(numerator / denominator)
    }
  }

  u_vals <- seq(0.01, 0.99, length.out = integralMaxEval)
  quantile_vals <- cond_quantile_forest(u_vals, S, X, data_train, S_vars, X_vars, Y_var)
  prod_vals <- quantile_vals*sapply(u_vals, function(u) cond_copula_dens(u, eta))

  counter <<- counter+1
  if (counter %% 100 == 0) {
    print(paste0("d_s_x progress: ", 100*counter/total, "%"))  # show progress
  }
  return(approx_integral(y_vals = prod_vals, upperLimit = 0.99, lowerLimit = 0.01))
}
