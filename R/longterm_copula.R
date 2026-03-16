#' Estimate Long-Term Treatment Effects for a Copula-Based Sensitivity Analysis
#'
#' This function implements the copula-based sensitivity analysis proposed in
#' \cite{Fan, Manzanares, Park, Qi (2026)} to estimate long-term treatment effects
#' using surrogate outcomes. The procedure combines experimental and observational
#' samples and uses flexible machine learning methods to estimate nuisance
#' components required for the orthogonal moment equations.
#'
#' @param data A data frame containing the analysis dataset. The data must include
#' the binary indicators \code{"treatment"} (treatment assignment) and
#' \code{"observe"} (indicator for observational versus experimental sample),
#' along with pre-treatment covariates, short-term outcomes, and the long-term
#' outcome. If certain variables are structurally missing in part of the data
#' (for example, long-term outcomes not observed in the experimental sample),
#' they can be filled with any placeholder value because those observations
#' will not enter the corresponding estimation steps.
#'
#' @param S_vars Character vector containing the names of the surrogate (short-term)
#' outcome variables.
#'
#' @param X_vars Character vector containing the names of the pre-treatment
#' covariates.
#'
#' @param Y_var A single character string specifying the long-term outcome variable.
#'
#' @param type Method used to estimate WSI nuisance components. Supported options
#' are \code{"glmnet"} (cross-validated lasso), \code{"grf"} (generalized random
#' forests), and \code{"xgboost"} (gradient boosting).
#'
#' @param type_prop Method used to estimate propensity and surrogacy score nuisance
#' components. Supported options are \code{"glmnet"}, \code{"grf"}, and
#' \code{"xgboost"}.
#'
#' @param cop Copula family used to model dependence between potential outcomes.
#' Currently supported options are \code{"Frank"} and \code{"Plackett"}.
#'
#' @param param Numeric value specifying the copula parameter.
#'
#' @param prop_lb Lower trimming threshold applied to estimated propensity scores.
#'
#' @param prop_ub Upper trimming threshold applied to estimated propensity scores.
#'
#' @param alpha Significance level used when constructing confidence intervals.
#' The resulting intervals have nominal coverage (1-alpha).
#'
#' @param integralMaxEval Maximum number of function evaluations used when
#' numerically approximating integrals appearing in the estimation procedure.
#'
#' @param cross_fit_fold Number of folds used for cross-fitting nuisance parameter
#' estimators.
#'
#' @param nuisance_cv_fold Number of folds used for cross-validation within
#' nuisance estimation procedures.
#'
#' @param grf_honesty Logical value controlling the \code{honesty} option in
#' \code{grf}.
#'
#' @param grf_tune_parameters Character value specifying how tuning parameters
#' are selected in \code{grf}.
#'
#' @param grf_num_threads Number of threads used by \code{grf}.
#'
#' @param xgb_cv_rounds Maximum number of boosting rounds used during
#' cross-validation for \code{xgboost}.
#'
#' @param xgb_eta Learning rate parameter used in \code{xgboost}.
#'
#' @param xgb_max_depth Maximum tree depth used in \code{xgboost}.
#'
#' @param xgb_threads Number of threads used by \code{xgboost}.
#'
#' @return A list with three elements:
#' \describe{
#' \item{\code{hat_tau}}{Point estimate of the average treatment effect.}
#' \item{\code{se}}{Estimated standard error.}
#' \item{\code{ci}}{Two-element vector containing the lower and upper bounds of
#' the confidence interval.}
#' }
#'
#' @export
#'
#' @references
#' Fan, Y., Manzanares, C. A., Park, H., & Qi, Y. (2026).
#' A Sensitivity Analysis of the Surrogate Index Approach for Estimating
#' Long-Term Treatment Effects. arXiv:2603.00580.
#'
#' @importFrom magrittr %>%
#' @importFrom rlang .data

longterm_copula <- function(data, S_vars, X_vars, Y_var, type, type_prop, cop, param,
                      prop_lb = 0.01, prop_ub = 0.99, alpha = 0.05, integralMaxEval = 1000,
                      cross_fit_fold = 5, nuisance_cv_fold = 5,
                      grf_honesty = TRUE, grf_tune_parameters = "all",
                      grf_num_threads = 1, xgb_cv_rounds = 100, xgb_eta = 0.1,
                      xgb_max_depth = 2, xgb_threads = 1){
  ### Input checks
  .validate_longterm_copula_inputs(
    data, S_vars, X_vars, Y_var, type, type_prop, cop, param,
    prop_lb, prop_ub, alpha, integralMaxEval,
    cross_fit_fold, nuisance_cv_fold
  )
  ### Estimate Long-Term Treatment Effects
  # Choose random k-fold of the data
  folds <- split(seq(nrow(data)), sample(rep(1:cross_fit_fold, length.out = nrow(data))))
  # Compute the nuisance parameters
  data_nuisance_out <- lapply(seq(cross_fit_fold),
                              compute_nuisance_on_fold_copula, data,
                              folds, S_vars, X_vars, Y_var, type, type_prop, cop, param,
                              prop_lb, prop_ub, integralMaxEval,
                              cross_fit_fold, nuisance_cv_fold,
                              grf_honesty, grf_tune_parameters, grf_num_threads,
                              xgb_cv_rounds, xgb_eta, xgb_max_depth, xgb_threads)
  data_nuisance     <- dplyr::bind_rows(data_nuisance_out)
  # Compute the estimate of tau
  hat_tau   <- compute_tau_copula(data_nuisance)

  # Compute the standard error and construct confidence intervals for bounds
  se_ci <- compute_ci_copula(hat_tau, data_nuisance, alpha)
  se    <- se_ci$se
  ci    <- se_ci$ci

  return(list(hat_tau = hat_tau, se = se, ci = ci))
}

compute_nuisance_on_fold_copula <- function(i, data, folds, S_vars, X_vars, Y_var, type,
                                     type_prop, cop, param, prop_lb, prop_ub, integralMaxEval,
                                     cross_fit_fold, nuisance_cv_fold,
                                     grf_honesty, grf_tune_parameters, grf_num_threads,
                                     xgb_cv_rounds, xgb_eta, xgb_max_depth, xgb_threads){
  print(paste0("------Estimating fold ", i, "/", cross_fit_fold, " for ", cop, " copula with parameter = ", param, "------"))
  ## Subset to training and testing samples
  data_train <- data[as.numeric(unlist(folds[setdiff(seq(cross_fit_fold), i)])), ]
  data_test  <- data[as.numeric(unlist(folds[i])), ]

  ## Train nuisance parameters
  nuisance <- train_nuisance_copula(data_train, S_vars, X_vars, Y_var, type, type_prop,
                             cop, param, prop_lb, prop_ub, integralMaxEval, nuisance_cv_fold,
                             grf_honesty, grf_tune_parameters, grf_num_threads,
                             xgb_cv_rounds, xgb_eta, xgb_max_depth, xgb_threads)

  ## Compute nuisance terms on test data and compile output into data frame
  print("Computing nuisance terms...")
  data_nuisance <- data_test %>% dplyr::select(all_of(c(Y_var, "treatment", "observe")))
  x             <- dplyr::select(data_test, all_of(X_vars))
  x_copy        <- x
  s             <- dplyr::select(data_test, all_of(S_vars))
  s_x           <- dplyr::select(data_test, all_of(c(S_vars, X_vars)))
  y             <- dplyr::select(data_test, all_of(Y_var))
  y <- as.vector(y[[1]])
  counter <<- 0
  total <<- nrow(data_test)
  print("h_1_s_x_y")
  data_nuisance$h_1_s_x_y <- mapply(h_s_x_y, S = split(as.matrix(s), row(s)),
                                    X = split(as.matrix(x), row(x)),
                                    Y = split(y, seq_along(y)),
                                    MoreArgs = list(W = 1,
                                    varrho_s_x_out = nuisance$varrho_s_x,
                                    data_train = data_train,
                                    S_vars = S_vars, X_vars = X_vars,
                                    Y_var = Y_var, type_prop = type_prop,
                                    nuisance_cv_fold = nuisance_cv_fold,
                                    prop_ub = prop_ub, prop_lb = prop_lb,
                                    integralMaxEval = integralMaxEval,
                                    cop = cop, param = param))
  counter <<- 0
  print("h_0_s_x_y")
  data_nuisance$h_0_s_x_y <- mapply(h_s_x_y, S = split(as.matrix(s), row(s)),
                                    X = split(as.matrix(x), row(x)),
                                    Y = split(y, seq_along(y)),
                                    MoreArgs = list(W = 0,
                                    varrho_s_x_out = nuisance$varrho_s_x,
                                    data_train = data_train,
                                    S_vars = S_vars, X_vars = X_vars,
                                    Y_var = Y_var, type_prop = type_prop,
                                    nuisance_cv_fold = nuisance_cv_fold,
                                    prop_ub = prop_ub, prop_lb = prop_lb,
                                    integralMaxEval = integralMaxEval,
                                    cop = cop, param = param))
  counter <<- 0
  print("mu_1_s_x")
  curr_time <<- Sys.time()
  data_nuisance$mu_1_s_x <- mapply(mu_s_x_copula, S = split(as.matrix(s), row(s)),
                                  X = split(as.matrix(x), row(x)),
                                  MoreArgs = list(W = 1,
                                  varrho_s_x_out = nuisance$varrho_s_x,
                                  data_train = data_train,
                                  S_vars = S_vars, X_vars = X_vars,
                                  Y_var = Y_var, type_prop = type_prop,
                                  prop_ub = prop_ub, prop_lb = prop_lb,
                                  integralMaxEval = integralMaxEval,
                                  cop = cop, param = param))
  counter <<- 0
  print("mu_0_s_x")
  curr_time <<- Sys.time()
  data_nuisance$mu_0_s_x <- mapply(mu_s_x_copula, S = split(as.matrix(s), row(s)),
                                  X = split(as.matrix(x), row(x)),
                                  MoreArgs = list(W = 0,
                                                  varrho_s_x_out = nuisance$varrho_s_x,
                                                  data_train = data_train,
                                                  S_vars = S_vars, X_vars = X_vars,
                                                  Y_var = Y_var, type_prop = type_prop,
                                                  prop_ub = prop_ub, prop_lb = prop_lb,
                                                  integralMaxEval = integralMaxEval,
                                                  cop = cop, param = param))
  counter <<- 0
  print("d_s_x")
  data_nuisance$d_s_x <- mapply(d_s_x, S = split(as.matrix(s), row(s)),
                                  X = split(as.matrix(x), row(x)),
                                  MoreArgs = list(varrho_s_x_out = nuisance$varrho_s_x,
                                                  data_train = data_train,
                                                  S_vars = S_vars, X_vars = X_vars,
                                                  Y_var = Y_var, type_prop = type_prop,
                                                  prop_ub = prop_ub, prop_lb = prop_lb,
                                                  integralMaxEval = integralMaxEval,
                                                  cop = cop, param = param))
  # print("bar_mu_1_x and bar_mu_0_x")
  if(type == "glmnet"){
    if (ncol(x)==1) {
      x <- cbind(0,as.matrix(x))
    }
    data_nuisance$bar_mu_1_x <- if (is.numeric(nuisance$bar_mu_1_x) && length(nuisance$bar_mu_1_x) == 1) {
      rep(nuisance$bar_mu_1_x, nrow(x))
    } else {
      predict(nuisance$bar_mu_1_x, newx = as.matrix(x), s = nuisance$bar_mu_1_x$lambda.min)
    }

    data_nuisance$bar_mu_0_x <- if (is.numeric(nuisance$bar_mu_0_x) && length(nuisance$bar_mu_0_x) == 1) {
      rep(nuisance$bar_mu_0_x, nrow(x))
    } else {
      predict(nuisance$bar_mu_0_x, newx = as.matrix(x), s = nuisance$bar_mu_0_x$lambda.min)
    }
  } else if(type == "grf"){
    data_nuisance$bar_mu_1_x <- if (is.numeric(nuisance$bar_mu_1_x) && length(nuisance$bar_mu_1_x) == 1) {
      rep(nuisance$bar_mu_1_x, nrow(x))
    } else {
      predict(nuisance$bar_mu_1_x, newdata = as.matrix(x))$predictions
    }

    data_nuisance$bar_mu_0_x <- if (is.numeric(nuisance$bar_mu_0_x) && length(nuisance$bar_mu_0_x) == 1) {
      rep(nuisance$bar_mu_0_x, nrow(x))
    } else {
      predict(nuisance$bar_mu_0_x, newdata = as.matrix(x))$predictions
    }
  } else if(type == "xgboost"){
    data_nuisance$bar_mu_1_x <- if (is.numeric(nuisance$bar_mu_1_x) && length(nuisance$bar_mu_1_x) == 1) {
      rep(nuisance$bar_mu_1_x, nrow(x))
    } else {
      predict(nuisance$bar_mu_1_x, newdata = as.matrix(x))
    }

    data_nuisance$bar_mu_0_x <- if (is.numeric(nuisance$bar_mu_0_x) && length(nuisance$bar_mu_0_x) == 1) {
      rep(nuisance$bar_mu_0_x, nrow(x))
    } else {
      predict(nuisance$bar_mu_0_x, newdata = as.matrix(x))
    }
  }

  # print("varrho_x, varrho_s_x, phi_x, phi_s_x, and phi")
  if (type_prop == "glmnet") {
    if (ncol(x)==1) {
      x <- cbind(0,as.matrix(x))
    }
    data_nuisance$varrho_x     <- pmax(pmin(predict(nuisance$varrho_x,     newx = as.matrix(x),   s = nuisance$varrho_x$lambda.min,     type = "response"), prop_ub), prop_lb)
    data_nuisance$varrho_s_x   <- pmax(pmin(predict(nuisance$varrho_s_x,   newx = as.matrix(s_x), s = nuisance$varrho_s_x$lambda.min,   type = "response"), prop_ub), prop_lb)
    data_nuisance$phi_x      <- pmax(pmin(predict(nuisance$phi_x,      newx = as.matrix(x),   s = nuisance$phi_x$lambda.min,      type = "response"), prop_ub), prop_lb)
    data_nuisance$phi_s_x    <- pmax(pmin(predict(nuisance$phi_s_x,    newx = as.matrix(s_x), s = nuisance$phi_s_x$lambda.min,    type = "response"), prop_ub), prop_lb)
  } else if (type_prop == "grf") {
    data_nuisance$varrho_x   <- pmax(pmin(predict(nuisance$varrho_x,     newdata = as.matrix(x_copy))$predictions,   prop_ub), prop_lb)
    data_nuisance$varrho_s_x <- pmax(pmin(predict(nuisance$varrho_s_x,   newdata = as.matrix(s_x))$predictions, prop_ub), prop_lb)
    data_nuisance$phi_x    <- pmax(pmin(predict(nuisance$phi_x,      newdata = as.matrix(x_copy))$predictions,   prop_ub), prop_lb)
    data_nuisance$phi_s_x  <- pmax(pmin(predict(nuisance$phi_s_x,    newdata = as.matrix(s_x))$predictions,  prop_ub), prop_lb)
  } else if (type_prop == "xgboost") {
    data_nuisance$varrho_x   <- pmax(pmin(predict(nuisance$varrho_x,     newdata = as.matrix(x_copy)),   prop_ub), prop_lb)
    data_nuisance$varrho_s_x <- pmax(pmin(predict(nuisance$varrho_s_x,   newdata = as.matrix(s_x)), prop_ub), prop_lb)
    data_nuisance$phi_x    <- pmax(pmin(predict(nuisance$phi_x,      newdata = as.matrix(x_copy)),   prop_ub), prop_lb)
    data_nuisance$phi_s_x  <- pmax(pmin(predict(nuisance$phi_s_x,    newdata = as.matrix(s_x)),  prop_ub), prop_lb)
  }
  data_nuisance$phi    <- 1 - mean(data_train$observe)
  return(data_nuisance)
}

train_nuisance_copula <- function(data_train, S_vars, X_vars, Y_var, type, type_prop, cop, param,
                           prop_lb, prop_ub, integralMaxEval, nuisance_cv_fold,
                           grf_honesty, grf_tune_parameters, grf_num_threads,
                           xgb_cv_rounds, xgb_eta, xgb_max_depth, xgb_threads){
  #### Train nuisance parameters
  ### Train Propensity Scores
  print("Training propensity score models...")
  varrho_x_out   <- varrho_x(data_train, X_vars, type_prop,
                             nuisance_cv_fold, grf_honesty, grf_tune_parameters, grf_num_threads,
                             xgb_cv_rounds, xgb_eta, xgb_max_depth, xgb_threads)
  varrho_s_x_out <- varrho_s_x(data_train, X_vars, S_vars, type_prop,
                               nuisance_cv_fold, grf_honesty, grf_tune_parameters, grf_num_threads,
                               xgb_cv_rounds, xgb_eta, xgb_max_depth, xgb_threads)
  phi_x_out    <- phi_x(data_train, X_vars, type_prop,
                        nuisance_cv_fold, grf_honesty, grf_tune_parameters, grf_num_threads,
                        xgb_cv_rounds, xgb_eta, xgb_max_depth, xgb_threads)
  phi_s_x_out  <- phi_s_x(data_train, X_vars, S_vars, type_prop,
                          nuisance_cv_fold, grf_honesty, grf_tune_parameters, grf_num_threads,
                          xgb_cv_rounds, xgb_eta, xgb_max_depth, xgb_threads)

  ### Train WSI
  ## \bar{r}(x)
  print("Training bar_mu models...")
  print("bar_mu_1_x")
  bar_mu_1_x_out <- bar_mu_x_copula(nuisance_type = type, W = 1, data_train, S_vars, X_vars, Y_var,
                           nuisance_cv_fold, type_prop, prop_ub, prop_lb, integralMaxEval,
                           cop, param, grf_honesty, grf_tune_parameters, grf_num_threads,
                           xgb_cv_rounds, xgb_eta, xgb_max_depth, xgb_threads)
  print("bar_mu_0_x")
  bar_mu_0_x_out <- bar_mu_x_copula(nuisance_type = type, W = 0, data_train, S_vars, X_vars, Y_var,
                           nuisance_cv_fold, type_prop, prop_ub, prop_lb, integralMaxEval,
                           cop, param, grf_honesty, grf_tune_parameters, grf_num_threads,
                           xgb_cv_rounds, xgb_eta, xgb_max_depth, xgb_threads)

  ### Package nuisance estimates
  return(list(bar_mu_1_x = bar_mu_1_x_out,
              bar_mu_0_x = bar_mu_0_x_out,
              varrho_x = varrho_x_out, varrho_s_x = varrho_s_x_out,
              phi_x  = phi_x_out,  phi_s_x  = phi_s_x_out))
}

compute_tau_copula <- function(data_nuisance){
  m <- data_nuisance %>%
    dplyr::mutate(m_0_a = ((1-observe) / phi)*((treatment/varrho_x)*(mu_1_s_x - bar_mu_1_x)
                                         - ((1-treatment)/(1-varrho_x))*(mu_0_s_x - bar_mu_0_x)),
           m_0_b = ((1-observe) / phi)*(bar_mu_1_x - bar_mu_0_x),
           m_0_c = (observe / phi)*(phi_s_x / (1-phi_s_x))*( (varrho_s_x/varrho_x)*(h_1_s_x_y - mu_1_s_x) - ((1-varrho_s_x)/(1-varrho_x))*(h_0_s_x_y - mu_0_s_x) ),
           m_0_d = ((1-observe) / phi)*(1/varrho_x)*(d_s_x-mu_1_s_x)*(treatment - varrho_s_x),
           m_0_e = ((1-observe) / phi)*(1/(1-varrho_x))*(d_s_x-mu_0_s_x)*(treatment - varrho_s_x),
           m_0 = m_0_a + m_0_b + m_0_c + m_0_d + m_0_e) %>%
    dplyr::select(m_0)
  multiplier <- mean((1-data_nuisance$observe))/data_nuisance$phi[1]  # phi column is constant, take the first
  return(mean(m$m_0)/multiplier)
}

compute_m_table_copula <- function(tau_0, data_nuisance){
  m <- data_nuisance %>%
    dplyr::mutate(m_0_a = ((1-observe) / phi)*((treatment/varrho_x)*(mu_1_s_x - bar_mu_1_x)
                                        - ((1-treatment)/(1-varrho_x))*(mu_0_s_x - bar_mu_0_x)),
           m_0_b = ((1-observe) / phi)*(bar_mu_1_x - bar_mu_0_x - tau_0),
           m_0_c = (observe / phi)*(phi_s_x / (1-phi_s_x))*( (varrho_s_x/varrho_x)*(h_1_s_x_y - mu_1_s_x) - ((1-varrho_s_x)/(1-varrho_x))*(h_0_s_x_y - mu_0_s_x) ),
           m_0_d = ((1-observe) / phi)*(1/varrho_x)*(d_s_x-mu_1_s_x)*(treatment - varrho_s_x),
           m_0_e = ((1-observe) / phi)*(1/(1-varrho_x))*(d_s_x-mu_0_s_x)*(treatment - varrho_s_x),
           m_0 = m_0_a + m_0_b + m_0_c + m_0_d + m_0_e) %>%
    return()
}

compute_ci_copula <- function(hat_tau, data_nuisance, alpha){
  ### Compute (1-alpha)% confidence intervals for tau
  influence  <- compute_m_table_copula(hat_tau, data_nuisance)$m_0
  hat_V <- var(influence)
  ci_l <- hat_tau - qnorm(1-alpha/2, 0, 1)*sqrt(hat_V/length(influence))
  ci_u <- hat_tau + qnorm(1-alpha/2, 0, 1)*sqrt(hat_V/length(influence))
  return(list(se = sqrt(hat_V/length(influence)), ci = c(ci_l, ci_u)))
}
