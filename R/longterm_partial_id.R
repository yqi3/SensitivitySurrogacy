#' Estimate Worst-Case Bounds for Long-Term Treatment Effects
#'
#' This function computes worst-case (partial identification) bounds for the
#' average treatment effect on a long-term outcome using surrogate variables.
#' The method combines experimental and observational samples and estimates
#' the orthogonal moment equations derived in
#' \cite{Fan, Manzanares, Park, Qi (2026)}.
#'
#' @param data A data frame containing the analysis dataset. The data must
#' include the binary indicators \code{"treatment"} (treatment assignment) and
#' \code{"observe"} (indicator for observational versus experimental sample),
#' along with pre-treatment covariates, short-term outcomes, and the long-term
#' outcome. If certain variables are structurally missing for part of the data
#' (for example, long-term outcomes not observed in the experimental sample),
#' they may be filled with any placeholder value because those observations
#' will not enter the relevant estimation steps.
#'
#' @param S_vars Character vector giving the names of the surrogate
#' (short-term outcome) variables.
#'
#' @param X_vars Character vector giving the names of the pre-treatment
#' covariates.
#'
#' @param Y_var A single character string specifying the long-term outcome
#' variable.
#'
#' @param type Method used to estimate outcome functions entering the moment
#' equations. Supported options are \code{"glmnet"} (cross-validated lasso),
#' \code{"grf"} (generalized random forests), and \code{"xgboost"}.
#'
#' @param type_prop Method used to estimate propensity and surrogacy score
#' nuisance components. Supported options are \code{"glmnet"}, \code{"grf"},
#' and \code{"xgboost"}.
#'
#' @param prop_lb Lower trimming threshold applied to estimated propensity
#' scores.
#'
#' @param prop_ub Upper trimming threshold applied to estimated propensity
#' scores.
#'
#' @param alpha Significance level used when constructing confidence intervals.
#' The resulting intervals have nominal coverage (1-alpha).
#'
#' @param cross_fit_fold Number of folds used for cross-fitting nuisance
#' parameter estimators.
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
#' @return A list with six elements:
#' \describe{
#' \item{\code{hat_tau_upper}}{Estimated upper bound of the average treatment effect.}
#' \item{\code{se_upper}}{Estimated standard error for the upper bound.}
#' \item{\code{ci_upper}}{Two-element vector containing the confidence interval
#' for the upper bound.}
#' \item{\code{hat_tau_lower}}{Estimated lower bound of the average treatment effect.}
#' \item{\code{se_lower}}{Estimated standard error for the lower bound.}
#' \item{\code{ci_lower}}{Two-element vector containing the confidence interval
#' for the lower bound.}
#' }
#'
#' @export
#'
#' @references
#' Fan, Y., Manzanares, C. A., Park, H., & Qi, Y. (2026).
#' A Sensitivity Analysis of the Surrogate Index Approach for Estimating
#' Long-Term Treatment Effects. arXiv:2603.00580.

longterm_partial_id <- function(data, S_vars, X_vars, Y_var, type, type_prop,
                      prop_lb = 0.01, prop_ub = 0.99, alpha = 0.05,
                      cross_fit_fold = 5, nuisance_cv_fold = 5,
                      grf_honesty = TRUE, grf_tune_parameters = "all",
                      grf_num_threads = 1, xgb_cv_rounds = 100, xgb_eta = 0.1,
                      xgb_max_depth = 2, xgb_threads = 1){
  .validate_longterm_partial_id_inputs(
    data = data,
    S_vars = S_vars,
    X_vars = X_vars,
    Y_var = Y_var,
    type = type,
    type_prop = type_prop,
    prop_lb = prop_lb,
    prop_ub = prop_ub,
    alpha = alpha,
    cross_fit_fold = cross_fit_fold,
    nuisance_cv_fold = nuisance_cv_fold,
    grf_honesty = grf_honesty,
    grf_tune_parameters = grf_tune_parameters,
    grf_num_threads = grf_num_threads,
    xgb_cv_rounds = xgb_cv_rounds,
    xgb_eta = xgb_eta,
    xgb_max_depth = xgb_max_depth,
    xgb_threads = xgb_threads
  )
  ### Estimate Long-Term Treatment Effects
  # Choose random k-fold of the data
  folds <- split(seq(nrow(data)), sample(rep(1:cross_fit_fold, length.out = nrow(data))))
  # Compute the nuisance parameters
  data_nuisance_out <- lapply(seq(cross_fit_fold),
                              compute_nuisance_on_fold_partial_id, data,
                              folds, S_vars, X_vars, Y_var, type, type_prop,
                              prop_lb, prop_ub, cross_fit_fold, nuisance_cv_fold,
                              grf_honesty, grf_tune_parameters, grf_num_threads,
                              xgb_cv_rounds, xgb_eta, xgb_max_depth, xgb_threads)
  data_nuisance     <- dplyr::bind_rows(data_nuisance_out)
  # Compute the estimate of tau (upper or lower)
  hat_tau_upper   <- compute_tau_partial_id(data_nuisance, Y_var, "upper")
  hat_tau_lower   <- compute_tau_partial_id(data_nuisance, Y_var, "lower")

  # Compute the standard error and construct confidence intervals for bounds
  se_ci <- compute_bound_ci_partial_id(hat_tau_upper, data_nuisance, Y_var, "upper", alpha)
  se_upper    <- se_ci$se
  ci_upper    <- se_ci$ci

  se_ci <- compute_bound_ci_partial_id(hat_tau_lower, data_nuisance, Y_var, "lower", alpha)
  se_lower    <- se_ci$se
  ci_lower    <- se_ci$ci

  return(list(hat_tau_upper = hat_tau_upper, se_upper = se_upper, ci_upper = ci_upper, hat_tau_lower = hat_tau_lower, se_lower = se_lower, ci_lower = ci_lower))
}

compute_nuisance_on_fold_partial_id <- function(i, data, folds, S_vars, X_vars, Y_var, type, type_prop,
                                     prop_lb, prop_ub, cross_fit_fold, nuisance_cv_fold,
                                     grf_honesty, grf_tune_parameters, grf_num_threads,
                                     xgb_cv_rounds, xgb_eta, xgb_max_depth, xgb_threads){
  print(paste0("------Estimating fold ", i, "/", cross_fit_fold, "------"))

  ## Subset to training and testing samples
  data_train <- data[as.numeric(unlist(folds[setdiff(seq(cross_fit_fold), i)])), ]
  data_test  <- data[as.numeric(unlist(folds[i])), ]

  ## Train nuisance parameters
  nuisance <- train_nuisance_partial_id(data_train, S_vars, X_vars, Y_var, type, type_prop,
                             prop_lb, prop_ub, nuisance_cv_fold,
                             grf_honesty, grf_tune_parameters, grf_num_threads,
                             xgb_cv_rounds, xgb_eta, xgb_max_depth, xgb_threads)

  ## Compute nuisance terms on test data and compile output into data frame
  print("Computing nuisance terms...")
  data_nuisance <- data_test %>% dplyr::select(all_of(c(Y_var, "treatment", "observe")))
  x             <- dplyr::select(data_test, all_of(X_vars))
  x_copy        <- x
  s             <- dplyr::select(data_test, all_of(S_vars))
  s_x           <- dplyr::select(data_test, all_of(c(S_vars, X_vars)))
  data_nuisance$mu_1U_s_x <- mapply(mu_s_x_partial_id, S = split(as.matrix(s), row(s)),
                                    X = split(as.matrix(x), row(x)),
                                    MoreArgs = list(mu_type = "1U",
                                    varrho_s_x_out = nuisance$varrho_s_x,
                                    data_train = data_train,
                                    S_vars = S_vars, X_vars = X_vars,
                                    Y_var = Y_var, type_prop = type_prop,
                                    nuisance_cv_fold = nuisance_cv_fold,
                                    prop_ub = prop_ub, prop_lb = prop_lb))
  data_nuisance$mu_1L_s_x <- mapply(mu_s_x_partial_id, S = split(as.matrix(s), row(s)),
                                    X = split(as.matrix(x), row(x)),
                                    MoreArgs = list(mu_type = "1L",
                                    varrho_s_x_out = nuisance$varrho_s_x,
                                    data_train = data_train,
                                    S_vars = S_vars, X_vars = X_vars,
                                    Y_var = Y_var, type_prop = type_prop,
                                    nuisance_cv_fold = nuisance_cv_fold,
                                    prop_ub = prop_ub, prop_lb = prop_lb))
  data_nuisance$mu_0U_s_x <- mapply(mu_s_x_partial_id, S = split(as.matrix(s), row(s)),
                                    X = split(as.matrix(x), row(x)),
                                    MoreArgs = list(mu_type = "0U",
                                    varrho_s_x_out = nuisance$varrho_s_x,
                                    data_train = data_train,
                                    S_vars = S_vars, X_vars = X_vars,
                                    Y_var = Y_var, type_prop = type_prop,
                                    nuisance_cv_fold = nuisance_cv_fold,
                                    prop_ub = prop_ub, prop_lb = prop_lb))
  data_nuisance$mu_0L_s_x <- mapply(mu_s_x_partial_id, S = split(as.matrix(s), row(s)),
                                    X = split(as.matrix(x), row(x)),
                                    MoreArgs = list(mu_type = "0L",
                                    varrho_s_x_out = nuisance$varrho_s_x,
                                    data_train = data_train,
                                    S_vars = S_vars, X_vars = X_vars,
                                    Y_var = Y_var, type_prop = type_prop,
                                    nuisance_cv_fold = nuisance_cv_fold,
                                    prop_ub = prop_ub, prop_lb = prop_lb))
  data_nuisance$q_U_s_x <- mapply(q_s_x, S = split(as.matrix(s), row(s)),
                                  X = split(as.matrix(x), row(x)),
                                  MoreArgs = list(q_type = "U",
                                  data_train = data_train,
                                  S_vars = S_vars, X_vars = X_vars,
                                  Y_var = Y_var, type_prop = type_prop,
                                  prop_ub = prop_ub, prop_lb = prop_lb,
                                  varrho_s_x_out = nuisance$varrho_s_x))
  data_nuisance$q_L_s_x <- mapply(q_s_x, S = split(as.matrix(s), row(s)),
                                  X = split(as.matrix(x), row(x)),
                                  MoreArgs = list(q_type = "L",
                                  data_train = data_train,
                                  S_vars = S_vars, X_vars = X_vars,
                                  Y_var = Y_var, type_prop = type_prop,
                                  prop_ub = prop_ub, prop_lb = prop_lb,
                                  varrho_s_x_out = nuisance$varrho_s_x))
  if(type == "glmnet"){
    if (ncol(x)==1) {
      x <- cbind(0,as.matrix(x))
    }
    data_nuisance$bar_mu_1U_x <- predict(nuisance$bar_mu_1U_x, newx = as.matrix(x), s = nuisance$bar_mu_1U_x$lambda.min)
    data_nuisance$bar_mu_1L_x <- predict(nuisance$bar_mu_1L_x, newx = as.matrix(x), s = nuisance$bar_mu_1L_x$lambda.min)
    data_nuisance$bar_mu_0U_x <- predict(nuisance$bar_mu_0U_x, newx = as.matrix(x), s = nuisance$bar_mu_0U_x$lambda.min)
    data_nuisance$bar_mu_0L_x <- predict(nuisance$bar_mu_0L_x, newx = as.matrix(x), s = nuisance$bar_mu_0L_x$lambda.min)
  } else if(type == "grf"){
    data_nuisance$bar_mu_1U_x <- predict(nuisance$bar_mu_1U_x, newdata = as.matrix(x))$predictions
    data_nuisance$bar_mu_1L_x <- predict(nuisance$bar_mu_1L_x, newdata = as.matrix(x))$predictions
    data_nuisance$bar_mu_0U_x <- predict(nuisance$bar_mu_0U_x, newdata = as.matrix(x))$predictions
    data_nuisance$bar_mu_0L_x <- predict(nuisance$bar_mu_0L_x, newdata = as.matrix(x))$predictions
  } else if(type == "xgboost"){
    data_nuisance$bar_mu_1U_x <- predict(nuisance$bar_mu_1U_x, newdata = as.matrix(x))
    data_nuisance$bar_mu_1L_x <- predict(nuisance$bar_mu_1L_x, newdata = as.matrix(x))
    data_nuisance$bar_mu_0U_x <- predict(nuisance$bar_mu_0U_x, newdata = as.matrix(x))
    data_nuisance$bar_mu_0L_x <- predict(nuisance$bar_mu_0L_x, newdata = as.matrix(x))
  }

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

train_nuisance_partial_id <- function(data_train, S_vars, X_vars, Y_var, type, type_prop,
                           prop_lb, prop_ub, nuisance_cv_fold,
                           grf_honesty, grf_tune_parameters, grf_num_threads,
                           xgb_cv_rounds, xgb_eta, xgb_max_depth, xgb_threads){
  #### Train nuisance parameters
  ### Train Long-term outcome means
  ## \bar{\mu}(x)
  print("Training outcome models...")
  bar_mu_1U_x_out <- bar_mu_x_partial_id(mu_type = "1U", data_train, treatment_val = 1, S_vars, X_vars, Y_var, type, type_prop, prop_lb, prop_ub, nuisance_cv_fold, grf_honesty, grf_tune_parameters, grf_num_threads, xgb_cv_rounds, xgb_eta, xgb_max_depth, xgb_threads)
  bar_mu_1L_x_out <- bar_mu_x_partial_id(mu_type = "1L", data_train, treatment_val = 1, S_vars, X_vars, Y_var, type, type_prop, prop_lb, prop_ub, nuisance_cv_fold, grf_honesty, grf_tune_parameters, grf_num_threads, xgb_cv_rounds, xgb_eta, xgb_max_depth, xgb_threads)
  bar_mu_0U_x_out <- bar_mu_x_partial_id(mu_type = "0U", data_train, treatment_val = 0, S_vars, X_vars, Y_var, type, type_prop, prop_lb, prop_ub, nuisance_cv_fold, grf_honesty, grf_tune_parameters, grf_num_threads, xgb_cv_rounds, xgb_eta, xgb_max_depth, xgb_threads)
  bar_mu_0L_x_out <- bar_mu_x_partial_id(mu_type = "0L", data_train, treatment_val = 0, S_vars, X_vars, Y_var, type, type_prop, prop_lb, prop_ub, nuisance_cv_fold, grf_honesty, grf_tune_parameters, grf_num_threads, xgb_cv_rounds, xgb_eta, xgb_max_depth, xgb_threads)

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

  ### Package nuisance estimates
  return(list(bar_mu_1U_x = bar_mu_1U_x_out,
              bar_mu_1L_x = bar_mu_1L_x_out,
              bar_mu_0U_x = bar_mu_0U_x_out,
              bar_mu_0L_x = bar_mu_0L_x_out,
              varrho_x = varrho_x_out, varrho_s_x = varrho_s_x_out,
              phi_x  = phi_x_out,  phi_s_x  = phi_s_x_out))
}

compute_tau_partial_id <- function(data_nuisance, Y_var, tau_type){
  if (tau_type == "upper") {
    m <- data_nuisance %>%
      dplyr::mutate(Y = .data[[Y_var]]) %>%
      dplyr::mutate(m_0_a = ((1-observe) / phi)*((treatment/varrho_x)*(mu_1U_s_x - bar_mu_1U_x)
                                          - ((1-treatment)/(1-varrho_x))*(mu_0L_s_x - bar_mu_0L_x)),
             m_0_b = ((1-observe) / phi)*(bar_mu_1U_x - bar_mu_0L_x),
             m_0_c = (observe / phi)*(phi_s_x / (1-phi_s_x))*( (varrho_s_x/varrho_x)*(H_y_s("U", Y, q_U_s_x, varrho_s_x) - mu_1U_s_x) - ((1-varrho_s_x)/(1-varrho_x))*(H_y_s("L", Y, q_U_s_x, (1-varrho_s_x)) - mu_0L_s_x) ),
             m_0_d = ((1-observe) / phi)*(1/varrho_x)*(q_U_s_x - mu_1U_s_x)*(treatment - varrho_s_x),
             m_0_e = ((1-observe) / phi)*(1/(1-varrho_x))*(q_U_s_x - mu_0L_s_x)*(treatment - varrho_s_x),
             m_0 = m_0_a + m_0_b + m_0_c + m_0_d + m_0_e) %>%
      dplyr::select(m_0)
    multiplier <- mean((1-data_nuisance$observe))/data_nuisance$phi[1]  # phi column is constant, take the first
    return(mean(m$m_0)/multiplier)
  } else {
    m <- data_nuisance %>%
      dplyr::mutate(Y = .data[[Y_var]]) %>%
      dplyr::mutate(m_0_a = ((1-observe) / phi)*((treatment/varrho_x)*(mu_1L_s_x - bar_mu_1L_x)
                                          - ((1-treatment)/(1-varrho_x))*(mu_0U_s_x - bar_mu_0U_x)),
             m_0_b = ((1-observe) / phi)*(bar_mu_1L_x - bar_mu_0U_x),
             m_0_c = (observe / phi)*(phi_s_x / (1-phi_s_x))*( (varrho_s_x/varrho_x)*(H_y_s("L", Y, q_L_s_x, varrho_s_x) - mu_1L_s_x) - ((1-varrho_s_x)/(1-varrho_x))*(H_y_s("U", Y, q_L_s_x, (1-varrho_s_x)) - mu_0U_s_x) ),
             m_0_d = ((1-observe) / phi)*(1/varrho_x)*(q_L_s_x - mu_1L_s_x)*(treatment - varrho_s_x),
             m_0_e = ((1-observe) / phi)*(1/(1-varrho_x))*(q_L_s_x - mu_0U_s_x)*(treatment - varrho_s_x),
             m_0 = m_0_a + m_0_b + m_0_c + m_0_d + m_0_e) %>%
      dplyr::select(m_0)
    multiplier <- mean((1-data_nuisance$observe))/data_nuisance$phi[1]  # phi column is constant, take the first
    return(mean(m$m_0)/multiplier)
  }
}

compute_m_table_partial_id <- function(tau_0, data_nuisance, Y_var, tau_type){
  if (tau_type == "upper") {
    m <- data_nuisance %>%
      dplyr::mutate(Y = .data[[Y_var]]) %>%
      dplyr::mutate(m_0_a = ((1-observe) / phi)*((treatment/varrho_x)*(mu_1U_s_x - bar_mu_1U_x)
                                           - ((1-treatment)/(1-varrho_x))*(mu_0L_s_x - bar_mu_0L_x)),
             m_0_b = ((1-observe) / phi)*(bar_mu_1U_x - bar_mu_0L_x - tau_0),
             m_0_c = (observe / phi)*(phi_s_x / (1-phi_s_x))*( (varrho_s_x/varrho_x)*(H_y_s("U", Y, q_U_s_x, varrho_s_x) - mu_1U_s_x) - ((1-varrho_s_x)/(1-varrho_x))*(H_y_s("L", Y, q_U_s_x, (1-varrho_s_x)) - mu_0L_s_x) ),
             m_0_d = ((1-observe) / phi)*(1/varrho_x)*(q_U_s_x - mu_1U_s_x)*(treatment - varrho_s_x),
             m_0_e = ((1-observe) / phi)*(1/(1-varrho_x))*(q_U_s_x - mu_0L_s_x)*(treatment - varrho_s_x),
             m_0 = m_0_a + m_0_b + m_0_c + m_0_d + m_0_e) %>%
      return()
  } else {
    m <- data_nuisance %>%
      dplyr::mutate(Y = .data[[Y_var]]) %>%
      dplyr::mutate(m_0_a = ((1-observe) / phi)*((treatment/varrho_x)*(mu_1L_s_x - bar_mu_1L_x)
                                          - ((1-treatment)/(1-varrho_x))*(mu_0U_s_x - bar_mu_0U_x)),
             m_0_b = ((1-observe) / phi)*(bar_mu_1L_x - bar_mu_0U_x - tau_0),
             m_0_c = (observe / phi)*(phi_s_x / (1-phi_s_x))*( (varrho_s_x/varrho_x)*(H_y_s("L", Y, q_L_s_x, varrho_s_x) - mu_1L_s_x) - ((1-varrho_s_x)/(1-varrho_x))*(H_y_s("U", Y, q_L_s_x, (1-varrho_s_x)) - mu_0U_s_x) ),
             m_0_d = ((1-observe) / phi)*(1/varrho_x)*(q_L_s_x - mu_1L_s_x)*(treatment - varrho_s_x),
             m_0_e = ((1-observe) / phi)*(1/(1-varrho_x))*(q_L_s_x - mu_0U_s_x)*(treatment - varrho_s_x),
             m_0 = m_0_a + m_0_b + m_0_c + m_0_d + m_0_e) %>%
      return()
  }
}

compute_bound_ci_partial_id <- function(hat_tau, data_nuisance, Y_var, tau_type, alpha){
  ### Compute (1-alpha)% confidence intervals for the bounds
  influence  <- compute_m_table_partial_id(hat_tau, data_nuisance, Y_var, tau_type)$m_0
  hat_V <- var(influence)
  ci_l <- hat_tau - qnorm(1-alpha/2, 0, 1)*sqrt(hat_V/length(influence))
  ci_u <- hat_tau + qnorm(1-alpha/2, 0, 1)*sqrt(hat_V/length(influence))
  return(list(se = sqrt(hat_V/length(influence)), ci = c(ci_l, ci_u)))
}
