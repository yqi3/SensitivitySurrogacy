.validate_longterm_partial_id_inputs <- function(
    data, S_vars, X_vars, Y_var, type, type_prop,
    prop_lb, prop_ub, alpha,
    cross_fit_fold, nuisance_cv_fold,
    grf_honesty, grf_tune_parameters, grf_num_threads,
    xgb_cv_rounds, xgb_eta, xgb_max_depth, xgb_threads
) {
  if (!is.data.frame(data)) {
    stop("`data` must be a data.frame.")
  }
  if (!is.character(S_vars) || length(S_vars) < 1L) {
    stop("`S_vars` must be a non-empty character vector of column names.")
  }
  if (!is.character(X_vars) || length(X_vars) < 1L) {
    stop("`X_vars` must be a non-empty character vector of column names.")
  }
  if (!is.character(Y_var) || length(Y_var) != 1L) {
    stop("`Y_var` must be a single character string naming the outcome column.")
  }
  required_cols <- unique(c("treatment", "observe", S_vars, X_vars, Y_var))
  missing_cols <- setdiff(required_cols, names(data))
  if (length(missing_cols) > 0L) {
    stop(
      "The following required columns are missing from `data`: ",
      paste(missing_cols, collapse = ", ")
    )
  }
  if (length(intersect(S_vars, X_vars)) > 0L) {
    stop("`S_vars` and `X_vars` must not overlap.")
  }
  if (!all(data$treatment %in% c(0, 1))) {
    stop("`treatment` must contain only 0/1 values.")
  }
  if (!all(data$observe %in% c(0, 1))) {
    stop("`observe` must contain only 0/1 values.")
  }
  valid_type <- c("glmnet", "grf", "xgboost")
  if (!(type %in% valid_type)) {
    stop("`type` must be one of: ", paste(valid_type, collapse = ", "))
  }
  if (!(type_prop %in% valid_type)) {
    stop("`type_prop` must be one of: ", paste(valid_type, collapse = ", "))
  }
  if (!is.numeric(prop_lb) || length(prop_lb) != 1L || !is.finite(prop_lb) ||
      !is.numeric(prop_ub) || length(prop_ub) != 1L || !is.finite(prop_ub) ||
      prop_lb <= 0 || prop_ub >= 1 || prop_lb >= prop_ub) {
    stop("`prop_lb` and `prop_ub` must satisfy 0 < prop_lb < prop_ub < 1.")
  }
  if (!is.numeric(alpha) || length(alpha) != 1L || !is.finite(alpha) ||
      alpha <= 0 || alpha >= 1) {
    stop("`alpha` must be a scalar in (0, 1).")
  }
  if (!is.numeric(cross_fit_fold) || length(cross_fit_fold) != 1L ||
      cross_fit_fold != as.integer(cross_fit_fold) || cross_fit_fold < 2L) {
    stop("`cross_fit_fold` must be an integer >= 2.")
  }
  if (!is.numeric(nuisance_cv_fold) || length(nuisance_cv_fold) != 1L ||
      nuisance_cv_fold != as.integer(nuisance_cv_fold) || nuisance_cv_fold < 2L) {
    stop("`nuisance_cv_fold` must be an integer >= 2.")
  }
  if (nrow(data) < cross_fit_fold) {
    stop("`cross_fit_fold` cannot exceed the number of rows in `data`.")
  }
  if (sum(data$observe == 0) == 0L || sum(data$observe == 1) == 0L) {
    stop("`data` must contain both experimental (`observe = 0`) and observational (`observe = 1`) samples.")
  }
  obs_data <- data[data$observe == 0, , drop = FALSE]
  if (length(unique(obs_data$treatment)) < 2L) {
    stop("The experimental sample (`observe = 0`) must contain both treatment arms 0 and 1.")
  }
  invisible(TRUE)
}

.validate_longterm_copula_inputs <- function(
    data, S_vars, X_vars, Y_var, type, type_prop, cop, param,
    prop_lb, prop_ub, alpha, integralMaxEval,
    cross_fit_fold, nuisance_cv_fold
) {
  if (!is.data.frame(data)) stop("`data` must be a data.frame.")
  required_cols <- unique(c("treatment", "observe", S_vars, X_vars, Y_var))
  missing_cols <- setdiff(required_cols, names(data))
  if (length(missing_cols) > 0) {
    stop("Missing required columns in `data`: ", paste(missing_cols, collapse = ", "))
  }
  if (length(Y_var) != 1L) stop("`Y_var` must be a single column name.")
  if (length(intersect(S_vars, X_vars)) > 0) stop("`S_vars` and `X_vars` must not overlap.")
  if (!all(data$treatment %in% c(0, 1))) stop("`treatment` must contain only 0/1 values.")
  if (!all(data$observe %in% c(0, 1))) stop("`observe` must contain only 0/1 values.")
  valid_type <- c("glmnet", "grf", "xgboost")
  if (!(type %in% valid_type)) stop("`type` must be one of: ", paste(valid_type, collapse = ", "))
  if (!(type_prop %in% valid_type)) stop("`type_prop` must be one of: ", paste(valid_type, collapse = ", "))
  valid_cop <- c("Frank", "Plackett")
  if (!(cop %in% valid_cop)) stop("`cop` must be either 'Frank' or 'Plackett'.")
  if (cop == "Frank") {
    if (!is.numeric(param) || length(param) != 1L || !is.finite(param) || abs(param) < 1e-10) {
      stop("For `cop = 'Frank'`, `param` must be finite and not too close to 0.")
    }
  }
  if (cop == "Plackett") {
    if (!is.numeric(param) || length(param) != 1L || !is.finite(param) || param <= 0) {
      stop("For `cop = 'Plackett'`, `param` must be a finite positive scalar.")
    }
  }
  if (!is.numeric(prop_lb) || !is.numeric(prop_ub) ||
      length(prop_lb) != 1L || length(prop_ub) != 1L ||
      !is.finite(prop_lb) || !is.finite(prop_ub) ||
      prop_lb <= 0 || prop_ub >= 1 || prop_lb >= prop_ub) {
    stop("`prop_lb` and `prop_ub` must satisfy 0 < prop_lb < prop_ub < 1.")
  }
  if (!is.numeric(alpha) || length(alpha) != 1L || !is.finite(alpha) ||
      alpha <= 0 || alpha >= 1) {
    stop("`alpha` must be a scalar in (0, 1).")
  }
  if (!is.numeric(cross_fit_fold) || cross_fit_fold < 2 || cross_fit_fold != as.integer(cross_fit_fold)) {
    stop("`cross_fit_fold` must be an integer >= 2.")
  }
  if (!is.numeric(nuisance_cv_fold) || nuisance_cv_fold < 2 || nuisance_cv_fold != as.integer(nuisance_cv_fold)) {
    stop("`nuisance_cv_fold` must be an integer >= 2.")
  }
  if (!is.numeric(integralMaxEval) || integralMaxEval < 2 || integralMaxEval != as.integer(integralMaxEval)) {
    stop("`integralMaxEval` must be an integer >= 2.")
  }
  if (nrow(data) < cross_fit_fold) {
    stop("`cross_fit_fold` cannot exceed the number of rows in `data`.")
  }
  if (sum(data$observe == 0) == 0 || sum(data$observe == 1) == 0) {
    stop("`data` must contain both observational and experimental samples.")
  }
  invisible(TRUE)
}
