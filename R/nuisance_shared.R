varrho_x <- function(data_train, X_vars, type_prop,
                     nuisance_cv_fold, grf_honesty,
                     grf_tune_parameters, grf_num_threads,
                     xgb_cv_rounds, xgb_eta, xgb_max_depth, xgb_threads){
  # Conditional treatment propensity
  df <- dplyr::filter(data_train, observe == 0)
  x <- dplyr::select(df, all_of(X_vars))
  w <- dplyr::select(df, all_of("treatment"))
  if (length(unique(as.numeric(w[[1]]))) < 2L) {
    stop("varrho_x: `treatment` must contain both 0 and 1.")
  }
  if (type_prop == "glmnet"){
    if (ncol(x)==1) {
      x <- cbind(0,as.matrix(x))
    }
    varrho_x <- glmnet::cv.glmnet(x = as.matrix(x), y = as.matrix(w), family = "binomial", nfolds = nuisance_cv_fold)
  } else if(type_prop == "grf"){
    varrho_x <- grf::regression_forest(X = as.matrix(x), Y = as.matrix(w),
                                       num.threads = grf_num_threads, ci.group.size = 1,
                                       honesty = grf_honesty, tune.parameters = grf_tune_parameters)
  } else if(type_prop == "xgboost"){
    dtrain <- xgboost::xgb.DMatrix(data = as.matrix(x), label = as.numeric(as.matrix(w)))
    cv <- xgboost::xgb.cv(
      data = dtrain,
      params = list(objective = "binary:logistic", eval_metric = "logloss",
                    max_depth = xgb_max_depth, eta = xgb_eta, nthread = xgb_threads),
      nrounds = xgb_cv_rounds, verbose = FALSE, nfold = nuisance_cv_fold)
    varrho_x <- xgboost::xgb.train(
      data = dtrain,
      params = list(objective = "binary:logistic", eval_metric = "logloss",
                    max_depth = xgb_max_depth, eta = xgb_eta, nthread = xgb_threads),
      nrounds = which.min(cv$evaluation_log$test_logloss_mean),
      verbose = FALSE)
  } else {
    stop('Enter a valid nuisance parameter estimation type')
  }
  return(varrho_x)
}

varrho_s_x <- function(data_train, X_vars, S_vars, type_prop,
                       nuisance_cv_fold, grf_honesty,
                       grf_tune_parameters, grf_num_threads,
                       xgb_cv_rounds, xgb_eta, xgb_max_depth, xgb_threads){
  # Conditional treatment propensity
  df <- dplyr::filter(data_train, observe == 0)
  s_x <- dplyr::select(df, all_of(c(S_vars, X_vars)))
  w <- dplyr::select(df, all_of("treatment"))
  if (length(unique(as.numeric(w[[1]]))) < 2L) {
    stop("varrho_s_x: `treatment` must contain both 0 and 1.")
  }
  if(type_prop == "glmnet"){
    varrho_s_x <- glmnet::cv.glmnet(x = as.matrix(s_x), y = as.matrix(w), family = "binomial", nfolds = nuisance_cv_fold)
  } else if(type_prop == "grf"){
    varrho_s_x <- grf::regression_forest(X = as.matrix(s_x), Y = as.matrix(w), num.threads = grf_num_threads,
                                         ci.group.size = 1, honesty = grf_honesty, tune.parameters = grf_tune_parameters)
  } else if(type_prop == "xgboost"){
    dtrain <- xgboost::xgb.DMatrix(data = as.matrix(s_x), label = as.numeric(as.matrix(w)))
    cv <- xgboost::xgb.cv(
      data = dtrain,
      params = list(objective = "binary:logistic", eval_metric = "logloss",
                    max_depth = xgb_max_depth, eta = xgb_eta, nthread = xgb_threads),
      nrounds = xgb_cv_rounds, verbose = FALSE, nfold = nuisance_cv_fold)
    varrho_s_x <- xgboost::xgb.train(
      data = dtrain,
      params = list(objective = "binary:logistic", eval_metric = "logloss",
                    max_depth = xgb_max_depth, eta = xgb_eta, nthread = xgb_threads),
      nrounds = which.min(cv$evaluation_log$test_logloss_mean),
      verbose = FALSE)
  } else {
    stop('Enter a valid nuisance parameter estimation type')
  }
  return(varrho_s_x)
}

phi_x <- function(data_train, X_vars, type_prop,
                  nuisance_cv_fold, grf_honesty,
                  grf_tune_parameters, grf_num_threads,
                  xgb_cv_rounds, xgb_eta, xgb_max_depth, xgb_threads){
  # Conditional observational sample propensity
  x <- dplyr::select(data_train, all_of(X_vars))
  g <- 1-dplyr::select(data_train, observe) # dependent variable is 1[P_i=E]
  if (length(unique(as.numeric(g[[1]]))) < 2L) {
    stop("phi_x: `observe` must contain both 0 and 1.")
  }
  if(type_prop == "glmnet"){
    if (ncol(x)==1) {
      x <- cbind(0,as.matrix(x))
    }
    phi_x <- glmnet::cv.glmnet(x = as.matrix(x), y = as.matrix(g), family = "binomial", nfolds = nuisance_cv_fold)
  } else if(type_prop == "grf"){
    phi_x <- grf::regression_forest(X = as.matrix(x), Y = as.matrix(g), num.threads = grf_num_threads,
                                    ci.group.size = 1, honesty = grf_honesty, tune.parameters = grf_tune_parameters)
  } else if(type_prop == "xgboost"){
    dtrain <- xgboost::xgb.DMatrix(data = as.matrix(x), label = as.numeric(as.matrix(g)))
    cv <- xgboost::xgb.cv(
      data = dtrain,
      params = list(objective = "binary:logistic", eval_metric = "logloss",
                    max_depth = xgb_max_depth, eta = xgb_eta, nthread = xgb_threads),
      nrounds = xgb_cv_rounds, verbose = FALSE, nfold = nuisance_cv_fold)
    phi_x <- xgboost::xgb.train(
      data = dtrain,
      params = list(objective = "binary:logistic", eval_metric = "logloss",
                    max_depth = xgb_max_depth, eta = xgb_eta, nthread = xgb_threads),
      nrounds = which.min(cv$evaluation_log$test_logloss_mean),
      verbose = FALSE)
  } else {
    stop('Enter a valid nuisance parameter estimation type')
  }
  return(phi_x)
}

phi_s_x <- function(data_train, X_vars, S_vars, type_prop,
                    nuisance_cv_fold, grf_honesty,
                    grf_tune_parameters, grf_num_threads,
                    xgb_cv_rounds, xgb_eta, xgb_max_depth, xgb_threads){
  # Conditional observational sample propensity
  s_x <- dplyr::select(data_train, all_of(c(S_vars, X_vars)))
  g <- 1-dplyr::select(data_train, observe) # dependent variable is 1[P_i=E]
  if (length(unique(as.numeric(g[[1]]))) < 2L) {
    stop("phi_s_x: `observe` must contain both 0 and 1.")
  }
  if(type_prop == "glmnet"){
    phi_s_x <- glmnet::cv.glmnet(x = as.matrix(s_x), y = as.matrix(g), family = "binomial", nfolds = nuisance_cv_fold)
  } else if(type_prop == "grf"){
    phi_s_x <- grf::regression_forest(X = as.matrix(s_x), Y = as.matrix(g), num.threads = grf_num_threads,
                                      ci.group.size = 1, honesty = grf_honesty, tune.parameters = grf_tune_parameters)
  } else if(type_prop == "xgboost"){
    dtrain <- xgboost::xgb.DMatrix(data = as.matrix(s_x), label = as.numeric(as.matrix(g)))
    cv <- xgboost::xgb.cv(
      data = dtrain,
      params = list(objective = "binary:logistic", eval_metric = "logloss",
                    max_depth = xgb_max_depth, eta = xgb_eta, nthread = xgb_threads),
      nrounds = xgb_cv_rounds, verbose = FALSE, nfold = nuisance_cv_fold)
    phi_s_x <- xgboost::xgb.train(
      data = dtrain,
      params = list(objective = "binary:logistic", eval_metric = "logloss",
                    max_depth = xgb_max_depth, eta = xgb_eta, nthread = xgb_threads),
      nrounds = which.min(cv$evaluation_log$test_logloss_mean),
      verbose = FALSE)
  } else {
    stop('Enter a valid nuisance parameter estimation type')
  }
  return(phi_s_x)
}