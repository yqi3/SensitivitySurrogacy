tau_plackett <- function(theta) {
  if (!is.numeric(theta) || length(theta) != 1L || !is.finite(theta) || theta <= 0) {
    stop("`theta` must be a finite positive scalar for the Plackett copula.")
  }

  if (abs(theta - 1) < 1e-8) {
    return(0)
  }

  1 + 2 / (theta - 1) - (2 * theta * log(theta)) / (theta - 1)^2
}

#' Convert Kendall's Tau to a Plackett Copula Parameter
#'
#' This helper function maps a user-specified Kendall's tau value to the
#' corresponding parameter of the Plackett copula. It is intended to simplify
#' sensitivity analysis with \code{\link{longterm_copula}}, since Kendall's tau
#' provides a more interpretable dependence measure than the raw copula
#' parameter.
#'
#' The function solves the nonlinear relationship between Kendall's tau and the
#' Plackett copula parameter by numerical root finding.
#'
#' @param tau_target A scalar Kendall's tau value. For the Plackett copula,
#' this must lie strictly between \eqn{-1} and \eqn{1}.
#'
#' @param tol Positive numeric tolerance passed to \code{\link[stats]{uniroot}}
#' when solving for the copula parameter.
#'
#' @return A positive numeric scalar giving the Plackett copula parameter
#' corresponding to \code{tau_target}. When \code{tau_target} is sufficiently
#' close to zero, the function returns \code{1}, which corresponds to the
#' independence case.
#'
#' @examples
#' tau0 <- 0.3
#' theta_plackett <- plackett_param_from_tau(tau0)
#' theta_plackett
#'
#' @export
#'
#' @seealso \code{\link{frank_param_from_tau}}, \code{\link{longterm_copula}}

plackett_param_from_tau <- function(tau_target, tol = 1e-6) {
  if (!is.numeric(tau_target) || length(tau_target) != 1L || !is.finite(tau_target)) {
    stop("`tau_target` must be a finite numeric scalar.")
  }
  if (tau_target <= -1 || tau_target >= 1) {
    stop("For the Plackett copula, `tau_target` must lie strictly between -1 and 1.")
  }
  if (!is.numeric(tol) || length(tol) != 1L || !is.finite(tol) || tol <= 0) {
    stop("`tol` must be a positive numeric scalar.")
  }

  if (abs(tau_target) < tol) {
    return(1)
  }

  objective <- function(theta) {
    tau_plackett(theta) - tau_target
  }

  if (tau_target < 0) {
    uniroot(objective, interval = c(1e-8, 1 - 1e-8), tol = tol)$root
  } else {
    uniroot(objective, interval = c(1 + 1e-8, 1e8), tol = tol)$root
  }
}

#' Convert Kendall's Tau to a Frank Copula Parameter
#'
#' This helper function maps a user-specified Kendall's tau value to the
#' corresponding parameter of the Frank copula. It is intended to simplify
#' sensitivity analysis with \code{\link{longterm_copula}}, since Kendall's tau
#' provides a more interpretable dependence measure than the raw copula
#' parameter.
#'
#' The function numerically inverts the relationship between Kendall's tau and
#' the Frank copula parameter. For tau values sufficiently close to zero, the
#' function returns a small nonzero value rather than exactly zero, since the
#' Frank copula parameterization becomes numerically unstable at zero in the
#' downstream estimation routines.
#'
#' @param tau_target A scalar Kendall's tau value. For the Frank copula, this
#' must lie strictly between \eqn{-1} and \eqn{1}.
#'
#' @param tol Positive numeric tolerance used both for checking whether
#' \code{tau_target} is close to zero and for numerical root finding.
#'
#' @param independence_buffer Positive numeric scalar returned when
#' \code{tau_target} is sufficiently close to zero. This provides a numerical
#' approximation to the independence case while avoiding an exact zero copula
#' parameter.
#'
#' @return A numeric scalar giving the Frank copula parameter corresponding to
#' \code{tau_target}. If \code{tau_target} is sufficiently close to zero, the
#' returned value is \code{independence_buffer}.
#'
#' @examples
#' tau0 <- 0.3
#' theta_frank <- frank_param_from_tau(tau0)
#' theta_frank
#'
#' @export
#'
#' @seealso \code{\link{plackett_param_from_tau}}, \code{\link{longterm_copula}}

frank_param_from_tau <- function(tau_target, tol = 1e-6, independence_buffer = 1e-6) {
  if (!is.numeric(tau_target) || length(tau_target) != 1L || !is.finite(tau_target)) {
    stop("`tau_target` must be a finite numeric scalar.")
  }
  if (tau_target <= -1 || tau_target >= 1) {
    stop("For the Frank copula, `tau_target` must lie strictly between -1 and 1.")
  }
  if (!is.numeric(tol) || length(tol) != 1L || !is.finite(tol) || tol <= 0) {
    stop("`tol` must be a positive numeric scalar.")
  }
  if (!is.numeric(independence_buffer) || length(independence_buffer) != 1L ||
      !is.finite(independence_buffer) || independence_buffer <= 0) {
    stop("`independence_buffer` must be a positive numeric scalar.")
  }

  if (abs(tau_target) < tol) {
    return(independence_buffer)
  }

  debye <- function(theta) {
    if (theta == 0) return(1)
    integrate(function(t) t / (exp(t) - 1), lower = 0, upper = theta)$value / theta
  }

  objective <- function(theta) {
    tau_target - (1 - (4 / theta) * (1 - debye(theta)))
  }

  uniroot(objective, interval = c(-500, 500), tol = tol)$root
}
