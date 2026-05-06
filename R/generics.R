# Generic functions re-exported or defined here so that S3 methods can be
# registered without requiring 'loo' to be a hard dependency.

#' Pointwise log-likelihood matrix
#'
#' Generic function. See \code{\link{log_lik.smoothbp_fit}} for the
#' \code{smoothbp_fit} method.
#'
#' @param object A fitted model object.
#' @param ... Additional arguments passed to methods.
#' @export
log_lik <- function(object, ...) UseMethod("log_lik")

#' Leave-one-out cross-validation
#'
#' Generic function. See \code{\link{loo.smoothbp_fit}} for the
#' \code{smoothbp_fit} method.
#'
#' @param x A fitted model object.
#' @param ... Additional arguments passed to methods.
#' @export
loo <- function(x, ...) UseMethod("loo")

#' Widely applicable information criterion (WAIC)
#'
#' Generic function. See \code{\link{waic.smoothbp_fit}} for the
#' \code{smoothbp_fit} method.
#'
#' @param x A fitted model object.
#' @param ... Additional arguments passed to methods.
#' @export
waic <- function(x, ...) UseMethod("waic")

#' Posterior predictive check
#'
#' Generic function. See \code{\link{pp_check.smoothbp_fit}} for the
#' \code{smoothbp_fit} method.
#'
#' @param object A fitted model object.
#' @param ... Additional arguments passed to methods.
#' @export
pp_check <- function(object, ...) UseMethod("pp_check")
