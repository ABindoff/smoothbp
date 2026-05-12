# Generic functions re-exported or defined here so that S3 methods can be
# registered without requiring 'loo' to be a hard dependency.

#' Pointwise log-likelihood matrix
#'
#' @param object A fitted model object.
#' @param ... Additional arguments passed to methods.
#' @export
log_lik <- function(object, ...) UseMethod("log_lik")

#' @importFrom loo loo
#' @export
loo::loo

#' @importFrom loo waic
#' @export
loo::waic

#' @importFrom bayesplot pp_check
#' @export
bayesplot::pp_check
