#' Specify a normal (or truncated normal) prior for a regression coefficient
#'
#' @param mean Prior mean. Default 0.
#' @param sd Prior standard deviation. Default 1.
#' @param lb Lower bound (use `-Inf` for unconstrained). Default `-Inf`.
#' @param ub Upper bound (use `Inf` for unconstrained). Default `Inf`.
#'
#' @return A `smoothbp_prior` object.
#' @export
prior_normal <- function(mean = 0, sd = 1, lb = -Inf, ub = Inf) {
  stopifnot(sd > 0, lb < ub)
  structure(
    list(family = "normal", mean = mean, sd = sd, lb = lb, ub = ub),
    class = "smoothbp_prior"
  )
}

#' Specify an inverse-gamma prior for a variance component
#'
#' @param shape Shape parameter (> 0).
#' @param scale Scale parameter (> 0).
#'
#' @return A `smoothbp_prior` object.
#' @export
prior_invgamma <- function(shape = 1, scale = 1) {
  stopifnot(shape > 0, scale > 0)
  structure(
    list(family = "invgamma", shape = shape, scale = scale),
    class = "smoothbp_prior"
  )
}

#' @export
print.smoothbp_prior <- function(x, ...) {
  if (x$family == "normal") {
    cat(sprintf("Normal(mean=%g, sd=%g", x$mean, x$sd))
    if (is.finite(x$lb) || is.finite(x$ub)) {
      cat(sprintf(", lb=%s, ub=%s", format(x$lb), format(x$ub)))
    }
    cat(")\n")
  } else if (x$family == "invgamma") {
    cat(sprintf("InvGamma(shape=%g, scale=%g)\n", x$shape, x$scale))
  }
  invisible(x)
}

#' Collect priors for all model parameters
#'
#' Each argument accepts either:
#' - A single `prior_normal()` applied to all coefficients of that parameter, or
#' - A named list mapping coefficient names (matching column names of the design
#'   matrix) to individual `prior_normal()` objects.
#'
#' @param b0     Prior(s) for `b0` regression coefficients.
#' @param b1     Prior(s) for `b1` regression coefficients.
#' @param b2     Prior(s) for `b2` regression coefficients.
#' @param omega  Prior(s) for `omega` regression coefficients.
#' @param rho    Prior(s) for `rho` regression coefficients.
#' @param sigma  `prior_invgamma()` for residual SD.
#' @param sigma_u `prior_invgamma()` for random-effect SD.
#'
#' @return A `smoothbp_priors` list.
#' @export
smoothbp_priors <- function(
    b0      = prior_normal(0, 10),
    b1      = prior_normal(0, 2),
    b2      = prior_normal(0, 2),
    omega   = prior_normal(3, 2, lb = 0),
    rho     = prior_normal(3, 2, lb = 0),
    sigma   = prior_invgamma(1, 1),
    sigma_u = prior_invgamma(1, 1)
) {
  structure(
    list(b0 = b0, b1 = b1, b2 = b2, omega = omega, rho = rho,
         sigma = sigma, sigma_u = sigma_u),
    class = "smoothbp_priors"
  )
}

#' @export
print.smoothbp_priors <- function(x, ...) {
  cat("smoothbp priors:\n")
  for (nm in c("b0", "b1", "b2", "omega", "rho", "sigma", "sigma_u")) {
    cat(sprintf("  %-8s: ", nm))
    print(x[[nm]])
  }
  invisible(x)
}

# ---------------------------------------------------------------------------
# Internal helpers: expand priors to per-coefficient vectors
# ---------------------------------------------------------------------------

# Given a prior spec (single prior_normal or named list) and a vector of
# coefficient names, return a data.frame with mean/sd/lb/ub per coefficient.
.expand_prior <- function(prior_spec, coef_names) {
  n <- length(coef_names)

  if (inherits(prior_spec, "smoothbp_prior") && prior_spec$family == "normal") {
    # Single prior applied to all coefficients
    data.frame(
      name = coef_names,
      mean = prior_spec$mean,
      sd   = prior_spec$sd,
      lb   = prior_spec$lb,
      ub   = prior_spec$ub,
      stringsAsFactors = FALSE
    )
  } else if (is.list(prior_spec) && !inherits(prior_spec, "smoothbp_prior")) {
    # Named list: start with defaults, then override
    default <- prior_spec[["."]] %||% prior_normal(0, 10)
    out <- data.frame(
      name = coef_names,
      mean = default$mean,
      sd   = default$sd,
      lb   = default$lb,
      ub   = default$ub,
      stringsAsFactors = FALSE
    )
    for (nm in intersect(names(prior_spec), coef_names)) {
      p <- prior_spec[[nm]]
      idx <- which(coef_names == nm)
      out$mean[idx] <- p$mean
      out$sd[idx]   <- p$sd
      out$lb[idx]   <- p$lb
      out$ub[idx]   <- p$ub
    }
    out
  } else {
    stop("Prior must be a prior_normal() or a named list of prior_normal() objects.")
  }
}

# Null-coalescing helper (base R)
`%||%` <- function(a, b) if (!is.null(a)) a else b
