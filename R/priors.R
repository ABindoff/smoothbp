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
#' ## Bounds (`lb`, `ub`)
#'
#' All five regression-coefficient parameters (`b0`, `b1`, `b2`, `omega`,
#' `rho`) support finite lower and upper bounds via the `lb` and `ub` arguments
#' of [prior_normal()].  The bounds are enforced by the sampler:
#'
#' - **`b0`, `b1`, `b2`**: The conjugate Gibbs draw is used as an independence
#'   Metropolis-Hastings proposal; the entire linear draw is rejected whenever
#'   any coefficient falls outside its `[lb, ub]` interval.  This is exact
#'   rejection sampling from the truncated full conditional and has no cost
#'   when all bounds are infinite (the default).
#' - **`omega`, `rho`**: Bounds are enforced by boundary reflection during HMC
#'   leapfrog integration (for multi-coefficient predictors) or by immediate
#'   rejection of out-of-bounds proposals (for intercept-only predictors).
#'
#' Typical usage: bound the `omega` intercept to the observed time range to
#' prevent the change-point from drifting into an unidentifiable region
#' (`prior_normal(3, 2, lb = 0, ub = max(data$tau))`).  Bounds on `b1` or
#' `b2` can encode scientific constraints such as requiring a non-negative
#' slope change (`b2 = prior_normal(0, 2, lb = 0)`).
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

#' Specify a spike-and-slab prior for variable selection on b2 coefficients
#'
#' Used with [smoothbp_ss()] to place a point-mass spike at zero on selected
#' `b2` coefficients (and their corresponding `omega`/`rho` coefficients).
#' When the spike is active (`gamma_k = 0`), the coefficient is exactly zero;
#' when inactive (`gamma_k = 1`), it follows the slab distribution.
#'
#' @param pi Prior inclusion probability.  Scalar (applied to all eligible
#'   coefficients) or a named numeric vector mapping coefficient names to
#'   individual probabilities.  Default `0.5`.
#' @param slab A [prior_normal()] object for the slab component.  Default
#'   `prior_normal(0, 2)`.
#' @param spike_intercept Logical; should the intercept of `b2` also receive
#'   a spike-and-slab prior?  Default `FALSE` (intercept always included).
#'
#' @return A `smoothbp_spike_slab` object.
#' @export
prior_spike_slab <- function(pi = 0.5, slab = prior_normal(0, 2),
                             spike_intercept = FALSE) {
  stopifnot(
    inherits(slab, "smoothbp_prior"),
    slab$family == "normal",
    is.numeric(pi),
    all(pi > 0 & pi < 1),
    is.logical(spike_intercept)
  )
  structure(
    list(family = "spike_slab", pi = pi, slab = slab,
         spike_intercept = spike_intercept),
    class = "smoothbp_spike_slab"
  )
}

#' @export
print.smoothbp_spike_slab <- function(x, ...) {
  cat(sprintf("SpikeSlab(pi=%s, slab=Normal(%g, %g), spike_intercept=%s)\n",
              paste(format(x$pi), collapse = ","),
              x$slab$mean, x$slab$sd,
              x$spike_intercept))
  invisible(x)
}
