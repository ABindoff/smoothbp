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

#' Specify a gamma prior for a parameter
#'
#' @param shape Shape parameter (> 0).
#' @param scale Scale parameter (> 0).
#'
#' @return A `smoothbp_prior` object.
#' @export
prior_gamma <- function(shape = 1, scale = 1) {
  stopifnot(shape > 0, scale > 0)
  structure(
    list(family = "gamma", shape = shape, scale = scale),
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
  } else if (x$family == "gamma") {
    cat(sprintf("Gamma(shape=%g, scale=%g)\n", x$shape, x$scale))
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
#' For multi-breakpoint models, `deltas`, `omega`, and `rho` can also be
#' **lists of prior specifications** (one per breakpoint slot). If a single
#' specification is provided, it is applied to all slots.
#'
#' @param b0      Prior(s) for `b0` regression coefficients.
#' @param b1      Prior(s) for `b1` regression coefficients.
#' @param deltas  Prior(s) for slope change coefficients (one list per segment).
#' @param omega   Prior(s) for `omega` coefficients (one list per segment).
#' @param rho     Prior(s) for `rho` coefficients (one list per segment).
#' @param sigma   `prior_invgamma()` for residual SD.
#' @param sigma_u `prior_invgamma()` for random-effect SD.
#'
#' @return A `smoothbp_priors` list.
#' @export
smoothbp_priors <- function(
    b0      = prior_normal(0, 10),
    b1      = prior_normal(0, 2),
    deltas  = prior_normal(0, 2),
    omega   = prior_normal(3, 2, lb = 0),
    rho     = prior_normal(3, 2, lb = 0),
    sigma   = prior_invgamma(1, 1),
    sigma_u = prior_invgamma(1, 1),
    sigma_re_om = prior_invgamma(1, 1)
) {
  structure(
    list(b0 = b0, b1 = b1, deltas = deltas, omega = omega, rho = rho,
         sigma = sigma, sigma_u = sigma_u, sigma_re_om = sigma_re_om),
    class = "smoothbp_priors"
  )
}

#' @export
print.smoothbp_priors <- function(x, ...) {
  cat("smoothbp priors:\n")
  for (nm in c("b0", "b1", "deltas", "omega", "rho", "sigma", "sigma_u", "sigma_re_om")) {
    cat(sprintf("  %-8s: ", nm))
    if (is.list(x[[nm]]) && !inherits(x[[nm]], "smoothbp_prior")) {
        cat("<list of priors>\n")
    } else {
        print(x[[nm]])
    }
  }
  invisible(x)
}

#' Generate evenly spaced priors for candidate breakpoints
#'
#' This helper function generates a list of `prior_normal` objects for `omega`
#' (breakpoint locations) that are evenly spaced across the range of your time
#' variable `tau`. This is highly recommended when using `smoothbp_ss()` to
#' ensure the candidate breakpoints cover the entire domain without clumping.
#'
#' For hierarchical models where `omega` has random effects (e.g., `~ 1 + (1 | group)`),
#' this function automatically names the prior `(Intercept)` so it applies correctly
#' to the global market mean, while the random effects are handled automatically by
#' the `sigma_re_om` shrinkage variance.
#'
#' @param K Number of candidate breakpoints.
#' @param tau_min Minimum value of the time/covariate variable.
#' @param tau_max Maximum value of the time/covariate variable.
#'
#' @return A list of length `K` containing prior specifications for `omega`.
#' @export
space_omega_priors <- function(K, tau_min, tau_max) {
  stopifnot(K >= 1, tau_max > tau_min)
  
  # Pad the edges so we don't push breakpoints right to the very limits
  means <- seq(tau_min, tau_max, length.out = K + 2)[2:(K + 1)]
  
  # Standard deviation heuristic: width of interval / K
  sd_val <- (tau_max - tau_min) / K
  
  lapply(means, function(m) {
    list(
      "(Intercept)" = prior_normal(mean = m, sd = sd_val, lb = tau_min, ub = tau_max)
    )
  })
}

# ---------------------------------------------------------------------------
# Internal helpers: expand priors to per-coefficient vectors
# ---------------------------------------------------------------------------

.expand_prior <- function(prior_spec, coef_names) {
  n <- length(coef_names)
  if (inherits(prior_spec, "smoothbp_prior") && prior_spec$family == "normal") {
    data.frame(
      name = coef_names,
      mean = prior_spec$mean,
      sd   = prior_spec$sd,
      lb   = prior_spec$lb,
      ub   = prior_spec$ub,
      stringsAsFactors = FALSE
    )
  } else if (is.list(prior_spec) && !inherits(prior_spec, "smoothbp_prior")) {
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

`%||%` <- function(a, b) if (!is.null(a)) a else b

#' Specify a spike-and-slab prior for variable selection
#'
#' Used with [smoothbp_ss()] to place a point-mass spike at zero on selected
#' coefficients.
#'
#' @param pi Prior inclusion probability. Default `0.5`.
#' @param slab A [prior_normal()] object for the slab component.
#' @param learn_pi Logical; if `TRUE`, place a `Beta(a, b)` hyperprior on pi.
#' @param a Shape parameter for the Beta hyperprior. Default `1`.
#' @param b Shape parameter for the Beta hyperprior. Default `1`.
#'
#' @return A `smoothbp_spike_slab` object.
#' @export
prior_spike_slab <- function(pi = 0.5, slab = prior_normal(0, 2),
                             learn_pi = FALSE, a = 1, b = 1) {
  stopifnot(
    inherits(slab, "smoothbp_prior"),
    slab$family == "normal",
    is.numeric(pi),
    all(pi > 0 & pi < 1),
    is.logical(learn_pi)
  )
  structure(
    list(family = "spike_slab", pi = pi, slab = slab,
         learn_pi = learn_pi, a = a, b = b),
    class = "smoothbp_spike_slab"
  )
}

#' @export
print.smoothbp_spike_slab <- function(x, ...) {
  if (x$learn_pi) {
    cat(sprintf("SpikeSlab(pi~Beta(%g,%g), slab=Normal(%g, %g), spike_intercept=%s)\n",
                x$a, x$b, x$slab$mean, x$slab$sd, x$spike_intercept))
  } else {
    cat(sprintf("SpikeSlab(pi=%s, slab=Normal(%g, %g), spike_intercept=%s)\n",
                paste(format(x$pi), collapse = ","),
                x$slab$mean, x$slab$sd, x$spike_intercept))
  }
  invisible(x)
}
