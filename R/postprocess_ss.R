#' Posterior inclusion probabilities from a spike-and-slab fit
#'
#' @param object A `smoothbp_ss_fit` object.
#' @param ... Ignored.
#'
#' @return A named numeric vector of posterior inclusion probabilities (PIPs)
#'   for each b2 coefficient that had a spike-and-slab prior.
#'
#' @export
pip <- function(object, ...) UseMethod("pip")

#' @export
pip.smoothbp_ss_fit <- function(object, ...) {
  dm <- posterior::as_draws_matrix(object$draws)
  gamma_cols <- object$gamma_names
  pips <- colMeans(as.matrix(dm[, gamma_cols, drop = FALSE]))
  # Clean up names: remove "gamma_" prefix
  names(pips) <- sub("^gamma_", "", names(pips))
  pips
}

#' @export
print.smoothbp_ss_fit <- function(x, ...) {
  cat("smoothbp spike-and-slab fit\n")
  cat(sprintf("  Formula: %s\n", deparse(x$formula)))
  cat(sprintf("  Chains: %d, Iterations: %d (warmup: %d)\n",
              x$chains, x$iter, x$warmup))
  if (x$n_divergent > 0L) {
    cat(sprintf("  WARNING: %d divergent transitions\n", x$n_divergent))
  }

  cat("\nPosterior inclusion probabilities (b2 coefficients):\n")
  pips <- pip(x)
  for (i in seq_along(pips)) {
    cat(sprintf("  %-20s  PIP = %.3f\n", names(pips)[i], pips[i]))
  }

  cat("\nParameter summary (included + always-on):\n")
  s <- summary(x)
  # Show non-gamma parameters
  non_gamma <- s[!grepl("^gamma_", s$variable), ]
  print(non_gamma, row.names = FALSE)

  invisible(x)
}

#' @export
summary.smoothbp_ss_fit <- function(object, ...) {
  # Delegate to the base smoothbp_fit summary which uses posterior::summarise_draws
  s <- posterior::summarise_draws(
    object$draws,
    mean   = mean,
    sd     = stats::sd,
    Q2.5   = ~ stats::quantile(.x, 0.025),
    Q97.5  = ~ stats::quantile(.x, 0.975),
    rhat   = posterior::rhat,
    ess_bulk = posterior::ess_bulk,
    ess_tail = posterior::ess_tail
  )
  as.data.frame(s)
}
