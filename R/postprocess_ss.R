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
print.smoothbp_ss_fit <- function(x, digits = 3, effects = "fixed", ...) {
  cat("smoothbp spike-and-slab fit\n")
  cat(sprintf("  Formula: %s\n", deparse(x$formula)))
  cat(sprintf("  Chains: %d, Iterations: %d (warmup: %d)\n",
              x$chains, x$iter, x$warmup))
  if (x$n_divergent > 0L) {
    cat(sprintf("  WARNING: %d divergent transitions\n", x$n_divergent))
  }

  cat("\nPosterior inclusion probabilities (spike-and-slab b2 coefficients):\n")
  pips <- pip(x)
  for (i in seq_along(pips)) {
    cat(sprintf("  %-20s  PIP = %.3f\n", names(pips)[i], pips[i]))
  }

  cat("\nParameter summary:\n")
  s <- summary(x, effects = effects, digits = digits)
  withr::with_options(
    list(width = max(getOption("width"), 120L)),
    print(s, row.names = FALSE)
  )

  invisible(x)
}

#' Summarise a smoothbp_ss_fit
#'
#' Returns a data frame of posterior summaries for selected parameters.
#'
#' @param object A \code{smoothbp_ss_fit} object.
#' @param effects Character vector controlling which parameters are included.
#'   Accepted values:
#'   \describe{
#'     \item{\code{"fixed"}}{Population-level regression coefficients
#'       (\eqn{b0}, \eqn{b1}, \eqn{b2}, \eqn{\omega}, \eqn{\rho}), residual SD,
#'       and spike-and-slab indicator variables (\eqn{\gamma}).}
#'     \item{\code{"ran_pars"}}{Random-effect variance parameter
#'       \eqn{\sigma_u}.}
#'     \item{\code{"ran_vals"}}{Individual group-level deviations
#'       \eqn{u_j}.}
#'     \item{\code{"all"}}{All of the above (default).}
#'   }
#' @param digits Number of decimal places. Default \code{3}.
#' @param ... Unused.
#'
#' @return A data frame with one row per selected parameter and columns
#'   \code{variable}, \code{mean}, \code{sd}, \code{Q2.5}, \code{Q97.5},
#'   \code{rhat}, \code{ess_bulk}, \code{ess_tail}. For gamma (spike-and-slab)
#'   parameters, the \code{mean} column contains the posterior inclusion
#'   probability (PIP).
#'
#' @export
summary.smoothbp_ss_fit <- function(object, effects = "all", digits = 3, ...) {

  show <- .resolve_effects(effects)

  # Extract all variables and classify them
  all_vars <- posterior::variables(object$draws)
  
  class_df <- data.frame(
    variable = all_vars,
    kind     = vapply(all_vars, function(v) {
      if (grepl("^gamma_", v))  "gamma"     # spike-and-slab indicators
      else if (grepl("^u\\[", v)) "ran_vals"   # individual deviations
      else if (v == "sigma_u")  "ran_pars"  # random-effect SD
      else                      "fixed"     # everything else
    }, character(1)),
    stringsAsFactors = FALSE
  )

  # Map effect class names to the kind labels used internally
  keep_kinds <- character(0)
  if ("fixed"    %in% show) keep_kinds <- c(keep_kinds, "fixed", "gamma")
  if ("ran_pars" %in% show) keep_kinds <- c(keep_kinds, "ran_pars")
  if ("ran_vals" %in% show) keep_kinds <- c(keep_kinds, "ran_vals")

  keep_vars <- class_df$variable[class_df$kind %in% keep_kinds]

  if (length(keep_vars) == 0) {
    message("No parameters match effects = ", paste(show, collapse = ", "),
            " for this model.")
    return(invisible(data.frame()))
  }

  s <- posterior::summarise_draws(
    object$draws[, , keep_vars, drop = FALSE],
    mean   = mean,
    sd     = stats::sd,
    Q2.5   = ~ stats::quantile(.x, 0.025),
    Q97.5  = ~ stats::quantile(.x, 0.975),
    rhat   = posterior::rhat,
    ess_bulk = posterior::ess_bulk,
    ess_tail = posterior::ess_tail
  )

  # Normalise column names regardless of how posterior:: qualifies them
  nms <- names(s)
  nms[grepl("^(stats::)?sd$",           nms)] <- "SD"
  nms[grepl("q2\\.5",                   nms)] <- "Q2.5"
  nms[grepl("q97\\.5",                  nms)] <- "Q97.5"
  nms[grepl("rhat",                     nms)] <- "Rhat"
  nms[grepl("ess_bulk",                 nms)] <- "Bulk_ESS"
  nms[grepl("ess_tail",                 nms)] <- "Tail_ESS"
  names(s) <- nms

  num_cols <- sapply(s, is.numeric)
  s[num_cols] <- lapply(s[num_cols], round, digits = digits)

  as.data.frame(s)
}
