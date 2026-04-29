# Post-processing utilities and S3 methods for smoothbp_fit

#' Print a smoothbp_fit
#'
#' Displays a model header followed by posterior summaries, optionally filtered
#' by effect type.  Arguments are passed to \code{\link{summary.smoothbp_fit}}.
#'
#' @param x      A \code{smoothbp_fit} object.
#' @param digits Number of decimal places. Default \code{3}.
#' @param effects Which parameters to display.  See
#'   \code{\link{summary.smoothbp_fit}} for accepted values.
#' @param ...    Unused.
#' @export
print.smoothbp_fit <- function(x, digits = 3, effects = "all", ...) {
  has_re <- x$dm$n_groups_b0 > 0

  cat("Smooth Change-Point Model (smoothbp)\n")
  cat("-------------------------------------\n")
  cat(sprintf("Response : %s\n", x$response))
  cat(sprintf("Time var : %s\n", x$time))
  cat(sprintf("Chains   : %d \u00d7 %d draws (%d warmup)\n",
              x$chains, x$iter - x$warmup, x$warmup))
  if (has_re) {
    cat(sprintf("Groups   : %d (%s)\n",
                x$dm$n_groups_b0,
                deparse(x$b0_formula)))
  }

  # Resolve which sections to print
  show <- .resolve_effects(effects)

  if ("fixed" %in% show) {
    cat("\nPopulation-Level Effects:\n")
    .print_summary_section(summary(x, effects = "fixed", digits = digits))
  }

  if ("ran_pars" %in% show && has_re) {
    cat("\nRandom-Effect SDs:\n")
    .print_summary_section(summary(x, effects = "ran_pars", digits = digits))
  }

  if ("ran_vals" %in% show && has_re) {
    cat(sprintf("\nGroup-Level Deviations  (n = %d):\n", x$dm$n_groups_b0))
    .print_summary_section(summary(x, effects = "ran_vals", digits = digits))
  }

  invisible(x)
}

#' Summarise a smoothbp_fit
#'
#' Returns a data frame of posterior summaries (mean, SD, 95% CI, Rhat,
#' bulk-ESS, tail-ESS) for selected parameters.
#'
#' @param object  A \code{smoothbp_fit} object.
#' @param effects Character vector controlling which parameters are included.
#'   Accepted values (may be combined):
#'   \describe{
#'     \item{\code{"fixed"}}{Population-level regression coefficients
#'       (\eqn{b0}, \eqn{b1}, \eqn{b2}, \eqn{\omega}, \eqn{\rho}) and the
#'       residual SD \eqn{\sigma}.  This is the most commonly needed subset.}
#'     \item{\code{"ran_pars"}}{Random-effect variance parameter
#'       \eqn{\sigma_u}.}
#'     \item{\code{"ran_vals"}}{Individual group-level deviations
#'       \eqn{u_j}.}
#'     \item{\code{"all"}}{All of the above (default).}
#'   }
#'   Combinations are accepted, e.g. \code{effects = c("fixed", "ran_pars")}.
#' @param digits  Number of decimal places. Default \code{3}.
#' @param ...     Unused.
#'
#' @return A data frame with one row per selected parameter and columns
#'   \code{variable}, \code{mean}, \code{SD}, \code{Q2.5}, \code{Q97.5},
#'   \code{rhat}, \code{ess_bulk}, \code{ess_tail}.
#'
#' @examples
#' \dontrun{
#' summary(fit)                              # all parameters (default)
#' summary(fit, effects = "fixed")           # population-level only
#' summary(fit, effects = "ran_pars")        # sigma_u only
#' summary(fit, effects = "ran_vals")        # u[j] deviations only
#' summary(fit, effects = c("fixed", "ran_pars"))  # fixed + SD, no u[j]
#' }
#' @export
summary.smoothbp_fit <- function(object, effects = "all", digits = 3, ...) {

  show <- .resolve_effects(effects)

  # Classify every parameter
  pnames <- object$param_names
  class_df <- data.frame(
    variable = pnames,
    kind     = .classify_params(pnames),
    stringsAsFactors = FALSE
  )

  # Map effect class names to the kind labels used internally
  keep_kinds <- character(0)
  if ("fixed"    %in% show) keep_kinds <- c(keep_kinds, "fixed")
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
    mean, stats::sd,
    ~ posterior::quantile2(.x, probs = c(0.025, 0.975)),
    posterior::rhat,
    posterior::ess_bulk,
    posterior::ess_tail
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

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Print a summary data frame without row names, forcing enough width to avoid
# wrapping the diagnostics columns onto a second set of rows.
.print_summary_section <- function(s) {
  withr::with_options(
    list(width = max(getOption("width"), 120L)),
    print(s, row.names = FALSE)
  )
}

# Classify parameter names into fixed / ran_pars / ran_vals
.classify_params <- function(pnames) {
  vapply(pnames, function(p) {
    if (grepl("^u\\[", p))  "ran_vals"   # individual deviations u[j]
    else if (p == "sigma_u") "ran_pars"  # random-effect SD
    else                     "fixed"     # everything else (incl. sigma)
  }, character(1))
}

# Expand "all" shorthand and validate
.resolve_effects <- function(effects) {
  valid <- c("fixed", "ran_pars", "ran_vals", "all")
  bad   <- setdiff(effects, valid)
  if (length(bad)) {
    stop(
      "Unknown effects value(s): ", paste(bad, collapse = ", "), ".\n",
      "Valid choices: ", paste(valid, collapse = ", "), "."
    )
  }
  if ("all" %in% effects) c("fixed", "ran_pars", "ran_vals") else effects
}

#' Extract posterior draws as a draws_array
#'
#' @param x A \code{smoothbp_fit} object.
#' @param ... Unused.
#' @return A \code{posterior::draws_array}.
#' @export
posterior_draws <- function(x, ...) UseMethod("posterior_draws")

#' @export
posterior_draws.smoothbp_fit <- function(x, ...) x$draws

#' Compute fitted (posterior mean) values for each observation
#'
#' @param object A \code{smoothbp_fit} object.
#' @param summary Logical; if \code{TRUE} (default) return posterior mean and
#'   95% credible interval per observation. If \code{FALSE} return a matrix of
#'   draws (n_draws × n_obs).
#' @param ... Unused.
#' @return A data frame or matrix.
#' @export
fitted.smoothbp_fit <- function(object, summary = TRUE, ...) {
  dm   <- object$dm
  data <- object$data
  da   <- object$draws
  pn   <- object$param_names

  y   <- as.double(data[[object$response]])
  tau <- as.double(data[[object$time]])
  n   <- length(y)

  # Extract draw matrix: (n_draws × n_params)
  draw_mat <- posterior::as_draws_matrix(da)
  colnames(draw_mat) <- posterior::variables(da)
  n_draws  <- nrow(draw_mat)

  p_b0      <- ncol(dm$X_b0)
  n_groups  <- dm$n_groups_b0
  p_b1      <- ncol(dm$X_b1)
  p_b2      <- ncol(dm$X_b2)
  p_om      <- ncol(dm$X_om)
  p_rho     <- ncol(dm$X_rho)

  # Column offsets in draw_mat (matching State::to_vec order)
  off_b0  <- 0L
  off_u   <- off_b0  + p_b0
  off_b1  <- off_u   + n_groups
  off_b2  <- off_b1  + p_b1
  off_om  <- off_b2  + p_b2
  off_rho <- off_om  + p_om

  fitted_draws <- matrix(0, nrow = n_draws, ncol = n)

  for (s in seq_len(n_draws)) {
    beta_b0  <- draw_mat[s, (off_b0  + 1):(off_b0  + p_b0)]
    u_b0     <- if (n_groups > 0) draw_mat[s, (off_u + 1):(off_u + n_groups)] else numeric(0)
    beta_b1  <- draw_mat[s, (off_b1  + 1):(off_b1  + p_b1)]
    beta_b2  <- draw_mat[s, (off_b2  + 1):(off_b2  + p_b2)]
    beta_om  <- draw_mat[s, (off_om  + 1):(off_om  + p_om)]
    beta_rho <- draw_mat[s, (off_rho + 1):(off_rho + p_rho)]

    omega_i  <- as.vector(dm$X_om  %*% beta_om)
    rho_i    <- as.vector(dm$X_rho %*% beta_rho)
    b0_i     <- as.vector(dm$X_b0  %*% beta_b0)
    b1_i     <- as.vector(dm$X_b1  %*% beta_b1)
    b2_i     <- as.vector(dm$X_b2  %*% beta_b2)

    d_i <- tau - omega_i
    s_i <- 1 / (1 + exp(-d_i * rho_i))

    mu_i <- b0_i + b1_i * d_i + b2_i * d_i * s_i

    if (n_groups > 0) {
      for (i in seq_len(n)) {
        g <- dm$group_b0[i]
        if (g >= 0L) mu_i[i] <- mu_i[i] + u_b0[g + 1L]
      }
    }

    fitted_draws[s, ] <- mu_i
  }

  if (!summary) return(fitted_draws)

  data.frame(
    .observation = seq_len(n),
    fitted_mean  = colMeans(fitted_draws),
    fitted_Q2.5  = apply(fitted_draws, 2, stats::quantile, probs = 0.025),
    fitted_Q97.5 = apply(fitted_draws, 2, stats::quantile, probs = 0.975)
  )
}

#' Posterior predictive check (density overlay)
#'
#' Overlays densities of a random sample of posterior predictive draws over the
#' observed response density.
#'
#' @param object A \code{smoothbp_fit} object.
#' @param n_draws Number of predictive draws to overlay. Default 50.
#' @param ... Unused.
#' @return A \code{ggplot} object.
#' @export
pp_check <- function(object, ...) UseMethod("pp_check")

#' @export
pp_check.smoothbp_fit <- function(object, n_draws = 50, ...) {
  y_obs  <- as.double(object$data[[object$response]])
  n      <- length(y_obs)

  fit_mat <- fitted(object, summary = FALSE)
  sigma_draws <- as.vector(
    posterior::as_draws_matrix(object$draws)[, "sigma"]
  )

  # Sample n_draws rows
  idx <- sample(nrow(fit_mat), min(n_draws, nrow(fit_mat)))
  y_rep <- matrix(0, nrow = length(idx), ncol = n)
  for (k in seq_along(idx)) {
    s <- idx[k]
    y_rep[k, ] <- stats::rnorm(n, mean = fit_mat[s, ], sd = sigma_draws[s])
  }

  # Build tidy data for ggplot
  df_obs <- data.frame(y = y_obs, .draw = "observed", stringsAsFactors = FALSE)
  df_rep <- do.call(rbind, lapply(seq_len(nrow(y_rep)), function(k) {
    data.frame(y = y_rep[k, ], .draw = paste0("rep_", k), stringsAsFactors = FALSE)
  }))

  ggplot2::ggplot() +
    ggplot2::geom_density(
      data = df_rep,
      ggplot2::aes(x = y, group = .draw),
      colour = "steelblue", alpha = 0.3, linewidth = 0.3
    ) +
    ggplot2::geom_density(
      data = df_obs,
      ggplot2::aes(x = y),
      colour = "black", linewidth = 1
    ) +
    ggplot2::labs(
      x = object$response,
      y = "Density",
      title = "Posterior predictive check",
      subtitle = sprintf("%d replicated datasets (blue) vs observed (black)", length(idx))
    )
}
