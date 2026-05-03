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
#' Computes the posterior distribution of the latent mean \eqn{\mu_i} at each
#' observation. By default, fitted values are computed for the observations in
#' the training data. A \code{newdata} argument allows evaluation at arbitrary
#' covariate combinations, enabling marginal (population-level) or conditional
#' (group-specific) predictions.
#'
#' @param object A \code{smoothbp_fit} object.
#' @param newdata Optional data frame of new observations at which to evaluate
#'   the model. Must contain:
#'   \describe{
#'     \item{The time variable}{Required; covariate values at which predictions
#'       are needed.}
#'     \item{Covariates in model formulas}{Any variables appearing in the
#'       \code{b0}, \code{b1}, \code{b2}, \code{omega}, or \code{rho} formulas.}
#'     \item{Random effect grouping variable (optional)}{If the grouping variable
#'       (e.g. from \code{(1 | group)} in the \code{b0} formula) is present and
#'       its levels match those in the training data, corresponding group-level
#'       deviations are included (conditional prediction). If absent or contains
#'       unknown levels, predictions are population-level only (marginal
#'       prediction, with \eqn{u_j = 0}).}
#'   }
#'   Defaults to \code{NULL}, in which case predictions are computed for the
#'   original training data observations.
#' @param summary Logical; if \code{TRUE} (default) return a data frame with
#'   posterior mean and 95\% credible interval per observation. If \code{FALSE}
#'   return a matrix of n_draws × n_obs posterior draws, useful for custom
#'   summaries or visualization.
#' @param ... Unused.
#' @return
#'   If `summary = TRUE`, a data frame with columns:
#'   * `.observation`: Row index (1 to n).
#'   * `fitted_mean`: Posterior mean of \eqn{\mu_i}.
#'   * `fitted_Q2.5`: 2.5% posterior quantile (lower credible limit).
#'   * `fitted_Q97.5`: 97.5% posterior quantile (upper credible limit).
#'
#'   If `summary = FALSE`, a matrix of posterior draws with dimensions
#'   n_draws × n_obs, suitable for custom summaries or integration with
#'   other packages.
#' @examples
#' \dontrun{
#' # Fitted values at training observations (conditional on subjects)
#' fitted(fit)
#'
#' # Population-level predictions at new time points
#' # (omit the grouping variable for marginal predictions)
#' newdata_marginal <- data.frame(
#'   time  = seq(0, 10, by = 0.5),
#'   group = "Control"
#' )
#' fitted(fit, newdata = newdata_marginal)
#'
#' # Subject-specific predictions
#' # (include the grouping variable for conditional predictions)
#' newdata_conditional <- data.frame(
#'   time    = seq(0, 10, by = 0.5),
#'   group   = "Control",
#'   subject = "SUBJ001"
#' )
#' fitted(fit, newdata = newdata_conditional)
#'
#' # Posterior draws for custom summaries
#' draws_mat <- fitted(fit, summary = FALSE)
#' # e.g. P(fitted value > 5) per observation
#' colMeans(draws_mat > 5)
#' }
#' @export
fitted.smoothbp_fit <- function(object, newdata = NULL, summary = TRUE, ...) {

  # ---- Design matrices and tau ----------------------------------------------
  if (is.null(newdata)) {
    dm  <- object$dm
    tau <- as.double(object$data[[object$time]])
  } else {
    if (!object$time %in% names(newdata)) {
      stop(sprintf("'newdata' must contain the time variable '%s'.", object$time))
    }
    tau <- as.double(newdata[[object$time]])
    dm  <- .build_newdata_dm(object, newdata)
  }

  n <- length(tau)

  # ---- Posterior draws ------------------------------------------------------
  draw_mat  <- posterior::as_draws_matrix(object$draws)
  col_names <- posterior::variables(object$draws)
  n_draws   <- nrow(draw_mat)

  # Column index vectors (computed once, outside loop)
  b0_cols  <- grepl("^b0_",    col_names)
  b1_cols  <- grepl("^b1_",    col_names)
  b2_cols  <- grepl("^b2_",    col_names)
  u_cols   <- grepl("^u\\[",   col_names)
  om_cols  <- grepl("^omega_", col_names)
  rho_cols <- grepl("^rho_",   col_names)

  n_groups <- dm$n_groups_b0

  # ---- Draw loop ------------------------------------------------------------
  fitted_draws <- matrix(0, nrow = n_draws, ncol = n)

  for (s in seq_len(n_draws)) {
    # Extract parameter vectors. Subsetting a draws_matrix returns a draws
    # object, so coerce to plain numeric.
    beta_b0  <- as.numeric(draw_mat[s, b0_cols])
    beta_b1  <- as.numeric(draw_mat[s, b1_cols])
    beta_b2  <- as.numeric(draw_mat[s, b2_cols])
    beta_om  <- as.numeric(draw_mat[s, om_cols])
    beta_rho <- as.numeric(draw_mat[s, rho_cols])
    u_b0     <- if (n_groups > 0) as.numeric(draw_mat[s, u_cols]) else numeric(0)

    omega_i <- as.vector(dm$X_om  %*% beta_om)
    rho_i   <- as.vector(dm$X_rho %*% beta_rho)
    b0_i    <- as.vector(dm$X_b0  %*% beta_b0)
    b1_i    <- as.vector(dm$X_b1  %*% beta_b1)
    b2_i    <- as.vector(dm$X_b2  %*% beta_b2)

    d_i  <- tau - omega_i
    s_i  <- 1 / (1 + exp(-d_i * rho_i))
    mu_i <- b0_i + b1_i * d_i + b2_i * d_i * s_i

    # Add group-level deviations where available (g == -1L means no RE)
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

# ---------------------------------------------------------------------------
# Internal: build design matrices for newdata, reusing training-data factor
# levels for the random-effect grouping variable.
# ---------------------------------------------------------------------------
.build_newdata_dm <- function(object, newdata) {

  b0_parsed  <- .parse_re(object$b0_formula)
  b1_parsed  <- .parse_re(object$b1_formula)
  b2_parsed  <- .parse_re(object$b2_formula)
  om_parsed  <- .parse_re(object$omega_formula)
  rho_parsed <- .parse_re(object$rho_formula)

  # Build a model matrix for newdata, recovering factor levels and scale()
  # parameters from the training data.
  #
  # Issue 1: When newdata holds a constant numeric column (e.g. mean(x)),
  # scale() computes sd = 0 and returns NaN; na.omit then drops every row.
  # Solution: pass NAs through and replace them using training means/SDs.
  #
  # Issue 2: When newdata has a factor with fewer levels than training data
  # (e.g. Sex = 'Female' only), model.matrix fails on contrasts.
  # Solution: coerce all factors in newdata to use training-data levels first.
  mk_mm <- function(fml, dat) {
    # Align factor levels with training data
    for (col in names(dat)) {
      if (col %in% names(object$data)) {
        train_col <- object$data[[col]]
        if (is.factor(train_col)) {
          dat[[col]] <- factor(dat[[col]], levels = levels(train_col))
        }
      }
    }

    old_na <- getOption("na.action")
    options(na.action = "na.pass")
    on.exit(options(na.action = old_na), add = TRUE)
    mm <- stats::model.matrix(fml, data = dat)

    nan_cols <- which(colSums(is.nan(mm)) > 0)
    if (length(nan_cols) > 0) {
      for (j in nan_cols) {
        col_nm  <- colnames(mm)[j]
        raw_var <- sub("^scale\\((.+)\\)$", "\\1", col_nm)
        if (raw_var != col_nm && raw_var %in% names(object$data) &&
            raw_var %in% names(dat)) {
          train_mean <- mean(object$data[[raw_var]], na.rm = TRUE)
          train_sd   <- stats::sd(object$data[[raw_var]], na.rm = TRUE)
          mm[, j] <- (dat[[raw_var]] - train_mean) / train_sd
        } else {
          mm[is.nan(mm[, j]), j] <- 0
        }
      }
    }
    mm
  }

  X_b0  <- mk_mm(b0_parsed$fixed,  newdata)
  X_b1  <- mk_mm(b1_parsed$fixed,  newdata)
  X_b2  <- mk_mm(b2_parsed$fixed,  newdata)
  X_om  <- mk_mm(om_parsed$fixed,  newdata)
  X_rho <- mk_mm(rho_parsed$fixed, newdata)

  # Random effects: use training-data group levels so indices align with draws
  re_var          <- b0_parsed$re_group
  group_levels_b0 <- object$dm$group_levels_b0
  n_groups_b0     <- object$dm$n_groups_b0

  if (!is.null(re_var) && re_var %in% names(newdata)) {
    gvar   <- newdata[[re_var]]
    gfac   <- factor(gvar, levels = group_levels_b0)  # use training levels
    n_new  <- nlevels(factor(gvar))   # how many distinct values in newdata
    n_unknown <- sum(is.na(gfac) & !is.na(gvar))
    if (n_unknown > 0) {
      warning(sprintf(
        "%d row(s) in newdata have a '%s' level not seen during fitting; ",
        n_unknown, re_var,
        "their random effects will be set to 0 (population-level)."
      ))
    }
    group_b0 <- ifelse(is.na(gfac), -1L, as.integer(gfac) - 1L)
  } else {
    # Grouping variable absent from newdata: marginal (population-level) predictions
    group_b0 <- rep(-1L, nrow(newdata))
    n_groups_b0 <- 0L
  }

  list(
    X_b0            = X_b0,
    X_b1            = X_b1,
    X_b2            = X_b2,
    X_om            = X_om,
    X_rho           = X_rho,
    group_b0        = group_b0,
    n_groups_b0     = n_groups_b0,
    group_levels_b0 = group_levels_b0
  )
}

#' Log-likelihood for smoothbp_fit objects
#'
#' Computes the pointwise log-likelihood matrix for leave-one-out
#' cross-validation and model comparison. Each row is a posterior draw;
#' each column is an observation.
#'
#' @param object A \code{smoothbp_fit} object.
#' @param ... Unused.
#' @return A matrix of dimensions (n_draws × n_obs) with log-likelihood values.
#'   Rows are posterior draws, columns are observations.
#' @details
#' The model likelihood is normal: \eqn{y_i \sim N(\mu_i, \sigma^2)}.
#' This function returns \eqn{\log p(y_i | \mu_i, \sigma)} for each
#' observation and posterior draw.
#' @examples
#' \dontrun{
#' ll <- log_lik(fit)
#' dim(ll)  # n_draws × n_obs
#'
#' # Use with loo package for model comparison
#' loo::loo(ll)
#' }
#' @export
log_lik.smoothbp_fit <- function(object, ...) {
  y_obs <- as.double(object$data[[object$response]])
  n_obs <- length(y_obs)

  # Fitted draws: n_draws × n_obs matrix of μᵢ
  fit_draws <- fitted(object, summary = FALSE)
  n_draws <- nrow(fit_draws)

  # Sigma draws: vector of length n_draws
  draw_mat <- posterior::as_draws_matrix(object$draws)
  sigma_draws <- as.numeric(draw_mat[, "sigma"])

  # Log-likelihood matrix: dnorm(y, μ, σ, log = TRUE)
  ll_matrix <- matrix(0, nrow = n_draws, ncol = n_obs)
  for (i in seq_len(n_obs)) {
    ll_matrix[, i] <- stats::dnorm(
      y_obs[i],
      mean = fit_draws[, i],
      sd = sigma_draws,
      log = TRUE
    )
  }

  ll_matrix
}

#' Leave-one-out cross-validation for smoothbp_fit objects
#'
#' Computes leave-one-out information criterion (LOO-IC) using Pareto
#' smoothed importance sampling (PSIS). Compatible with
#' \code{\link[loo]{loo_compare}} for model comparison, including comparisons
#' with \code{brmsfit} objects (which also return \code{psis_loo} objects from
#' their own \code{loo()} method).
#'
#' For Bayes factor comparisons (rather than predictive comparisons), see
#' \code{\link{bayes_factor.smoothbp_fit}}.
#'
#' @param x A \code{smoothbp_fit} object.
#' @param ... Additional arguments passed to \code{\link[loo]{loo}}.
#' @return An object of class \code{psis_loo} (from the \code{loo} package).
#' @examples
#' \dontrun{
#' loo_fit <- loo(fit)
#'
#' # Compare two smoothbp models
#' loo::loo_compare(loo(fit1), loo(fit2))
#'
#' # Compare a smoothbp model with a brmsfit
#' library(brms)
#' fit_linear <- brm(y ~ tau + (1|subject), data = d)
#' loo::loo_compare(loo(fit_piece), loo(fit_linear))
#' }
#' @export
loo.smoothbp_fit <- function(x, ...) {
  if (!requireNamespace("loo", quietly = TRUE)) {
    stop("The 'loo' package is required for LOO-IC computation.")
  }
  ll <- log_lik(x)
  loo::loo(ll, ...)
}

#' Watanabe-Akaike information criterion for smoothbp_fit objects
#'
#' Computes WAIC for model comparison and assessment.
#'
#' @param x A \code{smoothbp_fit} object.
#' @param ... Additional arguments passed to \code{\link[loo]{waic}}.
#' @return An object of class \code{waic} (from the \code{loo} package).
#' @examples
#' \dontrun{
#' waic_fit <- waic(fit)
#'
#' # Compare two smoothbp models
#' loo::loo_compare(waic(fit1), waic(fit2))
#' }
#' @export
waic.smoothbp_fit <- function(x, ...) {
  if (!requireNamespace("loo", quietly = TRUE)) {
    stop("The 'loo' package is required for WAIC computation.")
  }
  ll <- log_lik(x)
  loo::waic(ll, ...)
}

pp_check <- function(object, ...) UseMethod("pp_check")

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
