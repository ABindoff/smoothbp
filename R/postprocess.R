# Post-processing utilities and S3 methods for smoothbp_fit

#' Print a smoothbp_fit
#'
#' @param x A `smoothbp_fit` object.
#' @param digits Number of decimal places to print.
#' @param effects Which effects to show: `"fixed"`, `"ran_pars"`, `"ran_vals"`, or `"all"`.
#' @param ... Unused.
#'
#' @export
print.smoothbp_fit <- function(x, digits = 3, effects = "all", ...) {
  cat("Smooth Change-Point Model (smoothbp)\n")
  cat("-------------------------------------\n")
  cat(sprintf("Response : %s\n", x$response))
  cat(sprintf("Time var : %s\n", x$time))
  cat(sprintf("Chains   : %d \u00d7 %d draws (%d warmup)\n",
              x$chains, x$iter - x$warmup, x$warmup))
  
  show <- .resolve_effects(effects)
  if ("fixed" %in% show) {
    cat("\nPopulation-Level Effects:\n")
    .print_summary_section(summary(x, effects = "fixed", digits = digits))
  }
  invisible(x)
}

#' Summarise a smoothbp_fit
#'
#' @param object A `smoothbp_fit` object.
#' @param effects Which effects to summarise: `"fixed"`, `"ran_pars"`, `"ran_vals"`, or `"all"`.
#' @param digits Number of decimal places for rounding.
#' @param ... Unused.
#'
#' @export
summary.smoothbp_fit <- function(object, effects = "all", digits = 3, ...) {
  show <- .resolve_effects(effects)
  pnames <- colnames(posterior::as_draws_matrix(object$draws))
  
  # Filter kinds
  keep_vars <- pnames
  if (!"ran_vals" %in% show) keep_vars <- keep_vars[!grepl("^u\\[", keep_vars)]
  if (!"ran_pars" %in% show) keep_vars <- keep_vars[keep_vars != "sigma_u"]
  
  s <- posterior::summarise_draws(
    object$draws[, , keep_vars, drop = FALSE],
    mean, stats::sd,
    ~ posterior::quantile2(.x, probs = c(0.025, 0.975)),
    posterior::rhat,
    posterior::ess_bulk,
    posterior::ess_tail
  )

  nms <- names(s)
  nms[grepl("sd$",       nms)] <- "SD"
  nms[grepl("q2\\.5",    nms)] <- "Q2.5"
  nms[grepl("q97\\.5",   nms)] <- "Q97.5"
  nms[grepl("rhat",      nms)] <- "Rhat"
  nms[grepl("ess_bulk",  nms)] <- "Bulk_ESS"
  nms[grepl("ess_tail",  nms)] <- "Tail_ESS"
  names(s) <- nms

  num_cols <- sapply(s, is.numeric)
  s[num_cols] <- lapply(s[num_cols], round, digits = digits)
  as.data.frame(s)
}

.print_summary_section <- function(s) {
  withr::with_options(list(width = 120), print(s, row.names = FALSE))
}

.resolve_effects <- function(effects) {
  valid <- c("fixed", "ran_pars", "ran_vals", "all")
  if ("all" %in% effects) return(c("fixed", "ran_pars", "ran_vals"))
  effects
}

#' Convert draws to a data frame
#'
#' @param x A `smoothbp_fit` object.
#' @param ... Passed to `as.data.frame.draws_df`.
#'
#' @export
as.data.frame.smoothbp_fit <- function(x, ...) {
  as.data.frame(posterior::as_draws_df(x$draws))
}

#' Fitted values for smoothbp_fit objects
#'
#' @param object A `smoothbp_fit` object.
#' @param newdata Optional data frame for prediction.
#' @param summary Logical; if `TRUE` (default), returns the mean and 95% CI of the fitted values.
#' @param ... Unused.
#'
#' @export
fitted.smoothbp_fit <- function(object, newdata = NULL, summary = TRUE, ...) {
  if (is.null(newdata)) {
    dm  <- object$dm
    tau <- as.double(object$data[[object$time]])
  } else {
    tau <- as.double(newdata[[object$time]])
    dm  <- .build_newdata_dm(object, newdata)
  }

  n <- length(tau)
  draw_mat  <- posterior::as_draws_matrix(object$draws)
  col_names <- colnames(draw_mat)
  n_draws   <- nrow(draw_mat)
  n_bp      <- length(dm$X_deltas)

  b0_cols  <- which(grepl("^b0_", col_names))
  b1_cols  <- which(grepl("^b1_", col_names))
  u_cols   <- which(grepl("^u\\[", col_names))
  delta_cols_list <- lapply(seq_len(n_bp), function(k) which(grepl(paste0("^delta", k, "_"), col_names)))
  om_cols_list    <- lapply(seq_len(n_bp), function(k) which(grepl(paste0("^omega", k, "_"), col_names)))
  rho_cols_list   <- lapply(seq_len(n_bp), function(k) which(grepl(paste0("^rho", k, "_"), col_names)))
  gamma_b1_cols <- which(grepl("^gamma_b1_", col_names))
  gamma_delta_cols_list <- lapply(seq_len(n_bp), function(k) which(grepl(paste0("^gamma_delta", k, "_"), col_names)))

  fitted_draws <- matrix(0, nrow = n_draws, ncol = n)
  for (s in seq_len(n_draws)) {
    mu_i <- as.vector(dm$X_b0 %*% as.numeric(draw_mat[s, b0_cols]))
    beta_b1 <- as.numeric(draw_mat[s, b1_cols])
    if (length(gamma_b1_cols) > 0) beta_b1 <- beta_b1 * as.numeric(draw_mat[s, gamma_b1_cols])
    b1_vals <- as.vector(dm$X_b1 %*% beta_b1)
    
    if (n_bp > 0) {
      om1_i <- as.vector(dm$X_om[[1]] %*% as.numeric(draw_mat[s, om_cols_list[[1]]]))
      mu_i  <- mu_i + b1_vals * (tau - om1_i)
    } else {
      mu_i  <- mu_i + b1_vals * tau
    }
    for (k in seq_len(n_bp)) {
      b_delta <- as.numeric(draw_mat[s, delta_cols_list[[k]]])
      if (length(gamma_delta_cols_list[[k]]) > 0) b_delta <- b_delta * as.numeric(draw_mat[s, gamma_delta_cols_list[[k]]])
      delta_i <- as.vector(dm$X_deltas[[k]] %*% b_delta)
      om_i    <- as.vector(dm$X_om[[k]] %*% as.numeric(draw_mat[s, om_cols_list[[k]]]))
      rho_i   <- as.vector(dm$X_rho[[k]] %*% as.numeric(draw_mat[s, rho_cols_list[[k]]]))
      di <- tau - om_i
      si <- 1 / (1 + exp(-di * rho_i))
      mu_i <- mu_i + delta_i * di * si
    }
    if (dm$n_groups_b0 > 0) {
      u_b0 <- as.numeric(draw_mat[s, u_cols])
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

.build_newdata_dm <- function(object, newdata) {
  .safe_mk_mm <- function(f, dat, train_dat) {
    if (inherits(f, "smoothbp_fixed")) {
      X <- matrix(as.numeric(f), nrow = nrow(dat), ncol = 1)
      colnames(X) <- "(Intercept)"
      attr(X, "re_mask") <- 0L
      if (length(f) > 1) {
        attr(X, "fixed_value") <- 1.0
      } else {
        X[] <- 1.0
        attr(X, "fixed_value") <- as.numeric(f)
      }
      return(X)
    }
    .mk_mm_single(.parse_re(f)$fixed, dat, train_dat)
  }

  mk_mm_list <- function(fml_list, dat) {
    if (!is.list(fml_list)) fml_list <- list(fml_list)
    lapply(fml_list, function(f) .safe_mk_mm(f, dat, object$data))
  }
  
  X_b0     <- .safe_mk_mm(object$b0_formula, newdata, object$data)
  X_b1     <- .safe_mk_mm(object$b1_formula, newdata, object$data)
  X_deltas <- mk_mm_list(object$deltas_formula, newdata)
  X_om     <- mk_mm_list(object$omega_formula,  newdata)
  X_rho    <- mk_mm_list(object$rho_formula,    newdata)

  if (inherits(object$b0_formula, "smoothbp_fixed")) {
    re_var <- NULL
  } else {
    re_var <- .parse_re(object$b0_formula)$re_group
  }
  
  group_levels_b0 <- object$dm$group_levels_b0
  if (!is.null(re_var) && re_var %in% names(newdata)) {
    gfac <- factor(newdata[[re_var]], levels = group_levels_b0)
    group_b0 <- ifelse(is.na(gfac), -1L, as.integer(gfac) - 1L)
  } else {
    group_b0 <- rep(-1L, nrow(newdata))
  }
  list(
    X_b0 = X_b0, X_b1 = X_b1, X_deltas = X_deltas, X_om = X_om, X_rho = X_rho,
    group_b0 = group_b0, n_groups_b0 = object$dm$n_groups_b0,
    group_levels_b0 = group_levels_b0
  )
}

.mk_mm_single <- function(fml, dat, train_dat) {
  for (col in names(dat)) {
    if (col %in% names(train_dat) && is.factor(train_dat[[col]])) {
      dat[[col]] <- factor(dat[[col]], levels = levels(train_dat[[col]]))
    }
  }
  stats::model.matrix(fml, data = dat)
}

#' @rdname log_lik
#' @param object A `smoothbp_fit` object.
#' @param ... Unused.
#' @export
log_lik.smoothbp_fit <- function(object, ...) {
  y_obs <- as.double(object$data[[object$response]])
  fit_draws <- fitted(object, summary = FALSE)
  sigma_draws <- as.numeric(posterior::as_draws_matrix(object$draws)[, "sigma"])
  ll_matrix <- matrix(0, nrow = nrow(fit_draws), ncol = length(y_obs))
  for (i in seq_along(y_obs)) {
    ll_matrix[, i] <- stats::dnorm(y_obs[i], mean = fit_draws[, i], sd = sigma_draws, log = TRUE)
  }
  ll_matrix
}

#' @importFrom loo loo waic
#' @export
loo.smoothbp_fit <- function(x, ...) {
  loo::loo(log_lik(x), ...)
}

#' @export
waic.smoothbp_fit <- function(x, ...) {
  loo::waic(log_lik(x), ...)
}

#' @importFrom bayesplot pp_check
#' @export
pp_check.smoothbp_fit <- function(object, n_draws = 50, ...) {
  y_obs <- as.double(object$data[[object$response]])
  fit_mat <- fitted(object, summary = FALSE)
  sigma_draws <- as.numeric(posterior::as_draws_matrix(object$draws)[, "sigma"])
  idx <- sample(nrow(fit_mat), min(n_draws, nrow(fit_mat)))
  y_rep <- do.call(rbind, lapply(idx, function(s) stats::rnorm(length(y_obs), mean = fit_mat[s, ], sd = sigma_draws[s])))
  bayesplot::ppc_dens_overlay(y_obs, y_rep)
}
