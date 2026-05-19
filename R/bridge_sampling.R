# Bridge sampling and Bayes factor methods for smoothbp_fit

.log_dinvgamma2 <- function(x_sq, shape, scale) {
  shape * log(scale) - lgamma(shape) - (shape + 1) * log(x_sq) - scale / x_sq
}

.log_dinvgamma_sigma <- function(sigma, shape, scale) {
  # exact prior for sigma when sigma^2 ~ InvGamma
  # f(sigma) = 2 * scale^shape / Gamma(shape) * sigma^(-2*shape - 1) * exp(-scale / sigma^2)
  log(2) + shape * log(scale) - lgamma(shape) - (2 * shape + 1) * log(sigma) - scale / (sigma^2)
}

.log_dtnorm <- function(x, mean, sd, lb = -Inf, ub = Inf) {
  if (is.finite(lb) && x < lb) return(-Inf)
  if (is.finite(ub) && x > ub) return(-Inf)
  log_p <- stats::dnorm(x, mean, sd, log = TRUE)
  if (is.finite(lb) || is.finite(ub)) {
    log_norm <- log(stats::pnorm(ub, mean, sd) - stats::pnorm(lb, mean, sd))
    log_p <- log_p - log_norm
  }
  log_p
}

.smoothbp_param_bounds <- function(fit) {
  dm     <- fit$dm
  priors <- fit$priors
  n_bp   <- length(dm$X_deltas)

  get_bounds <- function(p_list, nms, prefix) {
    lb <- stats::setNames(p_list$lb, paste0(prefix, nms))
    ub <- stats::setNames(p_list$ub, paste0(prefix, nms))
    list(lb = lb, ub = ub)
  }

  b0_b  <- get_bounds(fit$pv$b0, colnames(dm$X_b0), "b0_")
  b1_b  <- get_bounds(fit$pv$b1, colnames(dm$X_b1), "b1_")
  
  lb <- c(b0_b$lb, b1_b$lb)
  ub <- c(b0_b$ub, b1_b$ub)

  for (k in seq_len(n_bp)) {
    d_b  <- get_bounds(fit$pv$deltas[[k]], colnames(dm$X_deltas[[k]]), paste0("delta", k, "_"))
    o_b  <- get_bounds(fit$pv$om[[k]],     colnames(dm$X_om[[k]]),     paste0("omega", k, "_"))
    r_b  <- get_bounds(fit$pv$rho[[k]],    colnames(dm$X_rho[[k]]),    paste0("rho", k, "_"))
    lb <- c(lb, d_b$lb, o_b$lb, r_b$lb)
    ub <- c(ub, d_b$ub, o_b$ub, r_b$ub)
  }

  lb["sigma"] <- 0; ub["sigma"] <- Inf
  if (dm$n_groups_b0 > 0) {
    u_names <- paste0("u[", dm$group_levels_b0, "]")
    lb[u_names] <- -Inf; ub[u_names] <- Inf
    lb["sigma_u"] <- 0; ub["sigma_u"] <- Inf
  }
  
  # Gammas are fixed at 0/1 during bridge sampling (or rather, bridge sampling 
  # usually handles continuous parameters; discrete parameters like gammas 
  # are tricky. For Bayes Factor between models with different gammas, 
  # usually we compare models with fixed gammas or marginalize).
  # Here, we assume bridge sampling over the continuous parameters 
  # conditioned on gammas, or we include gammas as continuous 0/1 (risky).
  # Actually, smoothbp_ss uses Kuo-Mallick where gammas are sampled.
  # For bridge sampling, we'll treat them as fixed for a specific model 
  # or include them if the user really wants. 
  # Given the complexity, I'll exclude them from the bounds and 
  # assume they are handled by the caller or filtered.
  
  list(lb = lb, ub = ub)
}

.smoothbp_log_posterior <- function(pars, data_list) {
  dm     <- data_list$dm
  y      <- data_list$y
  tau    <- data_list$tau
  pv     <- data_list$pv
  n_bp   <- length(dm$X_deltas)
  sigma  <- pars["sigma"]
  if (is.na(sigma) || sigma <= 0) return(-Inf)

  # Reconstruct mu
  mu_i <- as.vector(dm$X_b0 %*% pars[paste0("b0_", colnames(dm$X_b0))])
  
  # b1
  beta_b1 <- pars[paste0("b1_", colnames(dm$X_b1))]
  # Apply gamma if present in pars
  g_b1_nms <- paste0("gamma_b1_", colnames(dm$X_b1))
  if (all(g_b1_nms %in% names(pars))) beta_b1 <- beta_b1 * pars[g_b1_nms]
  
  b1_vals <- as.vector(dm$X_b1 %*% beta_b1)

  if (n_bp > 0) {
    om1_i <- as.vector(dm$X_om[[1]] %*% pars[paste0("omega1_", colnames(dm$X_om[[1]]))])
    mu_i  <- mu_i + b1_vals * (tau - om1_i)
  } else {
    mu_i  <- mu_i + b1_vals * tau
  }

  for (k in seq_len(n_bp)) {
    bd <- pars[paste0("delta", k, "_", colnames(dm$X_deltas[[k]]))]
    g_dk_nms <- paste0("gamma_delta", k, "_", colnames(dm$X_deltas[[k]]))
    if (all(g_dk_nms %in% names(pars))) bd <- bd * pars[g_dk_nms]
    
    om_k  <- as.vector(dm$X_om[[k]] %*% pars[paste0("omega", k, "_", colnames(dm$X_om[[k]]))])
    rho_k <- as.vector(dm$X_rho[[k]] %*% pars[paste0("rho", k, "_", colnames(dm$X_rho[[k]]))])
    delta_k <- as.vector(dm$X_deltas[[k]] %*% bd)
    
    di <- tau - om_k
    si <- 1 / (1 + exp(-di * rho_k))
    mu_i <- mu_i + delta_k * di * si
  }

  if (dm$n_groups_b0 > 0) {
    u_vals <- pars[paste0("u[", dm$group_levels_b0, "]")]
    for (i in seq_along(y)) {
      g <- dm$group_b0[i]
      if (g >= 0L) mu_i[i] <- mu_i[i] + u_vals[g + 1L]
    }
  }

  ll <- sum(stats::dnorm(y, mu_i, sigma, log = TRUE))
  if (!is.finite(ll)) return(-Inf)

  # Priors
  lp <- 0
  log_p_block <- function(vals, p_obj) {
    sum(vapply(seq_along(vals), function(i) .log_dtnorm(vals[i], p_obj$mean[i], p_obj$sd[i], p_obj$lb[i], p_obj$ub[i]), numeric(1)))
  }
  
  lp <- lp + log_p_block(pars[paste0("b0_", colnames(dm$X_b0))], pv$b0)
  lp <- lp + log_p_block(pars[paste0("b1_", colnames(dm$X_b1))], pv$b1)
  for (k in seq_len(n_bp)) {
    lp <- lp + log_p_block(pars[paste0("delta", k, "_", colnames(dm$X_deltas[[k]]))], pv$deltas[[k]])
    lp <- lp + log_p_block(pars[paste0("omega", k, "_", colnames(dm$X_om[[k]]))],     pv$om[[k]])
    lp <- lp + .log_dtnorm(pars[paste0("rho", k, "_", colnames(dm$X_rho[[k]]))],     pv$rho[[k]]$mean, pv$rho[[k]]$sd, pv$rho[[k]]$lb, pv$rho[[k]]$ub)
  }
  lp <- lp + .log_dinvgamma_sigma(sigma, data_list$sigma_p$shape, data_list$sigma_p$scale)

  if (dm$n_groups_b0 > 0) {
    su <- pars["sigma_u"]
    lp <- lp + sum(stats::dnorm(u_vals, 0, su, log = TRUE))
    lp <- lp + .log_dinvgamma_sigma(su, data_list$sigma_u_p$shape, data_list$sigma_u_p$scale)
  }
  
  ll + lp
}

#' Bridge Sampler for smoothbp_fit
#'
#' @param samples A \code{smoothbp_fit} object.
#' @param method Character; either "auto", "rust", or "bridgesampling". Default "auto" uses Rust for continuous models.
#' @param seed Random seed for the bridge sampler.
#' @param ... Passed to \code{\link[bridgesampling]{bridge_sampler}}.
#'
#' @importFrom bridgesampling bridge_sampler
#' @method bridge_sampler smoothbp_fit
#' @export
bridge_sampler.smoothbp_fit <- function(samples, method = c("auto", "rust", "bridgesampling"), seed = 42, ...) {
  method <- match.arg(method)
  if (!requireNamespace("bridgesampling", quietly = TRUE)) stop("Install 'bridgesampling'.")
  
  draw_mat <- as.matrix(posterior::as_draws_matrix(samples$draws))
  param_names <- colnames(draw_mat)
  
  is_spike_slab <- any(grepl("^gamma_", param_names))
  has_re_om <- isTRUE(samples$dm$has_re_om)
  
  if (method == "auto") {
    if (!is_spike_slab && !has_re_om) method <- "rust" else method <- "bridgesampling"
  }
  
  if (method == "rust") {
    if (is_spike_slab || has_re_om) {
      warning("Rust bridge sampling does not fully support spike-and-slab or om random effects yet. Falling back to bridgesampling.")
      method <- "bridgesampling"
    } else {
      dm <- samples$dm
      pv <- samples$pv
      priors <- samples$priors
      
      y <- as.double(samples$data[[samples$response]])
      tau <- as.double(samples$data[[samples$time]])
      
      .safe_int <- function(x) if (length(x) == 0) -1L else as.integer(x)
      p_deltas_safe <- .safe_int(sapply(dm$X_deltas, ncol))
      p_om_safe     <- .safe_int(sapply(dm$X_om, ncol))
      p_rho_safe    <- .safe_int(sapply(dm$X_rho, ncol))
      group_b0_safe <- .safe_int(dm$group_b0)
      
      pnames <- .param_names(dm, pv)
      draw_mat_ordered <- draw_mat[, pnames, drop = FALSE]
      
      log_ml <- run_bridge(
        y             = y,
        tau           = tau,
        x_b0          = as.double(dm$X_b0),  p_b0  = ncol(dm$X_b0),
        x_b1          = as.double(dm$X_b1),  p_b1  = ncol(dm$X_b1),
        x_deltas      = lapply(dm$X_deltas, as.double),
        p_deltas      = p_deltas_safe,
        x_om          = lapply(dm$X_om, as.double),
        p_om          = p_om_safe,
        x_rho         = lapply(dm$X_rho, as.double),
        p_rho         = p_rho_safe,
        group_b0      = group_b0_safe,
        n_groups_b0   = dm$n_groups_b0,
        prior_mean_b0 = pv$b0$mean, prior_sd_b0 = pv$b0$sd, prior_lb_b0 = pv$b0$lb, prior_ub_b0 = pv$b0$ub,
        prior_mean_b1 = pv$b1$mean, prior_sd_b1 = pv$b1$sd, prior_lb_b1 = pv$b1$lb, prior_ub_b1 = pv$b1$ub,
        prior_mean_deltas = lapply(pv$deltas, `[[`, "mean"),
        prior_sd_deltas   = lapply(pv$deltas, `[[`, "sd"),
        prior_lb_deltas   = lapply(pv$deltas, `[[`, "lb"),
        prior_ub_deltas   = lapply(pv$deltas, `[[`, "ub"),
        prior_mean_om     = lapply(pv$om, `[[`, "mean"),
        prior_sd_om       = lapply(pv$om, `[[`, "sd"),
        prior_lb_om       = lapply(pv$om, `[[`, "lb"),
        prior_ub_om       = lapply(pv$om, `[[`, "ub"),
        prior_mean_rho    = lapply(pv$rho, `[[`, "mean"),
        prior_sd_rho      = lapply(pv$rho, `[[`, "sd"),
        prior_lb_rho      = lapply(pv$rho, `[[`, "lb"),
        prior_ub_rho      = lapply(pv$rho, `[[`, "ub"),
        sigma_shape   = priors$sigma$shape,
        sigma_scale   = priors$sigma$scale,
        sigma_u_shape = priors$sigma_u$shape,
        sigma_u_scale = priors$sigma_u$scale,
        mcmc_draws    = draw_mat_ordered,
        seed          = as.integer(seed)
      )
      
      return(structure(
        list(
          logml = log_ml,
          niter = 1000,
          method = "rust"
        ),
        class = "bridge"
      ))
    }
  }
  
  if (method == "bridgesampling") {
    lb_vec <- stats::setNames(rep(-Inf, length(param_names)), param_names)
    ub_vec <- stats::setNames(rep( Inf, length(param_names)), param_names)
    bounds <- .smoothbp_param_bounds(samples)
    for (nm in names(bounds$lb)) {
      if (nm %in% param_names) { lb_vec[[nm]] <- bounds$lb[[nm]]; ub_vec[[nm]] <- bounds$ub[[nm]] }
    }
  
    data_list <- list(
      dm = samples$dm, y = as.double(samples$data[[samples$response]]),
      tau = as.double(samples$data[[samples$time]]), pv = samples$pv,
      sigma_p = samples$priors$sigma, sigma_u_p = samples$priors$sigma_u
    )
    
    return(bridgesampling::bridge_sampler(
      samples = draw_mat, 
      log_posterior = function(pars, data) {
         names(pars) <- colnames(draw_mat)
         .smoothbp_log_posterior(pars, data)
      },
      data = data_list, lb = lb_vec, ub = ub_vec, ...
    ))
  }
}

#' Bayes Factor for smoothbp_fit
#'
#' @param x1 A \code{smoothbp_fit} object.
#' @param x2 A \code{smoothbp_fit} object.
#' @param log Logical; if TRUE, return log Bayes Factor.
#' @param ... Passed to \code{\link[bridgesampling]{bridge_sampler}}.
#'
#' @importFrom bridgesampling bayes_factor
#' @method bayes_factor smoothbp_fit
#' @export
bayes_factor.smoothbp_fit <- function(x1, x2, log = FALSE, ...) {
  bs1 <- bridgesampling::bridge_sampler(x1, ...)
  bs2 <- bridgesampling::bridge_sampler(x2, ...)
  bridgesampling::bayes_factor(bs1, bs2, log = log)
}
