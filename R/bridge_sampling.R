# Bridge sampling and Bayes factor methods for smoothbp_fit
#
# Enables Bayes factor comparisons between smoothbp_fit objects and between
# smoothbp_fit and brmsfit objects (or any model with a bridge_sampler method).

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Log density of an inverse-gamma distribution.
# Parameterisation: p(x) ∝ x^{-(shape+1)} * exp(-scale / x),  x > 0
.log_dinvgamma <- function(x, shape, scale) {
  shape * log(scale) - lgamma(shape) - (shape + 1) * log(x) - scale / x
}

# Log density of a (truncated) normal distribution.
# When lb and ub are both infinite, this is just dnorm(..., log=TRUE).
.log_dtnorm <- function(x, mean, sd, lb = -Inf, ub = Inf) {
  log_p <- stats::dnorm(x, mean, sd, log = TRUE)
  if (is.finite(lb) || is.finite(ub)) {
    log_norm <- log(stats::pnorm(ub, mean, sd) - stats::pnorm(lb, mean, sd))
    log_p <- log_p - log_norm
  }
  log_p
}

# ---------------------------------------------------------------------------
# .smoothbp_param_bounds()
#
# Returns a list(lb = named_vec, ub = named_vec) covering every parameter
# in the draws matrix, used by bridge_sampler to handle bounded parameters.
# ---------------------------------------------------------------------------
.smoothbp_param_bounds <- function(fit) {

  dm      <- fit$dm
  priors  <- fit$priors

  # Helper: expand a normal prior spec to per-coefficient lb/ub
  bounds_from_prior <- function(prior_spec, coef_names) {
    expanded <- .expand_prior(prior_spec, coef_names)
    list(lb = stats::setNames(expanded$lb, expanded$name),
         ub = stats::setNames(expanded$ub, expanded$name))
  }

  # Collect bounds for each parameter block
  b0_coefs  <- colnames(dm$X_b0)
  b1_coefs  <- colnames(dm$X_b1)
  b2_coefs  <- colnames(dm$X_b2)
  om_coefs  <- colnames(dm$X_om)
  rho_coefs <- colnames(dm$X_rho)

  # Prefix coefficient names to match draw variable names
  prefix <- function(nms, pre) stats::setNames(nms, paste0(pre, nms))

  b0_bounds  <- bounds_from_prior(priors$b0,    b0_coefs)
  b1_bounds  <- bounds_from_prior(priors$b1,    b1_coefs)
  b2_bounds  <- bounds_from_prior(priors$b2,    b2_coefs)
  om_bounds  <- bounds_from_prior(priors$omega, om_coefs)
  rho_bounds <- bounds_from_prior(priors$rho,   rho_coefs)

  # Rename to match draw variable names (e.g. "b0_(Intercept)")
  rename_bounds <- function(bounds, prefix_str) {
    names(bounds$lb) <- paste0(prefix_str, names(bounds$lb))
    names(bounds$ub) <- paste0(prefix_str, names(bounds$ub))
    bounds
  }

  b0_bounds  <- rename_bounds(b0_bounds,  "b0_")
  b1_bounds  <- rename_bounds(b1_bounds,  "b1_")
  b2_bounds  <- rename_bounds(b2_bounds,  "b2_")
  om_bounds  <- rename_bounds(om_bounds,  "omega_")
  rho_bounds <- rename_bounds(rho_bounds, "rho_")

  lb <- c(b0_bounds$lb, b1_bounds$lb, b2_bounds$lb,
          om_bounds$lb, rho_bounds$lb)
  ub <- c(b0_bounds$ub, b1_bounds$ub, b2_bounds$ub,
          om_bounds$ub, rho_bounds$ub)

  # sigma: InvGamma => lb = 0
  lb["sigma"] <- 0; ub["sigma"] <- Inf

  # Random effects and sigma_u
  if (dm$n_groups_b0 > 0) {
    u_names <- paste0("u[", seq_len(dm$n_groups_b0), "]")
    lb[u_names] <- -Inf
    ub[u_names] <-  Inf
    lb["sigma_u"] <- 0; ub["sigma_u"] <- Inf
  }

  list(lb = lb, ub = ub)
}

# ---------------------------------------------------------------------------
# .smoothbp_log_posterior(pars, data_list)
#
# Log unnormalized posterior for a single draw vector.
# `pars`      : named numeric vector (one row of the draws matrix)
# `data_list` : list with everything needed (stored inside bridge_sampler call)
# ---------------------------------------------------------------------------
.smoothbp_log_posterior <- function(pars, data_list) {

  dm        <- data_list$dm
  y         <- data_list$y
  priors    <- data_list$priors
  b0_coefs  <- data_list$b0_coefs
  b1_coefs  <- data_list$b1_coefs
  b2_coefs  <- data_list$b2_coefs
  om_coefs  <- data_list$om_coefs
  rho_coefs <- data_list$rho_coefs
  tau       <- data_list$tau
  n_groups  <- dm$n_groups_b0

  # ---- Extract parameter blocks from named vector --------------------------
  beta_b0  <- pars[paste0("b0_",    b0_coefs)]
  beta_b1  <- pars[paste0("b1_",    b1_coefs)]
  beta_b2  <- pars[paste0("b2_",    b2_coefs)]
  beta_om  <- pars[paste0("omega_", om_coefs)]
  beta_rho <- pars[paste0("rho_",   rho_coefs)]
  sigma    <- pars["sigma"]

  # ---- Reconstruct mu_i ----------------------------------------------------
  omega_i <- as.vector(dm$X_om  %*% beta_om)
  rho_i   <- as.vector(dm$X_rho %*% beta_rho)
  b0_i    <- as.vector(dm$X_b0  %*% beta_b0)
  b1_i    <- as.vector(dm$X_b1  %*% beta_b1)
  b2_i    <- as.vector(dm$X_b2  %*% beta_b2)

  d_i  <- tau - omega_i
  s_i  <- 1 / (1 + exp(-d_i * rho_i))
  mu_i <- b0_i + b1_i * d_i + b2_i * d_i * s_i

  # Add random intercepts where available
  if (n_groups > 0) {
    u_vals <- pars[paste0("u[", seq_len(n_groups), "]")]
    for (i in seq_along(y)) {
      g <- dm$group_b0[i]
      if (g >= 0L) mu_i[i] <- mu_i[i] + u_vals[g + 1L]
    }
  }

  # ---- Log likelihood -------------------------------------------------------
  ll <- sum(stats::dnorm(y, mean = mu_i, sd = sigma, log = TRUE))

  # ---- Log priors -----------------------------------------------------------
  lp <- 0

  # Helper: sum log density for a block of normal-prior coefficients
  log_prior_block <- function(vals, prior_spec, coef_names) {
    expanded <- .expand_prior(prior_spec, coef_names)
    total <- 0
    for (k in seq_along(vals)) {
      total <- total + .log_dtnorm(
        vals[k], expanded$mean[k], expanded$sd[k],
        expanded$lb[k], expanded$ub[k]
      )
    }
    total
  }

  lp <- lp + log_prior_block(beta_b0,  priors$b0,    b0_coefs)
  lp <- lp + log_prior_block(beta_b1,  priors$b1,    b1_coefs)
  lp <- lp + log_prior_block(beta_b2,  priors$b2,    b2_coefs)
  lp <- lp + log_prior_block(beta_om,  priors$omega, om_coefs)
  lp <- lp + log_prior_block(beta_rho, priors$rho,   rho_coefs)

  # sigma ~ InvGamma
  lp <- lp + .log_dinvgamma(sigma, priors$sigma$shape, priors$sigma$scale)

  # Random effects: only include when the model actually has random effects.
  # When n_groups == 0, sigma_u is held fixed at 1 by the sampler and is not
  # present in pars (it was stripped before calling bridge_sampler).
  if (n_groups > 0 && "sigma_u" %in% names(pars)) {
    sigma_u <- pars["sigma_u"]
    u_vals  <- pars[paste0("u[", seq_len(n_groups), "]")]

    # u[i] ~ N(0, sigma_u)
    lp <- lp + sum(stats::dnorm(u_vals, mean = 0, sd = sigma_u, log = TRUE))

    # sigma_u ~ InvGamma
    lp <- lp + .log_dinvgamma(sigma_u,
                               priors$sigma_u$shape,
                               priors$sigma_u$scale)
  }

  ll + lp
}

# ---------------------------------------------------------------------------
# bridge_sampler.smoothbp_fit
# ---------------------------------------------------------------------------

#' Bridge sampling for smoothbp_fit objects
#'
#' Computes the log marginal likelihood of a `smoothbp_fit` model via bridge
#' sampling. The result is a `bridge` object compatible with
#' [bridgesampling::bayes_factor()], allowing direct comparison with other
#' Bayesian models including those fitted with **brms**.
#'
#' @param samples A `smoothbp_fit` object.
#' @param repetitions Number of bridge-sampling repetitions. Default `1`.
#' @param method Bridge-sampling method: `"normal"` (default) or `"warp3"`.
#' @param cores Number of cores for evaluating the log posterior. Default `1`.
#' @param use_neff Logical; use effective sample size in bridge function.
#'   Default `TRUE`.
#' @param maxiter Maximum iterations for the iterative scheme. Default `1000`.
#' @param silent Logical; suppress iteration printing. Default `FALSE`.
#' @param verbose Logical; print internal debug info. Default `FALSE`.
#' @param ... Additional arguments passed to
#'   [bridgesampling::bridge_sampler()].
#'
#' @return A `bridge` object from the **bridgesampling** package. Use
#'   [bridgesampling::logml()] to extract the log marginal likelihood and
#'   [bridgesampling::bayes_factor()] to compare two models.
#'
#' @details
#' Bridge sampling requires evaluating the log unnormalized posterior density
#' at each posterior draw. Internally, this function reconstructs the model's
#' linear predictors in R using the stored design matrices and computes the
#' sum of the log likelihood and all log prior densities.
#'
#' For comparison with a **brms** model, the brms model must have been fitted
#' with `save_pars = save_pars(all = TRUE)`.
#'
#' ## Checking reliability
#'
#' Bridge sampling is a stochastic approximation.  Always check the
#' uncertainty of the estimate with [bridgesampling::error_measures()]:
#'
#' ```r
#' bs <- bridge_sampler(fit)
#' error_measures(bs)  # inspect $cv (coefficient of variation)
#' ```
#'
#' A coefficient of variation (`$cv`) below 0.05 (5%) indicates a reliable
#' estimate.  Values above ~0.15 suggest the posterior was poorly explored
#' — check `trace_plot(fit)` and consider increasing `iter`.  When
#' convergence is poor, [loo()] is a more robust comparison tool.
#'
#' @examples
#' \dontrun{
#' library(brms)
#'
#' fit_piece  <- smoothbp(y ~ tau, b0 = ~1 + (1|subject), data = d)
#' fit_linear <- brm(y ~ tau + (1|subject), data = d,
#'                   save_pars = save_pars(all = TRUE))
#'
#' # Compute log marginal likelihoods
#' bs_piece  <- bridge_sampler(fit_piece)
#' bs_linear <- bridge_sampler(fit_linear)
#'
#' # Assess reliability of the estimate before interpreting
#' bridgesampling::error_measures(bs_piece)   # check $cv < 0.05 ideally
#'
#' # Bayes factor: piecewise vs linear
#' bridgesampling::bayes_factor(bs_piece, bs_linear)
#' }
#' @export
bridge_sampler.smoothbp_fit <- function(
    samples,
    repetitions = 1,
    method      = "normal",
    cores       = 1,
    use_neff    = TRUE,
    maxiter     = 1000,
    silent      = FALSE,
    verbose     = FALSE,
    ...
) {
  if (!requireNamespace("bridgesampling", quietly = TRUE)) {
    stop(
      "The 'bridgesampling' package is required. ",
      "Install it with: install.packages('bridgesampling')"
    )
  }

  fit <- samples  # rename for clarity

  # ---- Posterior draws as plain matrix -------------------------------------
  draw_mat   <- posterior::as_draws_matrix(fit$draws)
  samp_mat   <- as.matrix(draw_mat)  # strips draws_matrix class -> base matrix

  # When there are no random effects, sigma_u is held at a dummy constant (1)
  # by the sampler and is not a free parameter -- exclude it from bridge
  # sampling along with any u[i] columns.
  if (fit$dm$n_groups_b0 == 0) {
    re_cols <- grepl("^sigma_u$|^u\\[", colnames(samp_mat))
    samp_mat <- samp_mat[, !re_cols, drop = FALSE]
  }

  # ---- Parameter bounds ----------------------------------------------------
  bounds   <- .smoothbp_param_bounds(fit)
  lb_vec   <- bounds$lb[colnames(samp_mat)]
  ub_vec   <- bounds$ub[colnames(samp_mat)]

  # Replace NA (any unmapped parameters) with -Inf/Inf
  lb_vec[is.na(lb_vec)] <- -Inf
  ub_vec[is.na(ub_vec)] <-  Inf

  # ---- Data list for log posterior -----------------------------------------
  dm <- fit$dm
  data_list <- list(
    dm        = dm,
    y         = as.double(fit$data[[fit$response]]),
    tau       = as.double(fit$data[[fit$time]]),
    priors    = fit$priors,
    b0_coefs  = colnames(dm$X_b0),
    b1_coefs  = colnames(dm$X_b1),
    b2_coefs  = colnames(dm$X_b2),
    om_coefs  = colnames(dm$X_om),
    rho_coefs = colnames(dm$X_rho)
  )

  # ---- Run bridge sampling -------------------------------------------------
  bridgesampling::bridge_sampler(
    samples      = samp_mat,
    log_posterior = .smoothbp_log_posterior,
    data         = data_list,
    lb           = lb_vec,
    ub           = ub_vec,
    repetitions  = repetitions,
    method       = method,
    cores        = cores,
    use_neff     = use_neff,
    maxiter      = maxiter,
    silent       = silent,
    verbose      = verbose,
    ...
  )
}

# ---------------------------------------------------------------------------
# bayes_factor.smoothbp_fit
# ---------------------------------------------------------------------------

#' Bayes factor comparison for smoothbp_fit objects
#'
#' Computes the Bayes factor between a `smoothbp_fit` model and another
#' Bayesian model. The second model (`y`) may be another `smoothbp_fit` or a
#' `brmsfit` object (or any object with a `bridge_sampler` method).
#'
#' @param x A `smoothbp_fit` object (the numerator model).
#' @param y A second model object: another `smoothbp_fit`, a `brmsfit`, or any
#'   object with a `bridge_sampler` method.
#' @param log Logical; if `TRUE` return the log Bayes factor. Default `FALSE`.
#' @param ... Additional arguments passed to `bridge_sampler()`.
#'
#' @return A `bf_bridge` object from the **bridgesampling** package. Print it
#'   to see the Bayes factor (BF10 = evidence for `x` over `y`).
#'
#' @details
#' The Bayes factor BF10 = p(data | model x) / p(data | model y) is computed
#' by dividing the marginal likelihoods obtained via bridge sampling.
#'
#' **Important**: if `y` is a `brmsfit`, it must have been fitted with
#' `save_pars = save_pars(all = TRUE)` so that bridge sampling can access all
#' posterior draws.
#'
#' Because S3 dispatch uses the class of the first argument, this function
#' handles the case where `x` is a `smoothbp_fit`. If your `brmsfit` is the
#' model you want in the numerator, either reverse the argument order and take
#' the reciprocal, or call `bridge_sampler()` on both models and use
#' `bridgesampling::bayes_factor()` directly.
#'
#' ## Checking reliability
#'
#' This function runs bridge sampling internally and does not expose the
#' intermediate `bridge` objects.  To inspect the reliability of each
#' marginal likelihood estimate (via [bridgesampling::error_measures()]),
#' call `bridge_sampler()` on each model separately:
#'
#' ```r
#' bs_x <- bridge_sampler(x)
#' bs_y <- bridge_sampler(y)
#' bridgesampling::error_measures(bs_x)  # check $cv
#' bridgesampling::bayes_factor(bs_x, bs_y)
#' ```
#'
#' A coefficient of variation (`$cv`) below 0.05 is ideal.  Above ~0.15,
#' treat the Bayes factor as approximate and consider increasing `iter`.
#'
#' @examples
#' \dontrun{
#' library(brms)
#'
#' # Fit models
#' fit_piece  <- smoothbp(y ~ tau, b0 = ~1 + (1|subject), data = d)
#' fit_linear <- brm(y ~ tau + (1|subject), data = d,
#'                   save_pars = save_pars(all = TRUE))
#'
#' # Bayes factor: piecewise (numerator) vs linear (denominator)
#' bayes_factor(fit_piece, fit_linear)
#'
#' # Two smoothbp models
#' fit_grouped <- smoothbp(y ~ tau, b0 = ~group + (1|subject), data = d)
#' bayes_factor(fit_piece, fit_grouped)
#' }
#' @export
bayes_factor.smoothbp_fit <- function(x, y, log = FALSE, ...) {

  if (!requireNamespace("bridgesampling", quietly = TRUE)) {
    stop(
      "The 'bridgesampling' package is required. ",
      "Install it with: install.packages('bridgesampling')"
    )
  }

  # Compute bridge samples for the smoothbp numerator model
  message("Computing bridge samples for model 1 (smoothbp_fit)...")
  bs_x <- bridge_sampler(x, ...)

  # Compute bridge samples for the denominator model
  if (inherits(y, "smoothbp_fit")) {
    message("Computing bridge samples for model 2 (smoothbp_fit)...")
    bs_y <- bridge_sampler(y, ...)
  } else if (inherits(y, "brmsfit")) {
    if (!requireNamespace("brms", quietly = TRUE)) {
      stop(
        "The 'brms' package is required to compare against a brmsfit. ",
        "Install it with: install.packages('brms')"
      )
    }
    message("Computing bridge samples for model 2 (brmsfit)...")
    bs_y <- tryCatch(
      brms::bridge_sampler(y, ...),
      error = function(e) {
        stop(
          "Bridge sampling failed for the brmsfit object. ",
          "Make sure it was fitted with save_pars = save_pars(all = TRUE).\n",
          "Original error: ", conditionMessage(e)
        )
      }
    )
  } else {
    # Try a generic bridge_sampler call for other model types
    has_bs <- tryCatch({
      utils::getS3method("bridge_sampler", class(y)[1])
      TRUE
    }, error = function(e) FALSE)

    if (!has_bs) {
      stop(
        "No 'bridge_sampler' method found for an object of class '",
        class(y)[1], "'. ",
        "Supported types: smoothbp_fit, brmsfit, or any class with a ",
        "bridge_sampler method from the bridgesampling package."
      )
    }
    message("Computing bridge samples for model 2 (", class(y)[1], ")...")
    bs_y <- bridge_sampler(y, ...)
  }

  bridgesampling::bayes_factor(bs_x, bs_y, log = log)
}

# ---------------------------------------------------------------------------
# Register bridge_sampler generic if not already available
# (bridgesampling exports it; we just need an importFrom to use it)
# ---------------------------------------------------------------------------

# Make bridge_sampler available as a generic in this package so users
# don't need to library(bridgesampling) first.
#' @importFrom bridgesampling bridge_sampler
NULL

# Make bayes_factor available as a generic in this package.
#' @importFrom bridgesampling bayes_factor
NULL
