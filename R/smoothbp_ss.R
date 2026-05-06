#' Fit a smooth change-point model with spike-and-slab variable selection
#'
#' @param formula A two-sided formula.
#' @param b0 One-sided formula for b0.
#' @param b1 One-sided formula for b1.
#' @param deltas List of formulas for slope changes.
#' @param omega List of formulas for change-points.
#' @param rho List of formulas for sharpness.
#' @param data A data frame.
#' @param priors A \code{\link{smoothbp_priors}} object.
#' @param spike A [prior_spike_slab()] object.
#' @param b1_spike Logical; should b1 coefficients be eligible for spike-and-slab?
#' @param chains Number of chains.
#' @param iter Total iterations.
#' @param warmup Warmup iterations.
#' @param seed Random seed.
#' @param step_om,step_rho,target_accept HMC/MH tuning parameters.
#' @param cores Number of CPU cores.
#' @param .verbose Print progress.
#'
#' @return A \code{smoothbp_fit} object.
#' @export
smoothbp_ss <- function(
    formula,
    b0     = ~ 1,
    b1     = ~ 1,
    deltas = list(~ 1),
    omega  = list(~ 1),
    rho    = list(~ 1),
    data,
    priors = smoothbp_priors(),
    spike  = prior_spike_slab(),
    b1_spike = FALSE,
    chains = 4L,
    iter   = 2000L,
    warmup = 1000L,
    seed   = NULL,
    step_om  = 0.3,
    step_rho = 0.3,
    target_accept = 0.65,
    cores    = getOption("smoothbp.cores", 1L),
    .verbose = TRUE
) {
  if (!inherits(formula, "formula") || length(formula) != 3L) {
    stop("`formula` must be a two-sided formula.")
  }
  
  response_name <- deparse(formula[[2]])
  time_name     <- deparse(formula[[3]])
  
  if (!response_name %in% names(data)) {
    stop(sprintf("Response variable '%s' not found in data.", response_name))
  }
  if (!time_name %in% names(data) && time_name != "1") {
    stop(sprintf("Time variable '%s' not found in data. Did you mean to use 'time'?", time_name))
  }

  y   <- as.double(data[[response_name]])
  tau <- as.double(data[[time_name]])
  
  if (is.null(seed)) seed <- sample.int(.Machine$integer.max, 1L)

  if (.verbose) message("Building design matrices...")
  dm <- .build_design_matrices(b0, b1, deltas, omega, rho, data)
  
  # Effective priors: override slab components
  priors_effective <- priors
  priors_effective$b1 <- if (b1_spike) spike$slab else priors$b1
  priors_effective$deltas <- spike$slab
  
  pv <- .build_prior_vectors(priors_effective, dm)

  # Build spike masks
  b1_mask <- integer(length(dm$col_names_b1))
  if (b1_spike) {
    b1_mask[] <- 1L
    # Usually we don't spike the intercept
    b1_mask[dm$col_names_b1 == "(Intercept)"] <- 0L
  }
  
  delta_masks <- lapply(dm$col_names_deltas, function(nms) {
    # Spike ALL delta parameters including the intercept.
    # The delta intercept IS the slope-change magnitude to be regularized.
    # (Unlike b1 where we keep the baseline slope, a zero delta = no breakpoint.)
    as.integer(rep(1L, length(nms)))
  })

  if (.verbose) message("Running sampler...")
  raw <- run_mcmc_ss(
    y             = y,
    tau           = tau,
    x_b0          = as.double(dm$X_b0),  p_b0  = ncol(dm$X_b0),
    x_b1          = as.double(dm$X_b1),  p_b1  = ncol(dm$X_b1),
    x_deltas      = lapply(dm$X_deltas, as.double),
    p_deltas      = as.integer(sapply(dm$X_deltas, ncol)),
    x_om          = lapply(dm$X_om, as.double),
    p_om          = as.integer(sapply(dm$X_om, ncol)),
    x_rho         = lapply(dm$X_rho, as.double),
    p_rho         = as.integer(sapply(dm$X_rho, ncol)),
    group_b0      = dm$group_b0,
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
    step_om  = step_om,
    step_rho = step_rho,
    target_accept = as.double(target_accept),
    b1_spike_mask = as.integer(b1_mask),
    delta_spike_mask = delta_masks,
    pi_init       = as.double(spike$pi[1]),
    pi_beta_a     = if (isTRUE(spike$learn_pi)) spike$a else 0.0,
    pi_beta_b     = if (isTRUE(spike$learn_pi)) spike$b else 0.0,
    chains   = as.integer(chains),
    iter     = as.integer(iter),
    warmup   = as.integer(warmup),
    seed     = as.integer(seed),
    verbose  = isTRUE(.verbose),
    n_cores  = as.integer(max(1L, cores))
  )

  # Labeling
  base_pnames <- .param_names(dm, pv)
  # gamma_b1 is always written to draws by the Rust sampler (even when b1_spike=FALSE,
  # the indicator is tracked but frozen at 1). Always include the names.
  gamma_names <- paste0("gamma_b1_", dm$col_names_b1)
  for (i in seq_along(dm$col_names_deltas)) {
    gamma_names <- c(gamma_names, paste0("gamma_delta", i, "_", dm$col_names_deltas[[i]]))
  }
  
  pnames <- c(base_pnames, gamma_names)
  if (isTRUE(spike$learn_pi)) pnames <- c(pnames, "pi")
  
  n_post <- nrow(raw$draws[[1]])
  n_params <- ncol(raw$draws[[1]])
  chain_arr <- array(
    data     = unlist(lapply(raw$draws, function(m) t(m))),
    dim      = c(n_params, n_post, chains),
    dimnames = list(variable = pnames, draw = NULL, chain = NULL)
  )
  da <- posterior::as_draws_array(aperm(chain_arr, c(2, 3, 1)))

  structure(
    list(
      draws         = da,
      formula       = formula,
      response      = response_name,
      time          = time_name,
      data          = data,
      dm            = dm,
      pv            = pv,
      b0_formula    = b0,
      b1_formula    = b1,
      deltas_formula = deltas,
      omega_formula  = omega,
      rho_formula    = rho,
      gamma_names   = gamma_names,
      chains        = as.integer(chains),
      iter          = as.integer(iter),
      warmup        = as.integer(warmup)
    ),
    class = c("smoothbp_ss_fit", "smoothbp_fit")
  )
}
