#' Fit a smooth change-point model with spike-and-slab variable selection
#'
#' @param formula A two-sided formula.
#' @param b0 One-sided formula for b0.
#' @param b1 One-sided formula for b1.
#' @param deltas List of formulas for slope changes.
#' @param omega List of formulas for change-points. Can also contain \code{\link{fixed}()} values.
#' @param rho List of formulas for sharpness. Can also contain \code{\link{fixed}()} values.
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
#' @param hierarchical Character vector specifying which parameters should be hierarchical. Currently only "omega" is supported.
#' @param reparameterise Character specifying the parameterisation for random change-points:
#'   \code{"none"} (centred) or \code{"omega"} (fully non-centred). Default is \code{"none"}.
#'   Only used if random effects are present.
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
    hierarchical = NULL,
    chains = 4L,
    iter   = 2000L,
    warmup = 1000L,
    seed   = NULL,
    step_om  = 0.3,
    step_rho = 0.3,
    target_accept = 0.9,
    cores    = getOption("smoothbp.cores", 1L),
    reparameterise = c("none", "omega"),
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

  if (anyNA(y))   stop(sprintf("Response variable '%s' contains NA values. Remove or impute missing observations before fitting.", response_name))
  if (anyNA(tau)) stop(sprintf("Time variable '%s' contains NA values.", time_name))

  if (length(tau) == 0L) {
    stop(sprintf("Time variable '%s' is empty or not found. A valid time/covariate column is required on the RHS of the formula.", time_name))
  }
  if (length(tau) != length(y)) {
    stop("Length of time variable (tau) does not match length of response variable (y).")
  }
  
  if (is.null(seed)) seed <- sample.int(.Machine$integer.max, 1L)

  if (.verbose) message("Building design matrices...")
  dm <- .build_design_matrices(b0, b1, deltas, omega, rho, data)

  if (nrow(dm$X_b0) != length(y)) {
    stop(sprintf(
      "Design matrices have %d rows but data has %d observations. This is usually caused by NA values in predictor variables. Remove or impute missing values before fitting.",
      nrow(dm$X_b0), length(y)
    ))
  }
  
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

  if (!is.null(hierarchical)) {
    .Deprecated(
      msg = paste0(
        "The `hierarchical` argument is deprecated and will be removed in a ",
        "future version. Random effects on change-point timing are now ",
        "auto-detected from formula syntax: use `omega = list(~ 1 + (1 | group))` ",
        "instead of `hierarchical = \"omega\"`."
      )
    )
  }

  has_re_om     <- .has_re(dm$X_om) || "omega" %in% hierarchical
  has_re_b1     <- .has_re(list(dm$X_b1))
  has_re_deltas <- .has_re(dm$X_deltas)
  has_re_any    <- has_re_om || has_re_b1 || has_re_deltas
  dm$has_re_om <- has_re_om

  if (.verbose) message("Running sampler...")

  .safe_int <- function(x) if (length(x) == 0) -1L else as.integer(x)
  p_deltas_safe <- .safe_int(sapply(dm$X_deltas, ncol))
  p_om_safe     <- .safe_int(sapply(dm$X_om, ncol))
  p_rho_safe    <- .safe_int(sapply(dm$X_rho, ncol))
  group_b0_safe <- .safe_int(dm$group_b0)
  b1_mask_safe  <- .safe_int(b1_mask)

  if (has_re_any) {
    re_mask_om <- .get_re_masks(dm$X_om)
    if (length(re_mask_om) == 0) re_mask_om <- list(as.integer(-1))

    reparameterise <- match.arg(reparameterise)
    nc_om_per_group <- .build_nc_om_per_group(dm, reparameterise, has_re_om)

    re_mask_b1_vec <- if (has_re_b1) {
      m <- attr(dm$X_b1, "re_mask"); if (is.null(m)) as.integer(-1) else as.integer(m)
    } else as.integer(-1)
    re_mask_deltas_list <- if (has_re_deltas) .get_re_masks(dm$X_deltas) else list(as.integer(-1))
    nc_b1     <- isTRUE(reparameterise == "omega" && has_re_b1)
    nc_deltas <- as.integer(rep(reparameterise == "omega" && has_re_deltas, length(dm$X_deltas)))
    group_re_vec <- {
      re_grp <- NULL
      if (has_re_b1) {
        re_cols <- which(attr(dm$X_b1, "re_mask") == 1L)
        if (length(re_cols)) re_grp <- max.col(dm$X_b1[, re_cols, drop = FALSE]) - 1L
      } else if (has_re_deltas) {
        for (k in seq_along(dm$X_deltas)) {
          mask <- attr(dm$X_deltas[[k]], "re_mask"); re_cols <- which(mask == 1L)
          if (length(re_cols)) { re_grp <- max.col(dm$X_deltas[[k]][, re_cols, drop = FALSE]) - 1L; break }
        }
      } else if (has_re_om) {
        re_cols <- which(attr(dm$X_om[[1]], "re_mask") == 1L)
        if (length(re_cols)) re_grp <- max.col(dm$X_om[[1]][, re_cols, drop = FALSE]) - 1L
      }
      if (is.null(re_grp)) as.integer(-1) else as.integer(re_grp)
    }
    n_subjects <- if (length(group_re_vec) > 1) max(group_re_vec) + 1L else 0L

    raw <- run_mcmc_re_ss(
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
      re_mask_om      = re_mask_om,
      nc_om_per_group = nc_om_per_group,
      re_mask_b1      = re_mask_b1_vec,
      re_mask_deltas = re_mask_deltas_list,
      nc_b1         = nc_b1,
      nc_deltas     = nc_deltas,
      group_re      = group_re_vec,
      n_subjects    = as.integer(n_subjects),
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
      hyper_priors = as.double(c(
        priors$sigma$shape, priors$sigma$scale,
        priors$sigma_u$shape, priors$sigma_u$scale,
        priors$sigma_re_om$shape, priors$sigma_re_om$scale,
        priors$sigma_re_b1$shape, priors$sigma_re_b1$scale,
        priors$sigma_re_deltas$shape, priors$sigma_re_deltas$scale
      )),
      step_om  = step_om,
      step_rho = step_rho,
      target_accept = as.double(target_accept),
      b1_spike_mask = b1_mask_safe,
      delta_spike_mask = delta_masks,
      pi_init       = as.double(spike$pi[1]),
      pi_beta_a     = if (isTRUE(spike$learn_pi)) spike$a else 0.0,
      pi_beta_b     = if (isTRUE(spike$learn_pi)) spike$b else 0.0,
      mcmc_control  = as.integer(c(
        chains, iter, warmup, seed,
        isTRUE(.verbose), max(1L, cores)
      ))
    )
  } else {
    raw <- run_mcmc_ss(
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
      step_om  = step_om,
      step_rho = step_rho,
      target_accept = as.double(target_accept),
      b1_spike_mask = b1_mask_safe,
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
  }

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
  if (has_re_any) {
    pnames <- c(pnames,
                paste0("sigma_re_omega", seq_along(dm$X_om)),
                "sigma_re_b1",
                paste0("sigma_re_delta", seq_along(dm$X_deltas)))
  }
  
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
      warmup        = as.integer(warmup),
      seed          = as.integer(seed),
      step_om       = step_om,
      step_rho      = step_rho,
      target_accept = target_accept,
      priors        = priors,
      spike         = spike,
      hierarchical  = hierarchical,
      n_divergent   = as.integer(sum(raw$n_divergent)),
      n_divergent_by_block = list(
        subj = as.integer(sum(raw$n_divergent_subj)),
        om   = as.integer(sum(raw$n_divergent_om)),
        rho  = as.integer(sum(raw$n_divergent_rho))
      )
    ),
    class = c("smoothbp_ss_fit", "smoothbp_fit")
  )
}
