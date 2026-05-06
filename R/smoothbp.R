#' Fit a hierarchical piecewise regression model with smoothed change-points
#'
#' @param formula A two-sided formula identifying the response and time variable,
#'   e.g. \code{value ~ tau}.
#' @param b0 One-sided formula for the \eqn{b0} linear predictor.
#' @param b1 One-sided formula for \eqn{b1}. Default \code{~ 1}.
#' @param deltas List of one-sided formulas for slope changes. Default \code{list(~ 1)}.
#' @param omega List of one-sided formulas for change-point locations. Default \code{list(~ 1)}.
#' @param rho List of one-sided formulas for transition sharpness. Default \code{list(~ 1)}.
#' @param data A data frame.
#' @param priors A \code{\link{smoothbp_priors}} object.
#' @param chains Number of chains. Default 4.
#' @param iter Total iterations per chain. Default 2000.
#' @param warmup Warmup iterations. Default 1000.
#' @param seed Random seed.
#' @param step_om Initial HMC/MH step size for omega.
#' @param step_rho Initial HMC/MH step size for rho.
#' @param target_accept Target HMC acceptance probability.
#' @param cores Number of CPU cores.
#' @param .verbose Print progress.
#'
#' @return A \code{smoothbp_fit} object.
#' @export
smoothbp <- function(
    formula,
    b0     = ~ 1,
    b1     = ~ 1,
    deltas = list(~ 1),
    omega  = list(~ 1),
    rho    = list(~ 1),
    data,
    priors = smoothbp_priors(),
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
    stop("`formula` must be a two-sided formula, e.g. value ~ tau.")
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
  pv <- .build_prior_vectors(priors, dm)

  if (.verbose) message("Running sampler...")
  raw <- run_mcmc(
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
    chains   = as.integer(chains),
    iter     = as.integer(iter),
    warmup   = as.integer(warmup),
    seed     = as.integer(seed),
    verbose  = isTRUE(.verbose),
    n_cores  = as.integer(max(1L, cores))
  )

  pnames <- .param_names(dm, pv)

  # Assemble fit object
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
      chains        = as.integer(chains),
      iter          = as.integer(iter),
      warmup        = as.integer(warmup)
    ),
    class = "smoothbp_fit"
  )
}

#' Update a fitted smoothbp model
#'
#' Re-fits the model, replacing any arguments supplied here with the
#' corresponding values stored in the original fit for anything left
#' unspecified.
#'
#' @param object A \code{smoothbp_fit} object.
#' @param formula,b0,b1,b2,omega,rho,data,priors,chains,iter,warmup,seed,step_om,step_rho,target_accept,cores,.verbose
#'   Replacements for the corresponding arguments of \code{\link{smoothbp}}.
#'   Any argument not supplied is taken from \code{object}.
#' @param ... Ignored.
#'
#' @return A new \code{smoothbp_fit} object.
#' @export
update.smoothbp_fit <- function(
    object,
    formula,
    b0, b1, b2, omega, rho,
    data,
    priors,
    chains, iter, warmup, seed,
    step_om, step_rho, target_accept,
    cores,
    .verbose = TRUE,
    ...
) {
  if (missing(formula))      formula      <- object$formula
  if (missing(b0))           b0           <- object$b0_formula
  if (missing(b1))           b1           <- object$b1_formula
  if (missing(b2))           b2           <- object$b2_formula
  if (missing(omega))        omega        <- object$omega_formula
  if (missing(rho))          rho          <- object$rho_formula
  if (missing(data))         data         <- object$data
  if (missing(priors))       priors       <- object$priors
  if (missing(chains))       chains       <- object$chains
  if (missing(iter))         iter         <- object$iter
  if (missing(warmup))       warmup       <- object$warmup
  if (missing(seed))         seed         <- object$seed
  if (missing(cores))        cores        <- object$cores
  # step_om, step_rho, target_accept are not stored; fall back to smoothbp() defaults
  if (missing(step_om))        step_om        <- 0.3
  if (missing(step_rho))       step_rho       <- 0.3
  if (missing(target_accept))  target_accept  <- 0.65

  smoothbp(
    formula       = formula,
    b0            = b0,
    b1            = b1,
    b2            = b2,
    omega         = omega,
    rho           = rho,
    data          = data,
    priors        = priors,
    chains        = chains,
    iter          = iter,
    warmup        = warmup,
    seed          = seed,
    step_om       = step_om,
    step_rho      = step_rho,
    target_accept = target_accept,
    cores         = cores,
    .verbose      = .verbose
  )
}
