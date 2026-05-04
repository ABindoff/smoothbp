#' Fit a smooth change-point model with spike-and-slab variable selection on b2
#'
#' Like [smoothbp()], but adds spike-and-slab priors on `b2` coefficients to
#' perform structured variable selection.
#'
#' When a `b2` coefficient is "spiked" to zero (\eqn{\gamma_k = 0}), the
#' corresponding omega and rho coefficients (matched by column name) are also
#' set to zero, reflecting the idea that if a covariate does not affect the
#' slope change, the change-point location/sharpness should not vary with that
#' covariate either.
#'
#' The sampler uses the Kuo & Mallick (1998) approach: all regression
#' coefficients are always sampled (even when "excluded"), but only the
#' included ones contribute to the mean function.  Inclusion indicators
#' \eqn{\gamma_k} are sampled from their Bernoulli full conditional each
#' iteration.
#'
#' @inheritParams smoothbp
#' @param spike A [prior_spike_slab()] object controlling which `b2`
#'   coefficients receive the spike-and-slab prior, the inclusion probability
#'   `pi`, and the slab distribution. Default: `prior_spike_slab()` (pi = 0.5,
#'   slab = Normal(0, 2), spike on non-intercept coefficients only).
#'
#' @return A `smoothbp_ss_fit` object (extends `smoothbp_fit`) with additional
#'   components:
#'   \describe{
#'     \item{`gamma_names`}{Character vector of names for the gamma indicators.}
#'     \item{`spike`}{The `prior_spike_slab` object used.}
#'   }
#'   The draws array includes gamma indicator columns (`gamma_<coef_name>`)
#'   whose posterior means are the posterior inclusion probabilities (PIPs).
#'
#' @examples
#' \dontrun{
#' dat <- simulate_smoothbp(
#'   n_subj = 1, n_obs = 60,
#'   b0 = 5, b1 = -0.4, b2 = 1.2,
#'   omega = 3, rho = 4, sigma = 0.4, sigma_u = 0
#' )
#' # Add a covariate with no true effect on b2
#' dat$x <- rnorm(nrow(dat))
#'
#' fit <- smoothbp_ss(
#'   formula = y ~ tau,
#'   b2    = ~ 1 + x,
#'   omega = ~ 1 + x,
#'   data  = dat,
#'   spike = prior_spike_slab(pi = 0.5)
#' )
#' pip(fit)
#' }
#'
#' @references
#' Kuo, L. and Mallick, B. (1998). Variable selection for regression models.
#' *Sankhya B*, 60(1):65--81.
#'
#' @export
smoothbp_ss <- function(
    formula,
    b0     = ~ 1,
    b1     = ~ 1,
    b2     = ~ 1,
    omega  = ~ 1,
    rho    = ~ 1,
    data,
    priors = smoothbp_priors(),
    spike  = prior_spike_slab(),
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
  # ---- Input validation ---------------------------------------------------
  if (!inherits(formula, "formula") || length(formula) != 3L) {
    stop("`formula` must be a two-sided formula, e.g. value ~ tau.")
  }
  stopifnot(warmup < iter, chains >= 1L)
  stopifnot(inherits(spike, "smoothbp_spike_slab"))

  response_name <- deparse(formula[[2]])
  time_name     <- deparse(formula[[3]])

  if (!response_name %in% names(data)) {
    stop(sprintf("Response variable '%s' not found in data.", response_name))
  }
  if (!time_name %in% names(data)) {
    stop(sprintf("Time variable '%s' not found in data.", time_name))
  }

  y   <- as.double(data[[response_name]])
  tau <- as.double(data[[time_name]])

  if (any(is.na(y)) || any(is.na(tau))) {
    stop("Missing values in response or time variable are not supported.")
  }

  if (is.null(seed)) seed <- sample.int(.Machine$integer.max, 1L)

  # ---- Build design matrices ----------------------------------------------
  if (.verbose) message("Building design matrices...")
  dm <- .build_design_matrices(b0, b1, b2, omega, rho, data)

  # ---- Build prior vectors ------------------------------------------------
  # Override b2 slab prior from spike object
  priors_effective <- priors
  priors_effective$b2 <- spike$slab
  pv <- .build_prior_vectors(priors_effective, dm)

  # ---- Build spike-and-slab configuration ---------------------------------
  b2_names  <- dm$col_names_b2
  om_names  <- dm$col_names_om
  rho_names <- dm$col_names_rho
  p_b2      <- length(b2_names)

  # spike_mask: which b2 coefficients get spike-and-slab
  spike_mask <- rep(1L, p_b2)
  if (!spike$spike_intercept) {
    # Don't spike the intercept
    spike_mask[b2_names == "(Intercept)"] <- 0L
  }

  # pi: per-coefficient inclusion probability
  pi_vec <- rep(0.5, p_b2)
  if (length(spike$pi) == 1L) {
    pi_vec[] <- spike$pi
  } else if (is.null(names(spike$pi))) {
    if (length(spike$pi) != p_b2) {
      stop(sprintf(
        "spike$pi has length %d but b2 has %d coefficients.",
        length(spike$pi), p_b2
      ))
    }
    pi_vec <- spike$pi
  } else {
    # Named vector: match by name
    for (nm in names(spike$pi)) {
      idx <- which(b2_names == nm)
      if (length(idx) > 0) pi_vec[idx] <- spike$pi[nm]
    }
  }

  # om_map: for each b2 coefficient, the index (0-based) of the same-named
  # omega coefficient, or -1 if no match
  om_map <- match(b2_names, om_names) - 1L
  om_map[is.na(om_map)] <- -1L

  # rho_map: same for rho
  rho_map <- match(b2_names, rho_names) - 1L
  rho_map[is.na(rho_map)] <- -1L

  if (.verbose) {
    n_spike <- sum(spike_mask == 1L)
    if (isTRUE(spike$learn_pi)) {
      message(sprintf(
        "Spike-and-slab on %d of %d b2 coefficients (pi ~ Beta(%g, %g))",
        n_spike, p_b2, spike$a, spike$b
      ))
    } else {
      message(sprintf(
        "Spike-and-slab on %d of %d b2 coefficients (pi = %s)",
        n_spike, p_b2,
        paste(format(pi_vec[spike_mask == 1L], digits = 2), collapse = ", ")
      ))
    }
  }

  # ---- Run sampler --------------------------------------------------------
  raw <- run_mcmc_ss(
    y             = y,
    tau           = tau,
    x_b0          = as.double(dm$X_b0),  p_b0  = ncol(dm$X_b0),
    x_b1          = as.double(dm$X_b1),  p_b1  = ncol(dm$X_b1),
    x_b2          = as.double(dm$X_b2),  p_b2  = ncol(dm$X_b2),
    x_om          = as.double(dm$X_om),  p_om  = ncol(dm$X_om),
    x_rho         = as.double(dm$X_rho), p_rho = ncol(dm$X_rho),
    group_b0      = dm$group_b0,
    n_groups_b0   = dm$n_groups_b0,
    prior_mean    = pv$mean,
    prior_sd      = pv$sd,
    prior_lb      = pv$lb,
    prior_ub      = pv$ub,
    sigma_shape   = priors$sigma$shape,
    sigma_scale   = priors$sigma$scale,
    sigma_u_shape = priors$sigma_u$shape,
    sigma_u_scale = priors$sigma_u$scale,
    step_om  = step_om,
    step_rho = step_rho,
    target_accept = as.double(target_accept),
    spike_mask    = as.integer(spike_mask),
    spike_pi      = as.double(pi_vec),
    om_map        = as.integer(om_map),
    rho_map       = as.integer(rho_map),
    pi_beta_a     = if (isTRUE(spike$learn_pi)) spike$a else 0.0,
    pi_beta_b     = if (isTRUE(spike$learn_pi)) spike$b else 0.0,
    chains   = as.integer(chains),
    iter     = as.integer(iter),
    warmup   = as.integer(warmup),
    seed     = as.integer(seed),
    verbose  = isTRUE(.verbose),
    n_cores  = as.integer(max(1L, cores))
  )

  # ---- Label parameters and wrap in posterior draws_array -----------------
  base_pnames <- .param_names(dm, pv)
  gamma_names <- paste0("gamma_", b2_names)
  pnames <- c(base_pnames, gamma_names)
  if (isTRUE(spike$learn_pi)) {
    pnames <- c(pnames, "pi")
  }

  n_post   <- nrow(raw$draws[[1]])
  n_params <- ncol(raw$draws[[1]])
  chain_arr <- array(
    data     = unlist(lapply(raw$draws, function(m) t(m))),
    dim      = c(n_params, n_post, chains),
    dimnames = list(variable = pnames, draw = NULL, chain = NULL)
  )
  da <- posterior::as_draws_array(aperm(chain_arr, c(2, 3, 1)))

  # ---- Check for divergent transitions -----------------------------------
  n_divergent <- raw$n_divergent %||% 0L
  if (n_divergent > 0L) {
    warning(
      sprintf(
        "%d divergent transition(s) after warmup. Posterior may be unreliable. ",
        n_divergent
      ),
      "Try increasing target_accept or tightening priors.",
      call. = FALSE
    )
  }

  # ---- Assemble fit object ------------------------------------------------
  structure(
    list(
      draws         = da,
      n_divergent   = n_divergent,
      param_names   = pnames,
      gamma_names   = gamma_names,
      formula       = formula,
      b0_formula    = b0,
      b1_formula    = b1,
      b2_formula    = b2,
      omega_formula = omega,
      rho_formula   = rho,
      priors        = priors,
      spike         = spike,
      data          = data,
      response      = response_name,
      time          = time_name,
      dm            = dm,
      chains        = as.integer(chains),
      iter          = as.integer(iter),
      warmup        = as.integer(warmup),
      seed          = seed,
      cores         = as.integer(max(1L, cores))
    ),
    class = c("smoothbp_ss_fit", "smoothbp_fit")
  )
}
