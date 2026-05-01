#' Fit a hierarchical piecewise regression model with a smoothed change-point
#'
#' Fits the model:
#' \deqn{y_i = b0_i + b1_i(\tau_i - \omega_i) + b2_i(\tau_i - \omega_i)\,\sigma\bigl((\tau_i - \omega_i)\rho_i\bigr) + \varepsilon_i}
#'
#' where \eqn{\sigma(\cdot)} is the logistic sigmoid, \eqn{\omega} is the
#' change-point location, and \eqn{\rho} controls the sharpness of the
#' transition.  Each parameter (\eqn{b0}, \eqn{b1}, \eqn{b2}, \eqn{\omega},
#' \eqn{\rho}) has its own linear predictor specified via a one-sided R formula.
#' A random intercept is supported for \eqn{b0} using standard \code{(1 | group)}
#' syntax.
#'
#' Posterior inference is performed via a Metropolis-within-Gibbs sampler
#' compiled in Rust:
#' \itemize{
#'   \item \strong{Gibbs} (exact conjugate) for \eqn{b0}, \eqn{b1}, \eqn{b2}
#'         coefficients and the random intercepts.
#'   \item \strong{Metropolis} (random-walk with adaptive proposal) for
#'         \eqn{\omega} and \eqn{\rho} coefficients. For multi-coefficient
#'         linear predictors the sampler runs a componentwise random walk
#'         with per-coordinate step tuning during the first 30\% of warmup,
#'         then switches to a joint Haario-style adaptive Metropolis with
#'         covariance learned online and global scale tuned to ~23.4\%
#'         acceptance, then freezes the proposal at end of warmup. For 1-D
#'         linear predictors (e.g. \code{omega = ~ 1}) the sampler is the
#'         classical scalar adaptive random walk targeting ~23.4\%.
#'   \item \strong{Gibbs} (inverse-gamma conjugate) for \eqn{\sigma} and
#'         \eqn{\sigma_u}.
#' }
#'
#' To keep the per-iteration cost low, the MH steps for \eqn{\omega} and
#' \eqn{\rho} reuse a precomputed linear-predictor cache so each proposal
#' costs O(n) rather than recomputing the entire fitted-mean function.
#'
#' @param formula A two-sided formula identifying the response and time variable,
#'   e.g. \code{value ~ tau}.
#' @param b0 One-sided formula for the \eqn{b0} linear predictor.
#'   May include a single random-intercept term, e.g.
#'   \code{~ 1 + age + sex + (1 | subject_id)}.
#' @param b1 One-sided formula for \eqn{b1}. Default \code{~ 1}.
#' @param b2 One-sided formula for \eqn{b2}. Default \code{~ 1}.
#' @param omega One-sided formula for the change-point \eqn{\omega}. Default \code{~ 1}.
#' @param rho One-sided formula for the sharpness \eqn{\rho}. Default \code{~ 1}.
#' @param data A data frame containing all variables referenced in formulas.
#' @param priors A \code{\link{smoothbp_priors}} object. Defaults to
#'   \code{smoothbp_priors()}.
#' @param chains Number of independent MCMC chains. Default 4.
#' @param iter Total iterations per chain (warmup + sampling). Default 2000.
#' @param warmup Number of warmup iterations (discarded). Default 1000.
#' @param seed Integer random seed for reproducibility.
#' @param step_om Initial Metropolis step size for \eqn{\omega} coefficients.
#'   Used as the initial per-coordinate proposal SD during the componentwise
#'   warmup phase and to seed the joint adaptive Metropolis scale. Tuned
#'   automatically during warmup; see Details. Default 0.3.
#' @param step_rho Initial Metropolis step size for \eqn{\rho} coefficients.
#'   Same semantics as \code{step_om}. Default 0.3.
#' @param cores Number of CPU cores used to run chains in parallel.  When
#'   \code{cores > 1} all chains run concurrently via Rayon (the per-iteration
#'   progress bar is suppressed in this mode — only a start and done message
#'   are printed).  Defaults to \code{getOption("smoothbp.cores", 1L)}.  Set
#'   \code{options(smoothbp.cores = parallel::detectCores())} in your
#'   \code{.Rprofile} to make parallel the default.
#' @param .verbose Logical; print progress messages. Default \code{TRUE}.
#'
#' @return A \code{smoothbp_fit} object.
#'
#' @examples
#' \dontrun{
#' fit <- smoothbp(
#'   formula = value ~ tau,
#'   b0    = ~ 1 + age + treatment + (1 | subject),
#'   b1    = ~ 1 + treatment,
#'   b2    = ~ 1 + treatment,
#'   omega = ~ 1 + treatment,
#'   rho   = ~ 1,
#'   data  = mydata,
#'   priors = smoothbp_priors(
#'     omega = list(
#'       "(Intercept)"  = prior_normal(3, 2, lb = 0, ub = 6),
#'       "treatmentExp" = prior_normal(0, 2)
#'     )
#'   ),
#'   chains = 4, iter = 2000, warmup = 1000, seed = 42
#' )
#' print(fit)
#' }
#' @export
smoothbp <- function(
    formula,
    b0     = ~ 1,
    b1     = ~ 1,
    b2     = ~ 1,
    omega  = ~ 1,
    rho    = ~ 1,
    data,
    priors = smoothbp_priors(),
    chains = 4L,
    iter   = 2000L,
    warmup = 1000L,
    seed   = NULL,
    step_om  = 0.3,
    step_rho = 0.3,
    cores    = getOption("smoothbp.cores", 1L),
    .verbose = TRUE
) {
  # ---- Input validation ---------------------------------------------------
  if (!inherits(formula, "formula") || length(formula) != 3L) {
    stop("`formula` must be a two-sided formula, e.g. value ~ tau.")
  }
  stopifnot(warmup < iter, chains >= 1L)

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
  pv <- .build_prior_vectors(priors, dm)

  # ---- Run sampler --------------------------------------------------------
  raw <- run_mcmc(
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
    chains   = as.integer(chains),
    iter     = as.integer(iter),
    warmup   = as.integer(warmup),
    seed     = as.integer(seed),
    verbose  = isTRUE(.verbose),
    n_cores  = as.integer(max(1L, cores))
  )

  # ---- Label parameters and wrap in posterior draws_array -----------------
  pnames <- .param_names(dm, pv)

  # raw$draws is a list of matrices (one per chain); convert to draws_array
  n_post   <- nrow(raw$draws[[1]])
  n_params <- ncol(raw$draws[[1]])
  chain_arr <- array(
    data     = unlist(lapply(raw$draws, function(m) t(m))),  # params × draws per chain
    dim      = c(n_params, n