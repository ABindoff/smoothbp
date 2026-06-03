#' Simulate data from the smooth change-point model
#'
#' Generates synthetic data from the model used by \code{\link{smoothbp}},
#' including optional between-subject random intercepts.  Supports any number
#' of breakpoints \eqn{K \geq 1}.  True parameter values are stored as the
#' \code{"true_params"} attribute so they can be compared against posterior
#' estimates.
#'
#' The data-generating model for \eqn{K} breakpoints is:
#' \deqn{
#'   y_{ij} = (b_0 + u_j) + b_1 (\tau_{ij} - \omega_1)
#'            + \sum_{k=1}^{K} \delta_k \, d_{ijk} \, \text{logistic}(d_{ijk} \, \rho_k)
#'            + \varepsilon_{ij}
#' }
#' where \eqn{d_{ijk} = \tau_{ij} - \omega_k} and \eqn{\text{logistic}(\cdot)} is the
#' logistic function \eqn{\text{logistic}(x) = (1 + e^{-x})^{-1}}.  The pre-break slope \eqn{b_1} is centred at the first
#' change-point \eqn{\omega_1}, so \eqn{b_0} represents the conditional mean
#' at \eqn{\tau = \omega_1} (consistent with the fitted model).
#'
#' For a single breakpoint (\eqn{K = 1}), scalar values of \code{omega},
#' \code{rho}, and \code{delta} are accepted for backward compatibility.
#'
#' @param n_subj    Number of subjects (groups).  Set to \code{1} and
#'   \code{sigma_u = 0} for a single-group simulation.
#' @param n_obs     Observations per subject.  May be a scalar (same for all
#'   subjects) or a length-\code{n_subj} integer vector for unbalanced designs.
#' @param b0        Overall intercept (conditional mean at \eqn{\tau = \omega_1}).
#' @param b1        Pre-change-point slope (evaluated relative to \eqn{\omega_1}).
#' @param delta     Change in slope at each change-point.  A numeric vector of
#'   length \eqn{K}; a scalar is treated as \eqn{K = 1}.
#' @param omega     Change-point location(s).  A numeric vector of length
#'   \eqn{K}, in ascending order.  A scalar is treated as \eqn{K = 1}.
#' @param rho       Sharpness of each transition.  A numeric vector of length
#'   \eqn{K} (all values must be positive); a scalar is recycled to length
#'   \eqn{K}.
#' @param sigma     Residual standard deviation.
#' @param sigma_u   Between-subject SD for random intercepts.  Set to \code{0}
#'   to suppress random effects.  Default \code{0.5}.
#' @param tau_range Numeric vector of length 2 giving the range of the time
#'   variable.  Observations are evenly spaced within this range for each
#'   subject.  Default \code{c(0, 6)}.
#' @param seed      Integer seed for reproducibility.  Sampled randomly if
#'   \code{NULL} (default).
#'
#' @return A \code{data.frame} with columns:
#'   \describe{
#'     \item{\code{subject}}{Subject identifier (factor).}
#'     \item{\code{tau}}{Time variable.}
#'     \item{\code{mu}}{Noise-free conditional mean \eqn{\mu_{ij}}.}
#'     \item{\code{y}}{Observed response.}
#'   }
#'   The attribute \code{"true_params"} is a named list containing the
#'   data-generating values of \code{b0}, \code{b1}, \code{delta}, \code{omega},
#'   \code{rho}, \code{sigma}, \code{sigma_u}, the vector of subject-level
#'   deviations \code{u}, and the \code{seed} used.
#'
#' @examples
#' # Single breakpoint (K = 1)
#' dat1 <- simulate_smoothbp(
#'   n_subj = 20, n_obs = 8,
#'   b0 = 5, b1 = -0.3, delta = 1.2,
#'   omega = 3, rho = 4, sigma = 0.4, sigma_u = 0.5,
#'   seed = 42
#' )
#' head(dat1)
#' attr(dat1, "true_params")
#'
#' # Two breakpoints (K = 2)
#' dat2 <- simulate_smoothbp(
#'   n_subj = 20, n_obs = 12,
#'   b0 = 5, b1 = -0.3,
#'   delta = c(1.2, -0.8),
#'   omega = c(2, 4),
#'   rho   = c(4, 4),
#'   sigma = 0.4, sigma_u = 0.5,
#'   seed = 42
#' )
#' head(dat2)
#'
#' @export
simulate_smoothbp <- function(
    n_subj    = 20L,
    n_obs     = 8L,
    b0        = 5.0,
    b1        = -0.3,
    delta     = 1.2,
    omega     = 3.0,
    rho       = 4.0,
    sigma     = 0.4,
    sigma_u   = 0.5,
    tau_range = c(0, 6),
    seed      = NULL
) {
  # ---- Coerce to vectors and infer K ----------------------------------------
  omega <- as.numeric(omega)
  delta <- as.numeric(delta)
  rho   <- as.numeric(rho)

  K <- length(omega)

  if (length(delta) == 1L && K > 1L) {
    stop("`delta` must be a vector of length K = ", K, " (one value per breakpoint).")
  }
  if (length(delta) != K) {
    stop("`delta` must have the same length as `omega` (K = ", K, ").")
  }
  # Recycle scalar rho to length K
  if (length(rho) == 1L && K > 1L) {
    rho <- rep(rho, K)
  }
  if (length(rho) != K) {
    stop("`rho` must be a scalar (recycled) or a vector of length K = ", K, ".")
  }

  # ---- Validation -----------------------------------------------------------
  if (is.null(seed)) seed <- sample.int(.Machine$integer.max, 1L)
  set.seed(seed)

  stopifnot(
    all(omega > 0),
    all(rho   > 0),
    is.numeric(sigma),   sigma   > 0,
    is.numeric(sigma_u), sigma_u >= 0,
    length(tau_range) == 2, tau_range[1] < tau_range[2],
    n_subj >= 1L
  )
  if (K > 1L && !all(diff(omega) > 0)) {
    stop("`omega` values must be in strictly ascending order.")
  }

  # Allow unbalanced designs
  if (length(n_obs) == 1L) n_obs <- rep(as.integer(n_obs), n_subj)
  if (length(n_obs) != n_subj) {
    stop("`n_obs` must be a scalar or a vector of length `n_subj`.")
  }

  # ---- Generate random intercepts ------------------------------------------
  u_j <- if (sigma_u > 0) rnorm(n_subj, 0, sigma_u) else rep(0.0, n_subj)

  # ---- Logistic sigmoid ----------------------------------------------------
  .sigmoid <- function(x) 1.0 / (1.0 + exp(-x))

  # ---- Build rows for each subject -----------------------------------------
  rows <- vector("list", n_subj)
  for (j in seq_len(n_subj)) {
    tau_j <- seq(tau_range[1], tau_range[2], length.out = n_obs[j])

    # b1 term centred at first breakpoint (consistent with Rust model)
    mu_j <- (b0 + u_j[j]) + b1 * (tau_j - omega[1])

    # Accumulate each breakpoint's contribution
    for (k in seq_len(K)) {
      d_jk  <- tau_j - omega[k]
      s_jk  <- .sigmoid(d_jk * rho[k])
      mu_j  <- mu_j + delta[k] * d_jk * s_jk
    }

    y_j <- mu_j + rnorm(n_obs[j], 0, sigma)

    rows[[j]] <- data.frame(
      subject = j,
      tau     = tau_j,
      mu      = mu_j,
      y       = y_j
    )
  }

  dat         <- do.call(rbind, rows)
  dat$subject <- factor(dat$subject)
  rownames(dat) <- NULL

  attr(dat, "true_params") <- list(
    b0      = b0,
    b1      = b1,
    delta   = delta,
    omega   = omega,
    rho     = rho,
    sigma   = sigma,
    sigma_u = sigma_u,
    u       = u_j,
    seed    = seed
  )

  dat
}

#' Print true parameters from a simulated dataset
#'
#' Convenience function to display the data-generating parameters stored in the
#' \code{"true_params"} attribute of a dataset returned by
#' \code{\link{simulate_smoothbp}}.
#'
#' @param dat A \code{data.frame} returned by \code{simulate_smoothbp}.
#' @return The \code{true_params} list, invisibly.
#' @export
true_params <- function(dat) {
  tp <- attr(dat, "true_params")
  if (is.null(tp)) stop("`dat` does not have a `true_params` attribute.")

  scalar_params <- tp[!names(tp) %in% c("u", "delta", "omega", "rho")]
  cat("Data-generating parameters:\n")
  for (nm in names(scalar_params)) {
    cat(sprintf("  %-10s %s\n", paste0(nm, ":"), scalar_params[[nm]]))
  }
  # Print vector params with index labels
  for (nm in c("omega", "delta", "rho")) {
    vals <- tp[[nm]]
    for (k in seq_along(vals)) {
      cat(sprintf("  %-10s %s\n", paste0(nm, "[", k, "]:"), vals[k]))
    }
  }
  invisible(tp)
}
