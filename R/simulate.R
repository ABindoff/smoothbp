#' Simulate data from the smooth change-point model
#'
#' Generates synthetic data from the model used by \code{\link{smoothbp}},
#' including optional between-subject random intercepts.  True parameter values
#' are stored as the \code{"true_params"} attribute so they can be compared
#' against posterior estimates.
#'
#' The data-generating model is:
#' \deqn{y_{ij} = (b0 + u_j) + b1 \cdot d_{ij} + b2 \cdot d_{ij} \cdot \sigma(d_{ij} \cdot \rho) + \varepsilon_{ij}}
#' where \eqn{d_{ij} = \tau_{ij} - \omega} and \eqn{\sigma(\cdot)} is the
#' logistic sigmoid.
#'
#' @param n_subj    Number of subjects (groups).  Set to \code{1} and
#'   \code{sigma_u = 0} for a single-group simulation.
#' @param n_obs     Observations per subject.  May be a scalar (same for all
#'   subjects) or a length-\code{n_subj} integer vector for unbalanced designs.
#' @param b0        Overall intercept.
#' @param b1        Pre-change-point slope.
#' @param b2        Change in slope at the change-point.
#' @param omega     Change-point location (must be positive).
#' @param rho       Sharpness of the transition (must be positive; larger values
#'   give a sharper kink).
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
#'   data-generating values of \code{b0}, \code{b1}, \code{b2}, \code{omega},
#'   \code{rho}, \code{sigma}, \code{sigma_u}, the vector of subject-level
#'   deviations \code{u}, and the \code{seed} used.
#'
#' @examples
#' dat <- simulate_smoothbp(
#'   n_subj = 20, n_obs = 8,
#'   b0 = 5, b1 = -0.3, b2 = 1.2,
#'   omega = 3, rho = 4, sigma = 0.4, sigma_u = 0.5,
#'   seed = 42
#' )
#' head(dat)
#' attr(dat, "true_params")
#'
#' @export
simulate_smoothbp <- function(
    n_subj    = 20L,
    n_obs     = 8L,
    b0        = 5.0,
    b1        = -0.3,
    b2        = 1.2,
    omega     = 3.0,
    rho       = 4.0,
    sigma     = 0.4,
    sigma_u   = 0.5,
    tau_range = c(0, 6),
    seed      = NULL
) {
  # ---- Validation -----------------------------------------------------------
  if (is.null(seed)) seed <- sample.int(.Machine$integer.max, 1L)
  set.seed(seed)

  stopifnot(
    is.numeric(omega), omega > 0,
    is.numeric(rho),   rho   > 0,
    is.numeric(sigma), sigma > 0,
    is.numeric(sigma_u), sigma_u >= 0,
    length(tau_range) == 2, tau_range[1] < tau_range[2],
    n_subj >= 1L
  )

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
    d_j   <- tau_j - omega
    s_j   <- .sigmoid(d_j * rho)
    mu_j  <- (b0 + u_j[j]) + b1 * d_j + b2 * d_j * s_j
    y_j   <- mu_j + rnorm(n_obs[j], 0, sigma)

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
    b2      = b2,
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

  scalar_params <- tp[names(tp) != "u"]
  cat("Data-generating parameters:\n")
  for (nm in names(scalar_params)) {
    cat(sprintf("  %-10s %s\n", paste0(nm, ":"), scalar_params[[nm]]))
  }
  invisible(tp)
}
