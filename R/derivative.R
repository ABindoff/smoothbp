#' Posterior derivative of the conditional mean
#'
#' @description
#' Computes the \eqn{d}-th derivative of the posterior conditional mean
#' \eqn{\partial^d \mu / \partial \tau^d} at each row of \code{newdata},
#' propagating full posterior uncertainty via central finite differences
#' applied to the posterior mean function.
#'
#' The \eqn{\tau}-independent terms \eqn{b_0 + u_{0,j}} vanish from all
#' finite-difference stencils of order \eqn{\geq 1} and are not evaluated.
#' Orders 1--4 are supported. For higher orders, apply finite differences to
#' the output of \code{derivative(order = d - 1)}.
#'
#' @param object A fitted \code{smoothbp_fit} or \code{smoothbp_ss_fit}.
#' @param newdata A data frame with the time variable and any covariates
#'   required by the model formulae. Include the grouping column to condition
#'   on subject-level change-point timing; omit it for population-level
#'   derivatives (subject random effects on \eqn{\omega} set to zero).
#' @param order Positive integer; order of the derivative. Default \code{1L}.
#'   Orders 1--4 are supported via central finite differences.
#' @param probs Length-2 numeric vector of credible-interval probabilities.
#'   Default \code{c(0.025, 0.975)}.
#' @param h Step size for numerical differentiation. \code{NULL} (default)
#'   uses \eqn{10^{-4}} times the range of the training \eqn{\tau} values.
#' @param draws Logical; if \code{TRUE} return the full \eqn{S \times N}
#'   matrix of per-draw derivative estimates rather than a summary.
#'   Default \code{FALSE}.
#' @param ... Unused.
#'
#' @return If \code{draws = FALSE} (default): a \code{data.frame} with
#'   columns \code{tau}, \code{estimate} (posterior mean derivative), and
#'   credible-interval bounds named from \code{probs} (e.g. \code{Q2.5},
#'   \code{Q97.5}). Rows correspond to rows of \code{newdata}.
#'
#'   If \code{draws = TRUE}: an \eqn{S \times N} numeric matrix of
#'   per-draw derivative estimates.
#'
#' @examples
#' \dontrun{
#' nd <- data.frame(tau = seq(0, 10, by = 0.1))
#'
#' # Population-level rate of change (1st derivative)
#' dfit <- derivative(fit, newdata = nd)
#'
#' # Curvature (2nd derivative)
#' dfit2 <- derivative(fit, newdata = nd, order = 2)
#'
#' # Subject-level rate of change
#' nd_subj <- data.frame(tau = seq(0, 10, by = 0.1), subject = "s01")
#' dfit_subj <- derivative(fit, newdata = nd_subj)
#'
#' # Full posterior draws for custom summaries
#' drmat <- derivative(fit, newdata = nd, draws = TRUE)
#' apply(drmat, 2, median)
#' }
#'
#' @export
derivative <- function(object, ...) UseMethod("derivative")

#' @rdname derivative
#' @export
derivative.smoothbp_fit <- function(object, newdata, order = 1L,
                                     probs  = c(0.025, 0.975),
                                     h      = NULL,
                                     draws  = FALSE, ...) {
  .smoothbp_deriv_impl(object, newdata, order, probs, h, draws)
}

#' @rdname derivative
#' @export
derivative.smoothbp_ss_fit <- function(object, newdata, order = 1L,
                                        probs  = c(0.025, 0.975),
                                        h      = NULL,
                                        draws  = FALSE, ...) {
  .smoothbp_deriv_impl(object, newdata, order, probs, h, draws)
}

# ---------------------------------------------------------------------------
# Shared implementation
# ---------------------------------------------------------------------------

.smoothbp_deriv_impl <- function(object, newdata, order, probs, h, draws) {
  order <- as.integer(order)
  if (order < 1L)
    stop("'order' must be >= 1")
  if (order > 4L)
    stop(sprintf(
      "'order' = %d is not supported; compute the order-%d derivative first, ",
      order, order - 1L),
      "then apply finite differences")

  tau <- as.double(newdata[[object$time]])
  n   <- length(tau)
  dm  <- .build_newdata_dm(object, newdata)

  if (is.null(h)) {
    tau_train <- as.double(object$data[[object$time]])
    h <- max(1e-6, diff(range(tau_train)) * 1e-4)
  }

  stencil   <- .fd_stencil(order)
  draw_mat  <- posterior::as_draws_matrix(object$draws)
  col_names <- colnames(draw_mat)
  n_draws   <- nrow(draw_mat)
  n_bp      <- length(dm$X_deltas)

  # ---- column indices -------------------------------------------------------
  b1_idx  <- which(grepl("^b1_",        col_names))
  g_b1    <- which(grepl("^gamma_b1_",  col_names))
  delta_idx <- lapply(seq_len(n_bp), function(k)
    which(grepl(paste0("^delta", k, "_"),       col_names)))
  om_idx  <- lapply(seq_len(n_bp), function(k)
    which(grepl(paste0("^omega", k, "_"),       col_names)))
  rho_idx <- lapply(seq_len(n_bp), function(k)
    which(grepl(paste0("^rho",   k, "_"),       col_names)))
  g_d_idx <- lapply(seq_len(n_bp), function(k)
    which(grepl(paste0("^gamma_delta", k, "_"), col_names)))

  # ---- precompute tau-independent linear predictors (n_draws x n) ----------
  # b1 (with optional spike-and-slab gamma)
  b1_raw <- draw_mat[, b1_idx, drop = FALSE]
  if (length(g_b1) > 0)
    b1_raw <- b1_raw * draw_mat[, g_b1, drop = FALSE]
  b1_lp <- tcrossprod(b1_raw, dm$X_b1)   # n_draws x n

  delta_lp <- vector("list", n_bp)
  om_lp    <- vector("list", n_bp)
  rho_lp   <- vector("list", n_bp)
  for (k in seq_len(n_bp)) {
    d_raw <- draw_mat[, delta_idx[[k]], drop = FALSE]
    if (length(g_d_idx[[k]]) > 0)
      d_raw <- d_raw * draw_mat[, g_d_idx[[k]], drop = FALSE]
    delta_lp[[k]] <- tcrossprod(d_raw, dm$X_deltas[[k]])
    om_lp[[k]]    <- tcrossprod(draw_mat[, om_idx[[k]],  drop = FALSE], dm$X_om[[k]])
    rho_lp[[k]]   <- tcrossprod(draw_mat[, rho_idx[[k]], drop = FALSE], dm$X_rho[[k]])
  }

  # ---- helper: evaluate mean at a given tau vector (b0/u omitted — cancel) -
  .eval_at_tau <- function(tau_j) {
    TAU <- matrix(tau_j, n_draws, n, byrow = TRUE)   # n_draws x n
    if (n_bp > 0L) {
      mu <- b1_lp * (TAU - om_lp[[1L]])
    } else {
      mu <- b1_lp * TAU
    }
    for (k in seq_len(n_bp)) {
      di <- TAU - om_lp[[k]]
      si <- 1 / (1 + exp(-di * rho_lp[[k]]))
      mu <- mu + delta_lp[[k]] * di * si
    }
    mu
  }

  # ---- accumulate finite-difference stencil --------------------------------
  deriv_mat <- matrix(0, n_draws, n)
  for (j in seq_along(stencil$pts)) {
    deriv_mat <- deriv_mat +
      stencil$coef[j] * .eval_at_tau(tau + stencil$pts[j] * h)
  }
  deriv_mat <- deriv_mat / h ^ order

  if (draws) return(deriv_mat)

  p_lo <- probs[1L]; p_hi <- probs[2L]
  out <- data.frame(
    tau      = tau,
    estimate = colMeans(deriv_mat),
    lo       = apply(deriv_mat, 2L, stats::quantile, probs = p_lo),
    hi       = apply(deriv_mat, 2L, stats::quantile, probs = p_hi),
    row.names = NULL
  )
  names(out)[3:4] <- paste0("Q", c(p_lo, p_hi) * 100)
  out
}

# ---------------------------------------------------------------------------
# Central finite-difference stencils (order 1--4)
# Coefficients satisfy: f^(d)(x) = sum_j coef[j]*f(x + pts[j]*h) / h^d + O(h^2)
# ---------------------------------------------------------------------------
.fd_stencil <- function(order) {
  switch(as.character(order),
    "1" = list(pts  = c(-1,  1),
               coef = c(-0.5,  0.5)),
    "2" = list(pts  = c(-1,  0,  1),
               coef = c( 1,  -2,   1  )),
    "3" = list(pts  = c(-2, -1,  1,  2),
               coef = c(-0.5,  1,  -1,   0.5)),
    "4" = list(pts  = c(-2, -1,  0,  1,  2),
               coef = c( 1,  -4,   6,  -4,   1))
  )
}
