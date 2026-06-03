# Internal utilities and global variable declarations
# ---------------------------------------------------------------------------

# Suppress R CMD check NOTEs for ggplot2 column names used via aes() and
# posterior draws data frames that are created at runtime.
utils::globalVariables(c(
  # plot_methods.R (.build_trace, .build_density, plot.smoothbp_pip)
  "iteration", "value", "chain", "xmin", "xmax", "ymin", "ymax",
  "pip", "lower", "upper", "parameter", "type",
  # postprocess.R (pp_check)
  "y", ".draw",
  # recovery.R
  "nm", "lo", "hi", "covered", "truth"
))

# Build per-group NC flag list for run_mcmc_re / run_mcmc_re_ss.
# Returns a list of integer vectors (one per breakpoint), each with one 0/1
# per RE group (or the sentinel -1L for breakpoints with no omega RE).
#
# re_fraction: NULL (use reparameterise globally), a list of per-breakpoint
#   numeric vectors, or a fibr_smoothbp_advice object.
.build_nc_om_per_group <- function(dm, reparameterise, re_fraction, has_re_om) {

  # Accept fibr advice objects directly
  if (inherits(re_fraction, "fibr_smoothbp_advice")) {
    if (!requireNamespace("fibr", quietly = TRUE))
      stop("Install the 'fibr' package to use a fibr_smoothbp_advice with re_fraction.")
    re_fraction <- fibr::as_smoothbp_re_fraction(re_fraction)
  }

  lapply(seq_along(dm$X_om), function(k) {
    mask <- attr(dm$X_om[[k]], "re_mask")
    if (is.null(mask)) mask <- rep(0L, ncol(dm$X_om[[k]]))
    n_re <- sum(mask == 1L)

    if (n_re == 0L) return(as.integer(-1L))  # no omega RE: sentinel

    if (!is.null(re_fraction)) {
      fracs <- re_fraction[[k]]
      if (is.null(fracs)) fracs <- rep(0, n_re)
      if (length(fracs) == 1L) fracs <- rep(fracs, n_re)
      if (length(fracs) != n_re)
        stop(sprintf(
          "re_fraction[[%d]] has %d values but breakpoint %d has %d RE groups.",
          k, length(fracs), k, n_re))
      as.integer(fracs > 0.5)  # threshold: fraction > 0.5 -> NC
    } else {
      global_nc <- isTRUE(reparameterise == "omega") && has_re_om
      as.integer(rep(if (global_nc) 1L else 0L, n_re))
    }
  })
}

#' Round numeric columns in a data frame
#' @keywords internal
round_df <- function(df, digits = 3) {
  num_cols <- vapply(df, is.numeric, logical(1))
  df[num_cols] <- lapply(df[num_cols], round, digits = digits)
  df
}

#' @importFrom stats dnorm fitted rnorm setNames terms
NULL
