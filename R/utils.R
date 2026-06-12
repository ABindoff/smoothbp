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
.build_nc_om_per_group <- function(dm, reparameterise, has_re_om) {
  lapply(seq_along(dm$X_om), function(k) {
    mask <- attr(dm$X_om[[k]], "re_mask")
    if (is.null(mask)) mask <- rep(0L, ncol(dm$X_om[[k]]))
    n_re <- sum(mask == 1L)

    if (n_re == 0L) return(as.integer(-1L))  # no omega RE: sentinel

    global_nc <- isTRUE(reparameterise == "omega") && has_re_om
    as.integer(rep(if (global_nc) 1L else 0L, n_re))
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
