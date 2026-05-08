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

#' Round numeric columns in a data frame
#' @keywords internal
round_df <- function(df, digits = 3) {
  num_cols <- vapply(df, is.numeric, logical(1))
  df[num_cols] <- lapply(df[num_cols], round, digits = digits)
  df
}

#' @importFrom stats dnorm fitted rnorm setNames terms
NULL
