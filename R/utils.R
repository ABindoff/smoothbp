# Internal utilities and global variable declarations
# ---------------------------------------------------------------------------

# Suppress R CMD check NOTEs for ggplot2 column names used via aes() and
# posterior draws data frames that are created at runtime.
utils::globalVariables(c(
  # plot_methods.R (.build_trace, .build_density)
  "iteration", "value", "chain", "xmin", "xmax", "ymin", "ymax",
  # postprocess.R (pp_check)
  "y", ".draw",
  # recovery.R
  "nm", "lo", "hi", "covered", "truth"
))

#' @importFrom stats dnorm fitted rnorm setNames terms
NULL
