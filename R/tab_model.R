# tab_model.R
#
# Presents fixed effects from one or more smoothbp_fit objects as a formatted
# table, with parameters on rows and models in columns — similar in spirit to
# sjPlot::tab_model() for brmsfit objects.
#
# Requires: gt (for the rendered table), dplyr (for tidy construction).
# Falls back to knitr::kable() if gt is not installed.

# ---------------------------------------------------------------------------
#' Fixed-effects table for smoothbp_fit objects
#'
#' Collects posterior summaries from one or more `smoothbp_fit` objects and
#' displays them in a single table with parameters on rows and models in
#' columns.  All parameters present in any model are shown; models that do not
#' include a given parameter display a dash.
#'
#' @param ... One or more `smoothbp_fit` or `smoothbp_ss_fit` objects.
#' @param labels Character vector of column headers, one per model.  If `NULL`
#'   (default) the deparsed call names are used.
#' @param digits Integer; number of decimal places (default `2`).
#' @param fmt Cell format: `"mean [CI]"` (default) shows mean and 95% credible
#'   interval; `"mean (SD)"` shows mean and posterior SD.
#' @param show_rhat Logical; append \eqn{\hat{R}} to each cell (default
#'   `FALSE`).
#'
#' @return A `gt_tbl` object (or a `knitr_kable` if **gt** is not installed).
#'
#' @examples
#' \dontrun{
#' tab_smoothbp(m.ep.pw1, m.wo.pw1, m.la.pw1,
#'              labels = c("Episodic", "Working", "Language"))
#' }
#' @export
tab_smoothbp <- function(...,
                         labels    = NULL,
                         digits    = 2,
                         fmt       = c("mean [CI]", "mean (SD)"),
                         show_rhat = FALSE) {

  fmt    <- match.arg(fmt)
  models <- list(...)
  n      <- length(models)

  if (n == 0L) stop("Supply at least one smoothbp_fit object.")
  if (!all(sapply(models, inherits, what = "smoothbp_fit"))) {
    stop("All positional arguments must be smoothbp_fit objects.")
  }

  # ---- Column labels --------------------------------------------------------
  if (is.null(labels)) {
    cl      <- match.call()
    cl_args <- as.list(cl)[-1]
    cl_args <- cl_args[!names(cl_args) %in% c("labels", "digits", "fmt", "show_rhat")]
    nm      <- sapply(cl_args, deparse)
    labels  <- if (length(nm) == n) nm else paste0("Model ", seq_len(n))
  }
  if (length(labels) != n) {
    stop("`labels` must have one entry per model (", n, " models supplied).")
  }

  # ---- Format one summary row into a single string -------------------------
  # smoothbp_ss names CI columns "2.5%" / "97.5%"; smoothbp uses "Q2.5" / "Q97.5"
  fmt_cell <- function(row, lo_col, hi_col) {
    m  <- round(as.numeric(row[["mean"]]),  digits)
    lo <- round(as.numeric(row[[lo_col]]),  digits)
    hi <- round(as.numeric(row[[hi_col]]),  digits)
    sd <- round(as.numeric(row[["SD"]]),    digits)
    rh <- round(as.numeric(row[["Rhat"]]),  3)

    cell <- if (fmt == "mean [CI]") {
      sprintf("%.*f [%.*f, %.*f]", digits, m, digits, lo, digits, hi)
    } else {
      sprintf("%.*f (%.*f)", digits, m, digits, sd)
    }
    if (show_rhat) cell <- paste0(cell, "  \u0052\u0302=", rh)
    cell
  }

  # ---- Extract and format fixed effects from each model --------------------
  cols <- lapply(models, function(fit) {
    s      <- summary(fit, effects = "fixed")
    lo_col <- if ("2.5%"  %in% names(s)) "2.5%"  else "Q2.5"
    hi_col <- if ("97.5%" %in% names(s)) "97.5%" else "Q97.5"
    vals   <- vapply(seq_len(nrow(s)),
                     function(i) fmt_cell(s[i, ], lo_col, hi_col),
                     character(1))
    stats::setNames(vals, s$variable)
  })

  # ---- Union of parameters in natural order --------------------------------
  all_vars <- unique(unlist(lapply(cols, names)))

  # ---- Assemble wide data frame --------------------------------------------
  tbl <- data.frame(Parameter = all_vars, stringsAsFactors = FALSE)
  for (j in seq_len(n)) {
    tbl[[labels[j]]] <- cols[[j]][all_vars]   # NA for absent parameters
  }

  # ---- Parse "block_term" variable names -----------------------------------
  # e.g. "delta1_(Intercept)" -> block = "delta1", term = "(Intercept)"
  tbl$block <- sub("_.*", "", tbl$Parameter)
  tbl$term  <- sub("^[^_]+_", "", tbl$Parameter)
  solo      <- tbl$block == tbl$term
  tbl$term[solo] <- tbl$block[solo]

  # Map block prefixes to pretty labels
  pretty_block <- function(b) {
    if (b == "b0") return("\u03B2\u2080 \u2013 Intercept and covariates")
    if (b == "b1") return("\u03B2\u2081 \u2013 Pre-transition slope")
    if (b == "sigma") return("\u03C3 \u2013 Residual SD")
    if (grepl("^delta([0-9]+)$", b)) {
       k <- sub("delta", "", b)
       return(sprintf("\u0394\u03B2 %s \u2013 Slope change at BP%s", k, k))
    }
    if (grepl("^omega([0-9]+)$", b)) {
       k <- sub("omega", "", b)
       return(sprintf("\u03C9%s \u2013 Transition point %s", k, k))
    }
    if (grepl("^rho([0-9]+)$", b)) {
       k <- sub("rho", "", b)
       return(sprintf("\u03C1%s \u2013 Sharpness %s", k, k))
    }
    if (grepl("^gamma_b1", b)) return("\u03B3 b1 \u2013 Inclusion (b1)")
    if (grepl("^gamma_delta([0-9]+)$", b)) {
       k <- sub("gamma_delta", "", b)
       return(sprintf("\u03B3%s \u2013 Inclusion (BP%s)", k, k))
    }
    b
  }
  tbl$block_label <- vapply(tbl$block, pretty_block, character(1))

  tbl_out <- tbl[, c("block_label", "term", labels)]

  # ---- Render --------------------------------------------------------------
  if (!requireNamespace("gt", quietly = TRUE)) {
    message("Install the 'gt' package for a richer table. Falling back to knitr::kable().")
    return(knitr::kable(tbl_out,
                        col.names = c("Block", "Parameter", labels),
                        row.names = FALSE,
                        align     = c("l", "l", rep("c", n))))
  }

  subtitle_text <- if (fmt == "mean [CI]") "Mean [95% credible interval]" else "Mean (posterior SD)"

  tbl_out |>
    gt::gt(groupname_col = "block_label", rowname_col = "term") |>
    gt::tab_header(title    = "Fixed effects",
                   subtitle = subtitle_text) |>
    gt::cols_align(align = "right", columns = dplyr::all_of(labels)) |>
    gt::cols_align(align = "left",  columns = "term") |>
    gt::tab_style(
      style     = gt::cell_text(weight = "bold"),
      locations = gt::cells_row_groups()
    ) |>
    gt::tab_style(
      style     = gt::cell_fill(color = "#efefef"),
      locations = gt::cells_row_groups()
    ) |>
    gt::sub_missing(missing_text = "\u2014") |>
    gt::opt_table_font(font = list(gt::google_font("IBM Plex Mono")), size = 13) |>
    gt::tab_options(table.width           = gt::pct(100),
                    row_group.border.top.width    = gt::px(2),
                    row_group.border.bottom.width = gt::px(1))
}
