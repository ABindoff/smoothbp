# ---------------------------------------------------------------------------
# hypothesis() S3 method for smoothbp_fit
# ---------------------------------------------------------------------------

#' Test hypotheses and compute evidence ratios from posterior draws
#'
#' Evaluates one or more hypothesis expressions against the posterior draws of
#' a \code{smoothbp_fit} object, returning posterior probabilities and evidence
#' ratios.
#'
#' @section Hypothesis syntax:
#'
#' Write hypotheses as character strings using exact parameter names as they
#' appear in \code{fit$param_names} (e.g. \code{"b2_(Intercept)"},
#' \code{"omega_(Intercept)"}). No backtick-quoting is needed; the function
#' handles special characters internally.
#'
#' Two forms are accepted:
#'
#' \describe{
#'   \item{Directional}{An expression containing \code{>}, \code{<},
#'     \code{>=}, or \code{<=}.  The hypothesis is evaluated as a contrast:
#'     the left-hand side minus the right-hand side (direction-adjusted), and
#'     \eqn{P(H \mid \text{data})} is the proportion of posterior draws
#'     satisfying the condition.
#'     Examples: \code{"b2_(Intercept) > 0"},
#'     \code{"omega_(Intercept) < 4"},
#'     \code{"b2_(Intercept) - b1_(Intercept) > 0"}.}
#'   \item{Contrast}{A numeric expression without a comparison operator.  The
#'     function summarises the posterior distribution of the derived quantity
#'     and reports \eqn{P(\text{expression} > 0)} as the directional
#'     probability.
#'     Example: \code{"b2_(Intercept) - b1_(Intercept)"}.}
#' }
#'
#' Point-null hypotheses (\code{==}) are not supported because they require
#' the Savage-Dickey density ratio; use \code{bayestestR::rope()} for
#' interval-based equivalence testing instead.
#'
#' @section Evidence ratio interpretation:
#'
#' \deqn{ER = \frac{P(H \mid \text{data})}{1 - P(H \mid \text{data})}}
#'
#' An ER of 19 corresponds to \eqn{P(H) = 0.95}; ER = 1 means the posterior
#' is equally split.  Star codes: \code{***} ER > 99, \code{**} ER > 19,
#' \code{*} ER > 3.
#'
#' @param object A \code{smoothbp_fit} object.
#' @param hypotheses A character vector of hypothesis strings.
#' @param ci  Width of the credible interval for the underlying contrast.
#'   Default \code{0.95}.
#' @param ... Unused.
#'
#' @return An object of class \code{c("smoothbp_hypothesis", "data.frame")}
#'   with one row per hypothesis and columns:
#'   \describe{
#'     \item{\code{Hypothesis}}{The original hypothesis string.}
#'     \item{\code{Estimate}}{Posterior mean of the contrast.}
#'     \item{\code{Est.Error}}{Posterior SD of the contrast.}
#'     \item{\code{CI.lower}, \code{CI.upper}}{Credible interval bounds.}
#'     \item{\code{P(H)}}{Posterior probability of the hypothesis.}
#'     \item{\code{Evid.Ratio}}{Evidence ratio \eqn{P(H)/(1-P(H))}.}
#'     \item{\code{Star}}{Informal star coding based on the evidence ratio.}
#'   }
#'
#' @examples
#' \dontrun{
#' # Is the change in slope positive?
#' hypothesis(fit, "b2_(Intercept) > 0")
#'
#' # Does the change-point fall before time 4?
#' hypothesis(fit, "omega_(Intercept) < 4")
#'
#' # Multiple hypotheses at once
#' hypothesis(fit, c(
#'   "b2_(Intercept) > 0",
#'   "omega_(Intercept) < 4",
#'   "b2_(Intercept) - b1_(Intercept) > 0"
#' ))
#'
#' # Posterior summary of a contrast (no comparison operator)
#' hypothesis(fit, "b2_(Intercept) - b1_(Intercept)")
#' }
#'
#' @export
hypothesis <- function(object, ...) UseMethod("hypothesis")

#' @export
hypothesis.smoothbp_fit <- function(object, hypotheses, ci = 0.95, ...) {
  if (!is.character(hypotheses) || length(hypotheses) == 0) {
    stop("`hypotheses` must be a non-empty character vector.")
  }
  if (any(grepl("==|!=", hypotheses))) {
    stop(
      "Point-null hypotheses (==) are not supported.\n",
      "For equivalence testing use bayestestR::rope() on posterior_draws(fit)."
    )
  }
  stopifnot(ci > 0, ci < 1)

  # Build a plain data frame of draws (no .chain / .iteration / .draw cols)
  draws_df <- as.data.frame(posterior::as_draws_df(object$draws))
  draws_df  <- draws_df[, !grepl("^\\.", names(draws_df)), drop = FALSE]

  # Sort parameter names longest-first to prevent partial substitution
  # (e.g. "sigma_u" must be replaced before "sigma")
  param_names   <- names(draws_df)
  sorted_params <- param_names[order(nchar(param_names), decreasing = TRUE)]

  # Map each parameter name to a syntactically safe placeholder
  placeholders <- setNames(
    paste0("..p", seq_along(sorted_params), ".."),
    sorted_params
  )
  safe_df       <- draws_df
  names(safe_df) <- placeholders[names(safe_df)]

  rows <- lapply(hypotheses, function(h) {
    .eval_hypothesis(h, safe_df, sorted_params, placeholders, ci)
  })

  out <- do.call(rbind, rows)
  structure(
    out,
    class    = c("smoothbp_hypothesis", "data.frame"),
    ci       = ci,
    n_draws  = nrow(draws_df),
    fit_call = object$formula
  )
}

# ---------------------------------------------------------------------------
# Internal: evaluate one hypothesis string
# ---------------------------------------------------------------------------

.eval_hypothesis <- function(hyp_str, safe_df, sorted_params, placeholders, ci) {

  # --- Step 1: escape parameter names in the expression string --------------
  escaped <- hyp_str
  for (nm in sorted_params) {
    escaped <- gsub(nm, placeholders[[nm]], escaped, fixed = TRUE)
  }

  # --- Step 2: detect comparison operator and split -------------------------
  # Matches >=, <=, >, < (but not -> or <-)
  op_match <- regexpr("(?<![=-])(>=|<=|>|<)(?![-=])", escaped, perl = TRUE)

  has_comparison <- op_match > 0

  if (has_comparison) {
    op_len <- attr(op_match, "match.length")
    op     <- substr(escaped, op_match, op_match + op_len - 1)
    lhs_e  <- trimws(substr(escaped, 1, op_match - 1))
    rhs_e  <- trimws(substr(escaped, op_match + op_len, nchar(escaped)))

    # Evaluate each side
    lhs_vals <- .eval_expr(lhs_e, safe_df, hyp_str)
    rhs_vals <- .eval_expr(rhs_e, safe_df, hyp_str)

    # Contrast = LHS - RHS, direction-adjusted so P(contrast > 0) == P(H)
    contrast <- if (op %in% c(">", ">=")) lhs_vals - rhs_vals
                else                       rhs_vals - lhs_vals

  } else {
    # Pure contrast expression
    contrast <- .eval_expr(escaped, safe_df, hyp_str)
    op       <- NA_character_
  }

  # --- Step 3: posterior summary of the contrast ----------------------------
  lo_p  <- (1 - ci) / 2
  hi_p  <- 1 - lo_p
  p_h   <- mean(contrast > 0, na.rm = TRUE)
  er    <- if (p_h >= 1 - .Machine$double.eps) {
             Inf
           } else if (p_h <= .Machine$double.eps) {
             0
           } else {
             p_h / (1 - p_h)
           }

  data.frame(
    Hypothesis  = hyp_str,
    Estimate    = round(mean(contrast, na.rm = TRUE), 4),
    Est.Error   = round(stats::sd(contrast, na.rm = TRUE), 4),
    CI.lower    = round(stats::quantile(contrast, lo_p, names = FALSE), 4),
    CI.upper    = round(stats::quantile(contrast, hi_p, names = FALSE), 4),
    "P(H)"      = round(p_h, 4),
    Evid.Ratio  = round(er, 2),
    Star        = .er_star(er),
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
}

.eval_expr <- function(expr_str, safe_df, original_hyp) {
  result <- tryCatch(
    eval(parse(text = expr_str), envir = safe_df),
    error = function(e) {
      stop(
        "Could not evaluate hypothesis '", original_hyp, "'.\n",
        "Check that all parameter names are spelled exactly as in ",
        "fit$param_names.\n",
        "  Parser error: ", conditionMessage(e),
        call. = FALSE
      )
    }
  )
  if (!is.numeric(result)) {
    stop(
      "Expression in '", original_hyp, "' did not evaluate to a numeric vector.\n",
      "Did you mean to include a comparison operator (>, <, >=, <=)?",
      call. = FALSE
    )
  }
  result
}

.er_star <- function(er) {
  if (is.infinite(er) || er > 99) "***"
  else if (er > 19)               "**"
  else if (er >  3)               "*"
  else                            ""
}

# ---------------------------------------------------------------------------
# S3 print method
# ---------------------------------------------------------------------------

#' @export
print.smoothbp_hypothesis <- function(x, digits = 3, ...) {
  ci      <- attr(x, "ci")
  n_draws <- attr(x, "n_draws")
  ci_pct  <- round(ci * 100)

  cat(sprintf(
    "Hypothesis tests for smoothbp_fit  (%d posterior draws)\n",
    n_draws
  ))
  cat(sprintf(
    "Credible interval: %d%%   Evidence ratio = P(H) / (1 - P(H))\n",
    ci_pct
  ))
  cat(sprintf("Stars: *** ER > 99  ** ER > 19  * ER > 3\n"))
  cat(rep("-", 72), "\n", sep = "")

  out <- x
  class(out) <- "data.frame"

  # Shorten column names for display
  names(out)[names(out) == "CI.lower"] <- sprintf("CI.lo(%d%%)", ci_pct)
  names(out)[names(out) == "CI.upper"] <- sprintf("CI.hi(%d%%)", ci_pct)

  # Round numeric columns
  num_cols <- sapply(out, is.numeric)
  out[num_cols] <- lapply(out[num_cols], round, digits = digits)

  print(out, row.names = FALSE)
  invisible(x)
}
