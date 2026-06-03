#' Generate a statistical methods description for a smoothbp_fit object
#'
#' Produces a human- and AI-readable description of the fitted model, including
#' the structural equation, priors, MCMC sampling details, convergence
#' diagnostics, and a reproducibility code snippet. The output is suitable for
#' direct inclusion in the Statistical Analysis section of a scientific
#' manuscript.
#'
#' @details
#' The function name `model_methods()` is used rather than `methods()` to avoid
#' masking the base-R `utils::methods()` function, which lists S3/S4 methods
#' for a generic and does not dispatch on object class.
#'
#' When \code{K = 0} (no breakpoints, i.e. \code{deltas = list()}), the model
#' collapses to a Bayesian linear regression (with optional random intercepts)
#' and the output reflects this accordingly.
#'
#' @param object A \code{smoothbp_fit} or \code{smoothbp_ss_fit} object.
#' @param width Integer; line-wrap width for the narrative paragraph (default
#'   \code{80}).
#' @param ... Unused.
#'
#' @return The full methods report as a single character string (invisibly).
#'   The text is also printed to the console via \code{cat()}.
#'
#' @references
#' Bacon, D. W. & Watts, D. G. (1971). Estimating the transition between two
#'   intersecting straight lines. \emph{Biometrika}, 58(3), 525--534.
#'   \doi{10.2307/2334389}
#'
#' Kuo, L. & Mallick, B. (1998). Variable selection for regression models.
#'   \emph{Sankhya: The Indian Journal of Statistics}, 60(1), 65--81.
#'
#' @examples
#' \dontrun{
#' fit <- smoothbp(y ~ tau, b0 = ~ 1 + (1 | subject), data = dat,
#'                 chains = 4L, iter = 2000L, warmup = 1000L, seed = 42L)
#' model_methods(fit)
#' txt <- model_methods(fit)
#' cat(txt)
#' }
#'
#' @export
model_methods <- function(object, ...) UseMethod("model_methods")

#' @rdname model_methods
#' @export
model_methods.smoothbp_fit <- function(object,
                                        width = 80,
                                        ...) {

  # ---- 0. Feature detection -------------------------------------------
  is_ss      <- inherits(object, "smoothbp_ss_fit")
  dm         <- object$dm
  K          <- length(object$deltas_formula)
  is_linear  <- K == 0L           # no breakpoints -> linear regression
  has_re_b0  <- dm$n_groups_b0 > 0L
  has_re_om  <- isTRUE(dm$has_re_om)
  has_re_b1  <- .has_re(list(dm$X_b1))
  has_re_del <- K > 0L && .has_re(dm$X_deltas)
  n_obs      <- nrow(object$data)
  n_grp      <- dm$n_groups_b0
  n_post     <- (object$iter - object$warmup) * object$chains
  is_robust  <- isTRUE(object$is_robust)

  is_fixed_omega <- if (K > 0L)
    vapply(object$omega_formula, inherits, logical(1L), "smoothbp_fixed")
  else logical(0L)
  is_fixed_rho <- if (K > 0L)
    vapply(object$rho_formula, inherits, logical(1L), "smoothbp_fixed")
  else logical(0L)

  pkg_version <- tryCatch(
    as.character(utils::packageVersion("smoothbp")),
    error = function(e) "unknown"
  )

  # ---- Internal formatting helpers ------------------------------------
  .fmt_norm <- function(row) {
    lb <- if (is.finite(row$lb)) sprintf(", lb = %g", row$lb) else ""
    ub <- if (is.finite(row$ub)) sprintf(", ub = %g", row$ub) else ""
    sprintf("Normal(mean = %g, sd = %g%s%s)", row$mean, row$sd, lb, ub)
  }
  .fmt_ig <- function(p) {
    sprintf("InvGamma(shape = %g, scale = %g)", p$shape, p$scale)
  }
  .fmt_fml <- function(f) {
    if (inherits(f, "smoothbp_fixed"))
      sprintf("fixed(%g)", as.numeric(f))
    else
      paste(deparse(f), collapse = " ")
  }
  .fml_list_str <- function(fmls) {
    if (!length(fmls)) return("list()")
    parts <- vapply(fmls, .fmt_fml, character(1L))
    paste0("list(", paste(parts, collapse = ", "), ")")
  }
  .section <- function(title) {
    bar <- paste0(rep("=", 70L), collapse = "")
    c(bar, title, bar)
  }

  # ---- 1. Narrative paragraph for manuscript --------------------------
  model_type_str <- if (is_linear) {
    if (has_re_b0)
      "Bayesian hierarchical linear regression (with random intercepts)"
    else
      "Bayesian linear regression"
  } else if (is_ss) {
    paste0(
      "Bayesian hierarchical piecewise regression with logistic-smoothed ",
      "transitions and Kuo & Mallick (1998) spike-and-slab variable selection"
    )
  } else {
    "Bayesian hierarchical piecewise regression with logistic-smoothed transitions (adapted from Bacon & Watts, 1971)"
  }

  bp_str <- if (is_linear) {
    "no breakpoints (linear regression)"
  } else if (K == 1L) {
    "a single candidate breakpoint"
  } else {
    sprintf("%d candidate breakpoints", K)
  }

  re_parts <- character(0L)
  if (has_re_b0)  re_parts <- c(re_parts, sprintf("random intercepts across %d groups", n_grp))
  if (has_re_om)  re_parts <- c(re_parts, "random change-point timing across groups")
  if (has_re_b1)  re_parts <- c(re_parts, "random pre-breakpoint slopes across groups")
  if (has_re_del) re_parts <- c(re_parts, "random slope-change parameters across groups")
  re_sent <- if (length(re_parts))
    paste0(" The model includes ", paste(re_parts, collapse = ", "), ".")
  else ""

  fixed_parts <- character(0L)
  for (k in seq_len(K)) {
    if (is_fixed_omega[k])
      fixed_parts <- c(fixed_parts,
        sprintf("omega_%d = %g", k, attr(dm$X_om[[k]], "fixed_value")))
    if (is_fixed_rho[k])
      fixed_parts <- c(fixed_parts,
        sprintf("rho_%d = %g", k, attr(dm$X_rho[[k]], "fixed_value")))
  }
  fixed_sent <- if (length(fixed_parts))
    paste0(" The following parameters were fixed (not estimated): ",
           paste(fixed_parts, collapse = ", "), ".")
  else ""

  ss_sent <- if (is_ss && K > 0L) {
    pi_str <- if (isTRUE(object$spike$learn_pi))
      sprintf("a Beta(%g, %g) hyperprior", object$spike$a, object$spike$b)
    else
      sprintf("a fixed prior inclusion probability of %g", object$spike$pi)
    sprintf(
      " Spike-and-slab priors (Kuo & Mallick, 1998) were placed on slope-change coefficients (delta_k), with %s on the global inclusion probability (pi); posterior inclusion probabilities (PIPs) quantify evidence for each breakpoint.",
      pi_str
    )
  } else ""

  robust_sent <- if (is_robust)
    sprintf(
      " Posterior draws were adjusted using a Bayesian sandwich covariance correction (robustify) clustering on '%s'.",
      object$robust_cluster
    )
  else ""

  bp_dim_sent <- if (is_linear) {
    ""
  } else {
    paste0(" The model includes ", bp_str, " in the '", object$time, "' dimension.")
  }

  narrative <- paste0(
    model_type_str,
    " was fitted to ", n_obs, " observation",
    if (n_obs != 1L) "s" else "",
    if (has_re_b0) sprintf(" from %d groups", n_grp) else "",
    " using the smoothbp R package (version ", pkg_version,
    "; Bindoff, 2026), powered by a Rust-based Metropolis-within-Gibbs sampler.",
    bp_dim_sent,
    re_sent,
    fixed_sent,
    ss_sent,
    robust_sent,
    " Posterior inference used ", object$chains, " Markov chain",
    if (object$chains != 1L) "s" else "",
    " of ", object$iter, " iterations (", object$warmup, " warmup), yielding ",
    n_post, " post-warmup draws.",
    " Results are summarised as posterior mean and 95% credible interval.",
    " Convergence was assessed using the Gelman-Rubin diagnostic (Rhat < 1.05)",
    " and effective sample size (ESS)."
  )

  # ---- 2. Structural equation -----------------------------------------
  time        <- object$time
  re_b0_term  <- if (has_re_b0) " + u_{j}" else ""

  if (is_linear) {
    slope_term   <- sprintf("b_1 * %s", time)
    smooth_terms <- NULL
    eq_str <- sprintf(
      "%s_{ij} = b_0%s + %s + epsilon_{ij}",
      object$response, re_b0_term, slope_term
    )
  } else {
    slope_term <- sprintf("b_1 * (%s - omega_1)", time)
    smooth_terms <- if (K == 1L) {
      sprintf("delta_1 * (%s - omega_1) * logistic(rho_1 * (%s - omega_1))", time, time)
    } else {
      sprintf(
        "sum_{k=1}^{%d} [ delta_k * (%s - omega_k) * logistic(rho_k * (%s - omega_k)) ]",
        K, time, time
      )
    }
    eq_str <- sprintf(
      "%s_{ij} = b_0%s + %s + %s + epsilon_{ij}",
      object$response, re_b0_term, slope_term, smooth_terms
    )
  }

  b0_def <- if (is_linear)
    "  b_0      : Intercept"
  else
    sprintf("  b_0      : Intercept (conditional mean at %s = omega_1)", time)

  b1_def <- if (is_linear)
    "  b_1      : Linear slope"
  else
    "  b_1      : Pre-breakpoint slope"

  param_defs <- c(
    b0_def,
    if (has_re_b0) "  u_{j}    : Random intercept for group j, u_{j} ~ Normal(0, sigma_u^2)"
    else           NULL,
    b1_def
  )
  for (k in seq_len(K)) {
    fv_om  <- attr(dm$X_om[[k]],  "fixed_value")
    fv_rho <- attr(dm$X_rho[[k]], "fixed_value")
    param_defs <- c(param_defs,
      sprintf("  delta_%d  : Change in slope at breakpoint %d", k, k),
      if (!is.null(fv_om))
        sprintf("  omega_%d  : Breakpoint %d location [FIXED = %g, not estimated]", k, k, fv_om)
      else
        sprintf("  omega_%d  : Breakpoint %d location", k, k),
      if (!is.null(fv_rho))
        sprintf("  rho_%d    : Transition %d sharpness [FIXED = %g, not estimated]", k, k, fv_rho)
      else
        sprintf("  rho_%d    : Transition %d sharpness (larger -> sharper kink)", k, k)
    )
  }
  param_defs <- c(param_defs,
    "  sigma    : Residual standard deviation",
    if (has_re_b0)  "  sigma_u  : SD of random intercepts (b_0)" else NULL,
    if (has_re_om)  "  sigma_re_omega_k : SD of random change-point timing (per breakpoint)" else NULL,
    if (has_re_b1)  "  sigma_re_b1 : SD of random slopes (b_1)" else NULL,
    if (has_re_del) "  sigma_re_delta_k : SD of random slope-change parameters (per breakpoint)" else NULL,
    if (!is_linear) "  logistic(x) : 1 / (1 + exp(-x))  [logistic function]" else NULL,
    "  epsilon_{ij} ~ Normal(0, sigma^2)"
  )

  # ---- 3. Formula specification ---------------------------------------
  fml_lines <- c(
    sprintf("  Response ~ Time : %s", deparse(object$formula)),
    sprintf("  b0              : %s", .fmt_fml(object$b0_formula)),
    sprintf("  b1              : %s", .fmt_fml(object$b1_formula))
  )
  for (k in seq_len(K)) {
    fml_lines <- c(fml_lines,
      sprintf("  delta[%d]        : %s", k, .fmt_fml(object$deltas_formula[[k]])),
      sprintf("  omega[%d]        : %s", k, .fmt_fml(object$omega_formula[[k]])),
      sprintf("  rho[%d]          : %s", k, .fmt_fml(object$rho_formula[[k]]))
    )
  }

  # ---- 4. Prior distributions -----------------------------------------
  pv <- object$pv
  prior_lines <- character(0L)

  for (i in seq_len(nrow(pv$b0)))
    prior_lines <- c(prior_lines,
      sprintf("  b0[%s] ~ %s", pv$b0$name[i], .fmt_norm(pv$b0[i, ])))

  for (i in seq_len(nrow(pv$b1)))
    prior_lines <- c(prior_lines,
      sprintf("  b1[%s] ~ %s", pv$b1$name[i], .fmt_norm(pv$b1[i, ])))

  for (k in seq_len(K)) {
    pdk <- pv$deltas[[k]]
    for (i in seq_len(nrow(pdk))) {
      if (is_ss) {
        slab   <- object$spike$slab
        pi_lbl <- if (isTRUE(object$spike$learn_pi))
          sprintf("pi ~ Beta(%g, %g)", object$spike$a, object$spike$b)
        else
          sprintf("pi = %g", object$spike$pi)
        prior_lines <- c(prior_lines,
          sprintf("  delta%d[%s] ~ SpikeSlab(%s, slab = Normal(%g, %g))",
                  k, pdk$name[i], pi_lbl, slab$mean, slab$sd))
      } else {
        prior_lines <- c(prior_lines,
          sprintf("  delta%d[%s] ~ %s", k, pdk$name[i], .fmt_norm(pdk[i, ])))
      }
    }

    fv_om <- attr(dm$X_om[[k]], "fixed_value")
    if (!is.null(fv_om)) {
      prior_lines <- c(prior_lines,
        sprintf("  omega%d = fixed(%g)  [not sampled]", k, fv_om))
    } else {
      pok <- pv$om[[k]]
      for (i in seq_len(nrow(pok)))
        prior_lines <- c(prior_lines,
          sprintf("  omega%d[%s] ~ %s", k, pok$name[i], .fmt_norm(pok[i, ])))
    }

    fv_rho <- attr(dm$X_rho[[k]], "fixed_value")
    if (!is.null(fv_rho)) {
      prior_lines <- c(prior_lines,
        sprintf("  rho%d = fixed(%g)  [not sampled]", k, fv_rho))
    } else {
      prk <- pv$rho[[k]]
      for (i in seq_len(nrow(prk)))
        prior_lines <- c(prior_lines,
          sprintf("  rho%d[%s] ~ %s", k, prk$name[i], .fmt_norm(prk[i, ])))
    }
  }

  prior_lines <- c(prior_lines,
    sprintf("  sigma   ~ %s", .fmt_ig(object$priors$sigma)))
  if (has_re_b0)
    prior_lines <- c(prior_lines,
      sprintf("  sigma_u ~ %s", .fmt_ig(object$priors$sigma_u)))
  if (has_re_om)
    prior_lines <- c(prior_lines,
      sprintf("  sigma_re_omega ~ %s  [shared across breakpoints]",
              .fmt_ig(object$priors$sigma_re_om)))
  if (has_re_b1)
    prior_lines <- c(prior_lines,
      sprintf("  sigma_re_b1 ~ %s", .fmt_ig(object$priors$sigma_re_b1)))
  if (has_re_del)
    prior_lines <- c(prior_lines,
      sprintf("  sigma_re_deltas ~ %s  [shared across breakpoints]",
              .fmt_ig(object$priors$sigma_re_deltas)))

  # ---- 5. MCMC algorithm ----------------------------------------------
  linear_par_names <- c("b0", "b1", if (K > 0L) "delta_k", if (has_re_b0) "u_j")
  linear_pars <- paste(linear_par_names, collapse = ", ")
  any_nonlin  <- K > 0L && (any(!is_fixed_omega) || any(!is_fixed_rho))

  nonlin_desc <- if (is_linear) {
    "  Nonlinear pars: none (linear regression; HMC not used)"
  } else if (any_nonlin) {
    c(
      "  Nonlinear pars: omega_k, rho_k updated by Hamiltonian Monte Carlo (HMC)",
      "                  with forward-mode automatic differentiation for leapfrog gradients"
    )
  } else {
    "  Nonlinear pars: all change-point parameters fixed; HMC not used"
  }

  algo_lines <- c(
    "  Algorithm    : Metropolis-within-Gibbs (custom Rust backend via extendr)",
    sprintf("  Linear pars  : %s updated by block conjugate Gibbs (exact draws)", linear_pars),
    nonlin_desc,
    if (is_ss && K > 0L)
      "  SS indicators: gamma_k updated by Gibbs from Bernoulli full conditional"
    else NULL,
    sprintf("  Chains       : %d", object$chains),
    sprintf("  Iterations   : %d total (%d warmup + %d post-warmup per chain)",
            object$iter, object$warmup, object$iter - object$warmup),
    sprintf("  Total draws  : %d", n_post),
    sprintf("  Target accept: %.2f  (HMC step size tuned during warmup)", object$target_accept),
    sprintf("  Seed         : %d", object$seed)
  )

  # ---- 6. Convergence check (warn on pathologies) ---------------------
  s_all <- tryCatch(summary(object, effects = "all"), error = function(e) NULL)

  # Collect pathologies
  max_rhat  <- NA_real_
  min_ebulk <- NA_real_
  min_etail <- NA_real_
  if (!is.null(s_all) && nrow(s_all) > 0L) {
    .vget <- function(col) {
      if (!col %in% names(s_all)) return(numeric(0L))
      v <- as.numeric(s_all[[col]]); v[is.finite(v)]
    }
    rv <- .vget("Rhat");     if (length(rv)) max_rhat  <- max(rv)
    eb <- .vget("Bulk_ESS"); if (length(eb)) min_ebulk <- min(eb)
    et <- .vget("Tail_ESS"); if (length(et)) min_etail <- min(et)
  }
  n_diverg <- object$n_divergent

  rhat_warn  <- !is.na(max_rhat)  && max_rhat  > 1.05
  ebulk_warn <- !is.na(min_ebulk) && min_ebulk < 100L
  etail_warn <- !is.na(min_etail) && min_etail < 100L
  divg_warn  <- n_diverg > 0L
  any_warn   <- rhat_warn || ebulk_warn || etail_warn || divg_warn

  if (!any_warn) {
    conv_lines <- c(
      "  No convergence concerns detected.",
      sprintf("  Max Rhat = %.3f,  min bulk ESS = %d,  min tail ESS = %d,  divergent transitions = %d.",
              if (is.na(max_rhat)) 0 else max_rhat,
              if (is.na(min_ebulk)) 0L else as.integer(min_ebulk),
              if (is.na(min_etail)) 0L else as.integer(min_etail),
              n_diverg),
      "  Use model_results() for the full posterior summary."
    )
  } else {
    issues <- character(0L)
    if (rhat_warn)
      issues <- c(issues,
        sprintf("  * Max Rhat = %.3f  (threshold 1.05) -- chains may not have mixed.", max_rhat))
    if (ebulk_warn)
      issues <- c(issues,
        sprintf("  * Min bulk ESS = %d  (threshold 100) -- posterior bulk may be poorly sampled.",
                as.integer(min_ebulk)))
    if (etail_warn)
      issues <- c(issues,
        sprintf("  * Min tail ESS = %d  (threshold 100) -- posterior tails may be poorly sampled.",
                as.integer(min_etail)))
    if (divg_warn)
      issues <- c(issues,
        sprintf("  * %d divergent transition%s -- possible model misspecification or geometry problem.",
                n_diverg, if (n_diverg == 1L) "" else "s"))

    conv_lines <- c(
      "  !! CONVERGENCE WARNING !!",
      "  The following issues were detected; consider whether the model is",
      "  correctly specified before reporting results:",
      "",
      issues,
      "",
      "  Suggested diagnostics: trace_plot(fit), pp_check(fit).",
      "  Common causes: poorly chosen priors on omega/rho, insufficient warmup,",
      "  or a breakpoint location prior that extends beyond the data range.",
      "  Use model_results() for the full posterior summary."
    )
  }

  # ---- 7. Reproducibility code snippet --------------------------------
  fn_call <- if (is_ss) "smoothbp_ss" else "smoothbp"
  repro_lines <- c(
    sprintf("library(smoothbp)  # version %s", pkg_version),
    sprintf("%s <- %s(", deparse(object$formula[[2L]]), fn_call),
    sprintf("  formula = %s,", deparse(object$formula)),
    sprintf("  b0      = %s,", .fmt_fml(object$b0_formula)),
    sprintf("  b1      = %s,", .fmt_fml(object$b1_formula)),
    sprintf("  deltas  = %s,", .fml_list_str(object$deltas_formula)),
    sprintf("  omega   = %s,", .fml_list_str(object$omega_formula)),
    sprintf("  rho     = %s,", .fml_list_str(object$rho_formula)),
    "  data    = <your_data_frame>,",
    "  ## priors: see Prior Distributions section above",
    sprintf("  chains  = %dL, iter = %dL, warmup = %dL,",
            object$chains, object$iter, object$warmup),
    sprintf("  seed    = %dL", object$seed),
    ")"
  )

  # ---- Assemble full report -------------------------------------------
  report <- c(
    .section("SMOOTHBP: STATISTICAL METHODS REPORT"),
    "",
    .section("NARRATIVE  (paste into manuscript Methods section)"),
    "",
    strwrap(narrative, width = width),
    "",
    .section("MODEL SPECIFICATION"),
    "",
    "Formula syntax:",
    fml_lines,
    "",
    "Structural equation:",
    paste0("  ", eq_str),
    "",
    "Parameter definitions:",
    param_defs,
    "",
    .section("PRIOR DISTRIBUTIONS"),
    "",
    prior_lines,
    "",
    .section("MCMC SAMPLING ALGORITHM"),
    "",
    algo_lines,
    "",
    .section("CONVERGENCE NOTE"),
    "",
    conv_lines
  )

  report <- c(report,
    "",
    .section("REPRODUCIBILITY"),
    "",
    repro_lines,
    "",
    .section("REFERENCES"),
    "",
    paste0(
      "Bacon, D. W. & Watts, D. G. (1971). Estimating the transition between ",
      "two intersecting straight lines. Biometrika, 58(3), 525-534. ",
      "doi:10.2307/2334389"
    ),
    "",
    paste0(
      "Kuo, L. & Mallick, B. (1998). Variable selection for regression models. ",
      "Sankhya: The Indian Journal of Statistics, 60(1), 65-81."
    ),
    "",
    paste0(
      "Bindoff, A. D. (2026). smoothbp: Hierarchical Piecewise Regression with ",
      "Smoothed Change-Points. R package version ", pkg_version, ". ",
      "https://github.com/ABindoff/smoothbp"
    ),
    paste0(rep("=", 70L), collapse = "")
  )

  out <- paste(report, collapse = "\n")
  cat(out, "\n")
  invisible(out)
}
