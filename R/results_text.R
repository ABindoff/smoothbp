#' Report the results of a smoothbp_fit object
#'
#' Produces a human- and AI-readable results report, including a plain-English
#' narrative, the fitted model equation with posterior mean coefficients
#' substituted to 3 d.p., per-breakpoint summaries, the full parameter table,
#' and convergence diagnostics.  The output is entirely self-contained: no
#' package documentation or source code is needed to interpret it.
#'
#' @details
#' The function name `model_results()` is used rather than `results()` for
#' consistency with \code{\link{model_methods}()} and to avoid any namespace
#' conflict with results functions in other packages.
#'
#' For models with covariates on any structural parameter, the substituted
#' equation shows the intercept-only (reference-level) value; the full set of
#' coefficients is listed in the \strong{PARAMETER ESTIMATES} section.
#'
#' @param object A \code{smoothbp_fit} or \code{smoothbp_ss_fit} object.
#' @param digits Integer; decimal places for all numerical output (default \code{3}).
#' @param width Integer; line-wrap width for narrative text (default \code{80}).
#' @param ... Unused.
#'
#' @return The full results report as a single character string (invisibly).
#'   The text is also printed to the console via \code{cat()}.
#'
#' @examples
#' \dontrun{
#' fit <- smoothbp(y ~ tau, b0 = ~ 1 + (1 | subject), data = dat,
#'                 chains = 4L, iter = 2000L, warmup = 1000L, seed = 42L)
#' model_results(fit)
#' txt <- model_results(fit)
#' }
#'
#' @export
model_results <- function(object, ...) UseMethod("model_results")

#' @rdname model_results
#' @export
model_results.smoothbp_fit <- function(object,
                                        digits = 3,
                                        width  = 80,
                                        ...) {

  # ---- 0. Feature detection -------------------------------------------
  is_ss      <- inherits(object, "smoothbp_ss_fit")
  dm         <- object$dm
  K          <- length(object$deltas_formula)
  is_linear  <- K == 0L
  has_re_b0  <- dm$n_groups_b0 > 0L
  has_re_om  <- isTRUE(dm$has_re_om)
  has_re_b1  <- .has_re(list(dm$X_b1))
  has_re_del <- K > 0L && .has_re(dm$X_deltas)
  n_obs      <- nrow(object$data)
  n_grp      <- dm$n_groups_b0
  n_post     <- (object$iter - object$warmup) * object$chains
  time       <- object$time
  resp       <- object$response

  is_fixed_omega <- if (K > 0L)
    vapply(object$omega_formula, inherits, logical(1L), "smoothbp_fixed")
  else logical(0L)
  is_fixed_rho <- if (K > 0L)
    vapply(object$rho_formula, inherits, logical(1L), "smoothbp_fixed")
  else logical(0L)

  # ---- Posterior summary tables ---------------------------------------
  s_fixed <- tryCatch(summary(object, effects = "fixed"),   error = function(e) NULL)
  s_rpar  <- tryCatch(summary(object, effects = "ran_pars"), error = function(e) NULL)
  s_rval  <- tryCatch(summary(object, effects = "ran_vals"), error = function(e) NULL)

  # ---- Internal helpers -----------------------------------------------
  d <- digits

  .val <- function(s, pat, col = "mean") {
    if (is.null(s)) return(NA_real_)
    r <- s[grepl(pat, s$variable, perl = TRUE), ]
    if (!nrow(r)) return(NA_real_)
    as.numeric(r[[col]][1L])
  }

  # Robust CI column detection (summary() uses Q2.5 / Q97.5)
  .lo_col <- function(s) {
    if ("Q2.5"  %in% names(s)) return("Q2.5")
    if ("2.5%"  %in% names(s)) return("2.5%")
    NULL
  }
  .hi_col <- function(s) {
    if ("Q97.5" %in% names(s)) return("Q97.5")
    if ("97.5%" %in% names(s)) return("97.5%")
    NULL
  }

  .ci <- function(s, pat) {
    if (is.null(s)) return("N/A")
    r   <- s[grepl(pat, s$variable, perl = TRUE), ]
    if (!nrow(r)) return("N/A")
    m   <- round(as.numeric(r$mean[1L]),          d)
    lc  <- .lo_col(s); hc <- .hi_col(s)
    if (is.null(lc) || is.null(hc))
      return(sprintf("%.*f", d, m))
    lo  <- round(as.numeric(r[[lc]][1L]), d)
    hi  <- round(as.numeric(r[[hc]][1L]), d)
    sprintf("%.*f [95%% CI: %.*f, %.*f]", d, m, d, lo, d, hi)
  }

  .fmt <- function(x) sprintf("%.*f", d, x)

  # Build a human-readable linear-predictor string from a pv data-frame row set.
  # Intercept contributes a constant; other columns contribute "value * varname".
  .lp_str <- function(pv_df, means) {
    parts <- character(nrow(pv_df))
    for (i in seq_len(nrow(pv_df))) {
      nm <- pv_df$name[i]
      v  <- .fmt(means[i])
      if (nm == "(Intercept)") parts[i] <- v
      else                     parts[i] <- sprintf("(%s) * %s", v, nm)
    }
    paste(parts, collapse = " + ")
  }

  # ---- Extract intercept-level posterior means per parameter ----------
  # (Used both for the substituted equation and narratives)
  b0_int  <- .val(s_fixed, "^b0_\\(Intercept\\)")
  b1_int  <- .val(s_fixed, "^b1_\\(Intercept\\)")
  sigma_v <- .val(s_fixed, "^sigma$")

  om_int  <- rho_int  <- del_int  <- numeric(K)
  for (k in seq_len(K)) {
    del_int[k] <- .val(s_fixed, sprintf("^delta%d_\\(Intercept\\)", k))
    om_int[k]  <- if (is_fixed_omega[k])
      attr(dm$X_om[[k]],  "fixed_value")
    else
      .val(s_fixed, sprintf("^omega%d_\\(Intercept\\)", k))
    rho_int[k] <- if (is_fixed_rho[k])
      attr(dm$X_rho[[k]], "fixed_value")
    else
      .val(s_fixed, sprintf("^rho%d_\\(Intercept\\)", k))
  }

  # Detect which parameters carry covariates beyond the intercept
  .has_cov <- function(pv_df) nrow(pv_df) > 1L
  b0_cov  <- .has_cov(object$pv$b0)
  b1_cov  <- .has_cov(object$pv$b1)
  del_cov <- if (K > 0L) vapply(object$pv$deltas, .has_cov, logical(1L)) else logical(0L)
  om_cov  <- if (K > 0L) vapply(seq_len(K), function(k) {
    !is_fixed_omega[k] && .has_cov(object$pv$om[[k]])
  }, logical(1L)) else logical(0L)
  rho_cov <- if (K > 0L) vapply(seq_len(K), function(k) {
    !is_fixed_rho[k] && .has_cov(object$pv$rho[[k]])
  }, logical(1L)) else logical(0L)
  any_cov <- b0_cov || b1_cov || any(del_cov) || any(om_cov) || any(rho_cov)

  # ---- 1. Convergence status (needed by narrative) --------------------
  max_rhat  <- NA_real_
  min_ess_b <- NA_real_
  if (!is.null(s_fixed)) {
    rv <- as.numeric(s_fixed$Rhat); rv <- rv[is.finite(rv)]
    if (length(rv)) max_rhat <- max(rv)
    eb <- as.numeric(if ("Bulk_ESS" %in% names(s_fixed)) s_fixed$Bulk_ESS else numeric(0))
    eb <- eb[is.finite(eb)]
    if (length(eb)) min_ess_b <- min(eb)
  }
  n_diverg   <- object$n_divergent
  conv_ok    <- (is.na(max_rhat) || max_rhat <= 1.05) && n_diverg == 0L
  conv_sent  <- if (conv_ok) {
    "MCMC chains converged satisfactorily (all Rhat <= 1.05, no divergent transitions)."
  } else {
    issues <- character(0L)
    if (!is.na(max_rhat) && max_rhat > 1.05)
      issues <- c(issues, sprintf("max Rhat = %.3f (threshold 1.05)", max_rhat))
    if (n_diverg > 0L)
      issues <- c(issues, sprintf("%d divergent transition(s)", n_diverg))
    paste0("Convergence concerns detected: ", paste(issues, collapse = "; "),
           ". Interpret with caution.")
  }

  # ---- 2. PIPs for SS -------------------------------------------------
  pip_by_k <- numeric(K)
  if (is_ss && !is.null(s_fixed)) {
    for (k in seq_len(K)) {
      pat <- sprintf("^gamma_delta%d_\\(Intercept\\)", k)
      pip_by_k[k] <- .val(s_fixed, pat)
    }
  }

  pip_strength <- function(p) {
    if (is.na(p))   return("unknown")
    if (p >= 0.95)  return("strong")
    if (p >= 0.75)  return("moderate")
    if (p >= 0.50)  return("weak")
    return("negligible")
  }

  # ---- 3. Narrative paragraph -----------------------------------------
  model_label <- if (is_linear) {
    if (has_re_b0) "hierarchical linear regression" else "linear regression"
  } else if (K == 1L) {
    "piecewise regression with a single breakpoint and logistic-smoothed transition"
  } else {
    sprintf("piecewise regression with %d breakpoints and logistic-smoothed transitions", K)
  }

  b0_sent <- if (is_linear) {
    sprintf("The estimated intercept was %s and the slope was %s.",
            .ci(s_fixed, "^b0_\\(Intercept\\)"), .ci(s_fixed, "^b1_\\(Intercept\\)"))
  } else {
    sprintf("The baseline intercept (at %s = omega_1) was %s and the pre-breakpoint slope was %s.",
            time, .ci(s_fixed, "^b0_\\(Intercept\\)"), .ci(s_fixed, "^b1_\\(Intercept\\)"))
  }

  bp_sents <- character(K)
  for (k in seq_len(K)) {
    om_ci  <- if (is_fixed_omega[k])
      sprintf("%.*f [fixed]", d, om_int[k])
    else
      .ci(s_fixed, sprintf("^omega%d_\\(Intercept\\)", k))

    del_ci <- .ci(s_fixed, sprintf("^delta%d_\\(Intercept\\)", k))

    rho_desc <- if (is_fixed_rho[k]) {
      sprintf("rho_%d was fixed at %.*f", k, d, rho_int[k])
    } else {
      rho_ci <- .ci(s_fixed, sprintf("^rho%d_\\(Intercept\\)", k))
      sharpness <- if (!is.na(rho_int[k])) {
        if      (rho_int[k] >= 20) "very sharp (near step-function)"
        else if (rho_int[k] >= 8)  "sharp"
        else if (rho_int[k] >= 3)  "moderately sharp"
        else if (rho_int[k] >= 1)  "gradual"
        else                       "very gradual (slow-onset)"
      } else "unknown sharpness"
      sprintf("rho_%d = %s (%s transition)", k, rho_ci, sharpness)
    }

    if (is_ss) {
      pip  <- pip_by_k[k]
      str  <- pip_strength(pip)
      bp_sents[k] <- sprintf(
        "Breakpoint %d had a posterior inclusion probability of %.3f (%s evidence for a structural change). Conditional on inclusion, its location was estimated at %s = %s, slope change delta_%d = %s, and %s.",
        k, pip, str, time, om_ci, k, del_ci, rho_desc
      )
    } else if (is_fixed_omega[k]) {
      slope_after <- if (!is.na(b1_int) && !is.na(del_int[k])) {
        sprintf("; asymptotic post-breakpoint slope = %.*f + %.*f = %.*f",
                d, b1_int, d, del_int[k], d, b1_int + del_int[k])
      } else ""
      bp_sents[k] <- sprintf(
        "Breakpoint %d was fixed at %s = %.3f. The slope change was %s%s. %s.",
        k, time, om_int[k], del_ci, slope_after, rho_desc
      )
    } else {
      slope_after <- if (!is.na(b1_int) && !is.na(del_int[k])) {
        sprintf("; asymptotic post-breakpoint slope = %.*f + %.*f = %.*f",
                d, b1_int, d, del_int[k], d, b1_int + del_int[k])
      } else ""
      bp_sents[k] <- sprintf(
        "Breakpoint %d was located at %s = %s%s. %s.",
        k, time, om_ci, slope_after, rho_desc
      )
    }
  }

  re_sent <- if (has_re_b0) {
    su_ci <- .ci(s_rpar, "^sigma_u$")
    sprintf("Between-group variability in the intercept was sigma_u = %s.", su_ci)
  } else ""

  sigma_sent <- sprintf("The residual standard deviation was sigma = %s.",
                        .ci(s_fixed, "^sigma$"))

  narrative_parts <- c(
    sprintf("Results are reported from a Bayesian %s fitted to %d observation%s%s.",
            model_label, n_obs, if (n_obs != 1L) "s" else "",
            if (has_re_b0) sprintf(" across %d groups", n_grp) else ""),
    b0_sent,
    bp_sents,
    if (nchar(re_sent)) re_sent else NULL,
    sigma_sent,
    conv_sent
  )
  narrative <- paste(narrative_parts, collapse = " ")

  # ---- 4. Fitted equation with numbers --------------------------------
  # Part A: general algebraic form (no numbers, with annotation)
  if (is_linear) {
    gen_form <- sprintf(
      "  %s_{ij} = b0 + b1 * %s + epsilon_{ij}",
      resp, time
    )
    if (has_re_b0)
      gen_form <- sprintf(
        "  %s_{ij} = (b0 + u_{j}) + b1 * %s + epsilon_{ij}  [u_{j} ~ N(0, sigma_u^2)]",
        resp, time
      )
  } else {
    re_str <- if (has_re_b0) " + u_{j}" else ""
    if (K == 1L) {
      gen_form <- sprintf(
        "  %s_{ij} = (b0%s) + b1*(tau - omega_1) + delta_1*(tau - omega_1)*logistic(rho_1*(tau - omega_1)) + epsilon_{ij}",
        resp, re_str
      )
    } else {
      gen_form <- sprintf(
        "  %s_{ij} = (b0%s) + b1*(tau - omega_1) + sum_{k=1}^{%d}[ delta_k*(tau - omega_k)*logistic(rho_k*(tau - omega_k)) ] + epsilon_{ij}",
        resp, re_str, K
      )
    }
    if (has_re_b0 && has_re_om)
      gen_form <- paste0(gen_form, "  [u_{j} ~ N(0,sigma_u^2); omega_{kj} ~ N(omega_k,sigma_re_om_k^2)]")
  }

  # Part B: substituted equation with posterior means
  # Build the b0 linear predictor string (expand all predictors)
  pv     <- object$pv
  b0_lp  <- .lp_str(pv$b0, .val(s_fixed, "^b0_"))

  # For b0 mean vector pull all b0 coefficients
  b0_means <- vapply(pv$b0$name, function(nm) {
    .val(s_fixed, paste0("^b0_", gsub("([().])", "\\\\\\1", nm), "$"))
  }, numeric(1L))
  b0_lp <- .lp_str(pv$b0, b0_means)

  b1_means <- vapply(pv$b1$name, function(nm) {
    .val(s_fixed, paste0("^b1_", gsub("([().])", "\\\\\\1", nm), "$"))
  }, numeric(1L))
  b1_lp <- .lp_str(pv$b1, b1_means)

  # Indent helper: align continuation lines under the first "="
  eq_lhs    <- sprintf("%s_hat =", resp)
  pad       <- paste0(rep(" ", nchar(eq_lhs)), collapse = "")

  if (is_linear) {
    sub_lines <- c(
      sprintf("  %s %s", eq_lhs, b0_lp),
      sprintf("  %s + %s * %s",      pad, b1_lp, time)
    )
  } else {
    # omega_1 intercept for centring b1
    om1_lp <- if (is_fixed_omega[1L]) {
      .fmt(om_int[1L])
    } else {
      om1_means <- vapply(pv$om[[1L]]$name, function(nm) {
        .val(s_fixed, paste0("^omega1_", gsub("([().])", "\\\\\\1", nm), "$"))
      }, numeric(1L))
      .lp_str(pv$om[[1L]], om1_means)
    }

    sub_lines <- c(
      sprintf("  %s %s", eq_lhs, b0_lp),
      sprintf("  %s + (%s) * (%s - %s)", pad, b1_lp, time, om1_lp)
    )

    for (k in seq_len(K)) {
      del_means <- vapply(pv$deltas[[k]]$name, function(nm) {
        .val(s_fixed, paste0("^delta", k, "_", gsub("([().])", "\\\\\\1", nm), "$"))
      }, numeric(1L))
      del_lp <- .lp_str(pv$deltas[[k]], del_means)

      om_lp <- if (is_fixed_omega[k]) {
        sprintf("%s [fixed]", .fmt(om_int[k]))
      } else {
        om_means <- vapply(pv$om[[k]]$name, function(nm) {
          .val(s_fixed, paste0("^omega", k, "_", gsub("([().])", "\\\\\\1", nm), "$"))
        }, numeric(1L))
        .lp_str(pv$om[[k]], om_means)
      }

      rho_lp <- if (is_fixed_rho[k]) {
        sprintf("%s [fixed]", .fmt(rho_int[k]))
      } else {
        rho_means <- vapply(pv$rho[[k]]$name, function(nm) {
          .val(s_fixed, paste0("^rho", k, "_", gsub("([().])", "\\\\\\1", nm), "$"))
        }, numeric(1L))
        .lp_str(pv$rho[[k]], rho_means)
      }

      d_str   <- sprintf("(%s)", del_lp)
      om_str  <- sprintf("(%s - %s)", time, om_lp)
      rho_str <- sprintf("logistic(%s * (%s - %s))", rho_lp, time, om_lp)
      sub_lines <- c(sub_lines,
        sprintf("  %s + %s * %s * %s", pad, d_str, om_str, rho_str)
      )
    }
  }

  sub_lines <- c(sub_lines,
    sprintf("  %s + epsilon_{ij}  [epsilon ~ N(0, sigma^2 = %.3f^2)]",
            pad, if (!is.na(sigma_v)) sigma_v else 0))

  if (!is_linear)
    sub_lines <- c(sub_lines,
      sprintf("  %s  where logistic(x) = 1/(1+exp(-x))", pad))

  if (any_cov)
    sub_lines <- c(sub_lines,
      "",
      "  NOTE: values shown are population-level posterior means.",
      "  For parameters with additional covariate effects, the full",
      "  linear predictors are shown above (e.g., 'a + b*var').",
      "  Individual-coefficient CIs are listed in PARAMETER ESTIMATES.")

  # ---- 5. Effective asymptotic slopes ---------------------------------
  slope_lines <- character(0L)
  if (!is_linear && !is.na(b1_int)) {
    slope_lines <- c(
      sprintf("  Pre-breakpoint asymptote  (as %s -> -inf relative to omega_1):  slope = %s",
              time, .fmt(b1_int))
    )
    cum_d <- 0
    for (k in seq_len(K)) {
      if (!is.na(del_int[k])) {
        cum_d <- cum_d + del_int[k]
        slope_lines <- c(slope_lines,
          sprintf("  Post-breakpoint %d asymptote (as %s -> +inf relative to omega_%d): slope = %s + %s = %s",
                  k, time, k, .fmt(b1_int), .fmt(cum_d), .fmt(b1_int + cum_d))
        )
      }
    }
    slope_lines <- c(slope_lines, "",
      "  [The logistic sigmoid creates a gradual transition. The asymptotic slopes",
      "   apply well beyond each breakpoint. The slope at any exact tau can be",
      "   computed by evaluating the derivative of the substituted equation above.]"
    )
  }

  # ---- 6. Per-breakpoint summary block --------------------------------
  bp_block <- character(0L)
  for (k in seq_len(K)) {
    hdr <- sprintf("  Breakpoint %d", k)
    bp_block <- c(bp_block, hdr, paste0("  ", paste0(rep("-", 40L), collapse = "")))

    # Omega
    if (is_fixed_omega[k]) {
      bp_block <- c(bp_block,
        sprintf("  Location (omega_%d)     : %s [fixed, not estimated]", k, .fmt(om_int[k])))
    } else {
      bp_block <- c(bp_block,
        sprintf("  Location (omega_%d)     : %s", k, .ci(s_fixed, sprintf("^omega%d_\\(Intercept\\)", k))))
      if (has_re_om)
        bp_block <- c(bp_block,
          sprintf("  RE timing SD           : %s",
                  .ci(s_rpar, sprintf("^sigma_re_omega%d$", k))))
    }

    # Rho
    if (is_fixed_rho[k]) {
      bp_block <- c(bp_block,
        sprintf("  Sharpness (rho_%d)      : %s [fixed]", k, .fmt(rho_int[k])))
    } else {
      bp_block <- c(bp_block,
        sprintf("  Sharpness (rho_%d)      : %s", k, .ci(s_fixed, sprintf("^rho%d_\\(Intercept\\)", k))))
    }

    # Delta (with PIP for SS)
    if (is_ss) {
      pip <- pip_by_k[k]
      bp_block <- c(bp_block,
        sprintf("  Slope change (delta_%d)  : %s", k, .ci(s_fixed, sprintf("^delta%d_\\(Intercept\\)", k))),
        sprintf("  Inclusion prob (PIP)    : %.3f  [%s evidence]", pip, pip_strength(pip))
      )
    } else {
      bp_block <- c(bp_block,
        sprintf("  Slope change (delta_%d)  : %s", k, .ci(s_fixed, sprintf("^delta%d_\\(Intercept\\)", k)))
      )
    }

    # Asymptotic slope after this breakpoint
    if (!is.na(b1_int) && !is.na(del_int[k])) {
      cum_d_k <- b1_int + sum(del_int[seq_len(k)])
      bp_block <- c(bp_block,
        sprintf("  Asymptotic slope after  : %s", .fmt(cum_d_k)))
    }
    bp_block <- c(bp_block, "")
  }

  # ---- 7. Full parameter table ----------------------------------------
  .fmt_tbl <- function(s) {
    if (is.null(s) || !nrow(s)) return(character(0L))
    s2 <- s
    nms <- names(s2)
    # round numeric columns
    num <- sapply(s2, is.numeric)
    s2[num] <- lapply(s2[num], round, d)
    paste0("  ", utils::capture.output(print(s2, row.names = FALSE)))
  }

  # For SS, separate gamma rows from coefficient rows
  if (is_ss && !is.null(s_fixed)) {
    gamma_rows  <- s_fixed[ grepl("^gamma_", s_fixed$variable), ]
    coeff_rows  <- s_fixed[!grepl("^gamma_", s_fixed$variable), ]
  } else {
    gamma_rows  <- NULL
    coeff_rows  <- s_fixed
  }

  # ---- 8. Random effects block ----------------------------------------
  re_block <- character(0L)
  if (!is.null(s_rpar) && nrow(s_rpar)) {
    re_block <- c(re_block,
      "  Variance / SD components (mean [95% CI]):",
      .fmt_tbl(s_rpar),
      ""
    )
  }
  if (!is.null(s_rval) && nrow(s_rval)) {
    u_v <- as.numeric(s_rval$mean)
    re_block <- c(re_block,
      sprintf("  Group-level intercept deviations u[j]  (%d groups):", nrow(s_rval)),
      sprintf("    Minimum  : %.*f", d, min(u_v)),
      sprintf("    Median   : %.*f", d, stats::median(u_v)),
      sprintf("    Maximum  : %.*f", d, max(u_v)),
      sprintf("    Empirical SD : %.*f  (compare to sigma_u above)", d, stats::sd(u_v)),
      "  Individual draws: summary(fit, effects = 'ran_vals')",
      ""
    )
  }

  # ---- 9. PIPs table for SS -------------------------------------------
  pip_block <- character(0L)
  if (is_ss && !is.null(gamma_rows) && nrow(gamma_rows)) {
    pip_block <- c(
      "  PIP = posterior mean of Bernoulli inclusion indicator (gamma_k).",
      "  Interpretation: PIP >= 0.95 strong, 0.75-0.95 moderate, < 0.50 weak.",
      "",
      "  [gamma means shown as PIPs; SD/CI reflect draw-to-draw variability",
      "   of the binary indicator, not coefficient uncertainty]",
      "",
      .fmt_tbl(gamma_rows)
    )
  }

  # ---- 10. Convergence table -------------------------------------------
  s_all <- tryCatch(summary(object, effects = "all"), error = function(e) NULL)
  conv_block <- character(0L)
  if (!is.null(s_all) && nrow(s_all)) {
    .cv <- function(col, label, fmt_fn, warn_fn) {
      if (!col %in% names(s_all)) return(NULL)
      v <- as.numeric(s_all[[col]]); v <- v[is.finite(v)]
      if (!length(v)) return(NULL)
      sprintf("  %-30s: %s  %s", label, fmt_fn(v),
              if (warn_fn(v)) "[WARNING]" else "[OK]")
    }
    conv_block <- c(
      .cv("Rhat",     "Max Rhat (threshold <= 1.05)",
          function(v) sprintf("%.3f", max(v)), function(v) max(v) > 1.05),
      .cv("Bulk_ESS", "Min bulk ESS (threshold >= 100)",
          function(v) as.character(as.integer(min(v))), function(v) min(v) < 100L),
      .cv("Tail_ESS", "Min tail ESS (threshold >= 100)",
          function(v) as.character(as.integer(min(v))), function(v) min(v) < 100L)
    )
    conv_block <- conv_block[!sapply(conv_block, is.null)]
  }
  conv_block <- c(conv_block,
    sprintf("  %-30s: %d  %s",
            "Divergent transitions", n_diverg,
            if (n_diverg > 0L) "[WARNING]" else "[OK]"),
    sprintf("  %-30s: %d chains x %d post-warmup draws = %d total",
            "Posterior draws", object$chains, object$iter - object$warmup, n_post)
  )

  # ---- Assemble report -----------------------------------------------
  .section <- function(title) {
    bar <- paste0(rep("=", 70L), collapse = "")
    c(bar, title, bar)
  }

  report <- c(
    .section("SMOOTHBP: RESULTS REPORT"),
    "",
    .section("NARRATIVE  (paste into manuscript Results section)"),
    "",
    strwrap(narrative, width = width),
    "",
    .section("FITTED MODEL  (posterior means substituted, 3 d.p.)"),
    "",
    "  -- General algebraic form --",
    gen_form,
    "",
    "  -- Numerical substitution (all coefficients are posterior means) --",
    sub_lines
  )

  if (length(slope_lines)) {
    report <- c(report,
      "",
      "  -- Effective asymptotic slopes --",
      slope_lines
    )
  }

  if (length(bp_block)) {
    report <- c(report,
      "",
      .section("BREAKPOINT SUMMARIES  (mean [95% credible interval])"),
      "",
      bp_block
    )
  }

  if (length(pip_block)) {
    report <- c(report,
      "",
      .section("POSTERIOR INCLUSION PROBABILITIES  (spike-and-slab)"),
      "",
      pip_block
    )
  }

  report <- c(report,
    "",
    .section("PARAMETER ESTIMATES  (mean, SD, 95% CI, Rhat, ESS)"),
    ""
  )
  if (!is.null(coeff_rows) && nrow(coeff_rows))
    report <- c(report, "  Fixed / population-level coefficients:", "", .fmt_tbl(coeff_rows))

  if (length(re_block)) {
    report <- c(report,
      "",
      .section("RANDOM EFFECTS"),
      "",
      re_block
    )
  }

  report <- c(report,
    "",
    .section("CONVERGENCE DIAGNOSTICS"),
    "",
    conv_block,
    paste0(rep("=", 70L), collapse = "")
  )

  out <- paste(report, collapse = "\n")
  cat(out, "\n")
  invisible(out)
}
