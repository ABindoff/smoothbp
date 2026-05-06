#' Trace and density plots for a smoothbp_fit
#'
#' A thin wrapper around \code{\link{trace_plot}} for the standard
#' \code{plot()} interface.
#'
#' @param x   A \code{smoothbp_fit} object.
#' @param type One of \code{"trace"} (default), \code{"density"}, or
#'   \code{"both"}.
#' @param pars Character vector of parameter names.  Defaults to all
#'   non-random-effect parameters.
#' @param ...  Passed to \code{\link{trace_plot}}.
#' @return A \code{ggplot} object, or a named list of two when
#'   \code{type = "both"}.
#' @export
plot.smoothbp_fit <- function(x, type = "trace", pars = NULL, ...) {
  if (is.null(pars)) {
    all_pars <- posterior::variables(x$draws)
    pars <- all_pars[!grepl("^u\\[", all_pars)]
  }
  trace_plot(x, pars = pars, type = type, ...)
}

# ---------------------------------------------------------------------------
# Internal helper: draws_df -> long data frame
# ---------------------------------------------------------------------------

.draws_to_long <- function(draws_obj, pars) {
  draws_df   <- posterior::as_draws_df(draws_obj[, , pars, drop = FALSE])
  param_cols <- setdiff(names(draws_df), c(".chain", ".iteration", ".draw"))
  do.call(rbind, lapply(param_cols, function(p) {
    data.frame(
      parameter = p,
      iteration = draws_df$.iteration,
      chain     = factor(draws_df$.chain),
      value     = draws_df[[p]],
      stringsAsFactors = FALSE
    )
  }))
}

#' Trace plots with automatic poor-mixing highlighting
#'
#' Produces per-parameter trace plots from a \code{smoothbp_fit} object.
#' Parameters with \eqn{\hat{R} > 1.05} are flagged with a light-red
#' background and their panel labels include the \eqn{\hat{R}} value and a
#' warning symbol.  Parameters with low bulk-ESS (< 100) are further annotated.
#'
#' @param fit  A \code{smoothbp_fit} object.
#' @param pars Character vector of parameter names to include.  Defaults to
#'   all non-random-effect parameters.
#' @param type One of \code{"trace"} (default), \code{"density"}, or
#'   \code{"both"}.
#' @param rhat_thresh Rhat threshold above which a parameter is flagged as
#'   poorly mixing.  Default \code{1.05}.
#' @param ess_thresh   Bulk-ESS threshold below which a parameter is flagged.
#'   Default \code{100}.
#'
#' @return A \code{ggplot} object (or a named list of two when
#'   \code{type = "both"}).
#' @export
trace_plot <- function(
    fit,
    pars        = NULL,
    type        = "trace",
    rhat_thresh = 1.05,
    ess_thresh  = 100
) {
  if (!inherits(fit, "smoothbp_fit")) {
    stop("`fit` must be a smoothbp_fit object.")
  }
  if (!type %in% c("trace", "density", "both")) {
    stop('`type` must be one of "trace", "density", or "both".')
  }

  if (is.null(pars)) {
    all_pars <- posterior::variables(fit$draws)
    pars <- all_pars[!grepl("^u\\[", all_pars)]
  }

  # ---- Compute mixing diagnostics -----------------------------------------
  diag_df <- .mixing_diagnostics(fit, pars, rhat_thresh, ess_thresh)

  # ---- Build long draws data frame ----------------------------------------
  long <- .draws_to_long(fit$draws, pars)

  # Attach diagnostics and relabel parameters
  long <- merge(long, diag_df[, c("parameter", "label", "flag")],
                by = "parameter", all.x = TRUE)

  # Use labelled parameter as the facet variable
  long$param_label <- long$label

  # Build background data for flagged parameters (only for trace)
  bad_params <- diag_df$parameter[diag_df$flag]

  if (type %in% c("trace", "both")) {
    p_trace <- .build_trace(long, bad_params, rhat_thresh, ess_thresh)
  }

  if (type %in% c("density", "both")) {
    p_dens <- .build_density(long)
  }

  # Print mixing summary to console if any flags
  n_bad <- sum(diag_df$flag)
  if (n_bad > 0) {
    bad_names <- diag_df$parameter[diag_df$flag]
    bad_rhats <- diag_df$rhat[diag_df$flag]
    message(sprintf(
      "%d parameter(s) flagged (Rhat > %.2f or ESS < %d): %s",
      n_bad, rhat_thresh, ess_thresh,
      paste(sprintf("%s (%.3f)", bad_names, bad_rhats), collapse = ", ")
    ))
  }

  if (type == "trace")   return(p_trace)
  if (type == "density") return(p_dens)
  list(trace = p_trace, density = p_dens)
}

# ---------------------------------------------------------------------------
# Internal: compute per-parameter Rhat and ESS, build flag + label
# ---------------------------------------------------------------------------

.mixing_diagnostics <- function(fit, pars, rhat_thresh, ess_thresh) {
  # Compute Rhat and ESS per parameter, guarding against empty results
  draws_sub <- fit$draws[, , pars, drop = FALSE]
  
  rhats <- vapply(pars, function(p) {
    tryCatch(posterior::rhat(draws_sub[, , p, drop = FALSE]), error = function(e) NA_real_)
  }, numeric(1))
  
  ess <- vapply(pars, function(p) {
    tryCatch(posterior::ess_bulk(draws_sub[, , p, drop = FALSE]), error = function(e) NA_real_)
  }, numeric(1))
  
  names(rhats) <- pars
  names(ess)   <- pars

  flag <- !is.na(rhats) & (rhats > rhat_thresh | (!is.na(ess) & ess < ess_thresh))

  # Build human-readable panel labels
  labels <- vapply(pars, function(p) {
    r <- rhats[p]
    e <- ess[p]
    rhat_str <- if (!is.na(r)) sprintf("Rhat=%.3f", r) else "Rhat=NA"
    ess_str  <- if (!is.na(e)) sprintf("ESS=%d", round(e)) else "ESS=NA"
    warn <- if (!is.na(r) && (r > rhat_thresh || (!is.na(e) && e < ess_thresh))) " \u26a0" else ""
    sprintf("%s\n%s  %s%s", p, rhat_str, ess_str, warn)
  }, character(1))

  data.frame(
    parameter = pars,
    rhat      = rhats,
    ess       = ess,
    flag      = flag,
    label     = labels,
    stringsAsFactors = FALSE,
    row.names = NULL
  )
}

# ---------------------------------------------------------------------------
# Internal: build trace ggplot
# ---------------------------------------------------------------------------

.build_trace <- function(long, bad_params, rhat_thresh, ess_thresh) {
  # Background rectangles for flagged parameters (very light red wash)
  has_bad <- length(bad_params) > 0
  if (has_bad) {
    bg_data <- unique(long[long$parameter %in% bad_params,
                           c("param_label", "iteration")])
    bg_data <- do.call(rbind, lapply(
      unique(bg_data$param_label), function(lbl) {
        iters <- bg_data$iteration[bg_data$param_label == lbl]
        data.frame(
          param_label = lbl,
          xmin = min(iters), xmax = max(iters),
          ymin = -Inf,        ymax = Inf,
          stringsAsFactors = FALSE
        )
      }
    ))
  }

  p <- ggplot2::ggplot(long,
         ggplot2::aes(x = iteration, y = value, colour = chain)) +
    ggplot2::geom_line(alpha = 0.7, linewidth = 0.3) +
    ggplot2::facet_wrap(~ param_label, scales = "free_y") +
    ggplot2::labs(
      title    = "Trace plots",
      subtitle = if (has_bad)
        sprintf("Parameters flagged (\u26a0) have Rhat > %.2f or ESS < %d",
                rhat_thresh, ess_thresh)
      else
        "All parameters appear well-mixed",
      x = "Post-warmup iteration",
      y = "Value"
    ) +
    ggplot2::theme(
      legend.position  = "bottom",
      strip.text       = ggplot2::element_text(size = 7.5, lineheight = 1.1),
      plot.subtitle    = ggplot2::element_text(
        colour = if (has_bad) "#cc3333" else "grey40", size = 9
      )
    )

  # Overlay red background on bad-mixing panels
  if (has_bad) {
    p <- p + ggplot2::geom_rect(
      data        = bg_data,
      ggplot2::aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax),
      fill        = "#ff4444",
      alpha       = 0.07,
      inherit.aes = FALSE
    )
    # Re-add lines on top of background (layer ordering)
    p <- p + ggplot2::geom_line(alpha = 0.7, linewidth = 0.3)
  }

  p
}

# ---------------------------------------------------------------------------
# Internal: build density ggplot
# ---------------------------------------------------------------------------

.build_density <- function(long) {
  ggplot2::ggplot(long,
    ggplot2::aes(x = value, colour = chain, fill = chain)) +
    ggplot2::geom_density(alpha = 0.2) +
    ggplot2::facet_wrap(~ param_label, scales = "free") +
    ggplot2::labs(
      title = "Posterior densities",
      x = "Value", y = "Density"
    ) +
    ggplot2::theme(
      legend.position = "bottom",
      strip.text      = ggplot2::element_text(size = 7.5, lineheight = 1.1)
    )
}
