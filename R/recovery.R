#' Parameter recovery plot
#'
#' Compares posterior estimates against the known data-generating values stored
#' in \code{attr(dat, "true_params")} (as produced by
#' \code{\link{simulate_smoothbp}}).  For each matched parameter the plot shows
#' the posterior mean, a credible interval, and the true value, coloured by
#' whether the interval contains the truth.
#'
#' The function looks for population-level intercept parameters only.  If the
#' fitted model has covariates in a given component (e.g.
#' \code{omega = ~ 1 + group}) the function still extracts the
#' \code{(Intercept)} term for comparison against the scalar true value from
#' the simulation.
#'
#' @param fit  A \code{smoothbp_fit} object.
#' @param dat  The data frame used to fit \code{fit}, which must carry a
#'   \code{"true_params"} attribute (returned by \code{\link{simulate_smoothbp}}).
#' @param level Credible interval width.  Default \code{0.95}.
#'
#' @return A \code{ggplot} object.
#'
#' @examples
#' \dontrun{
#' dat <- simulate_smoothbp(n_subj = 20, n_obs = 8, seed = 42)
#' fit <- smoothbp(y ~ tau, b0 = ~ 1 + (1 | subject), data = dat,
#'                 priors = smoothbp_priors(omega = prior_normal(3, 2, lb = 0)),
#'                 chains = 4L, iter = 2000L, warmup = 1000L, seed = 42L)
#' recovery_plot(fit, dat)
#' }
#'
#' @export
recovery_plot <- function(fit, dat, level = 0.95) {
  if (!inherits(fit, "smoothbp_fit")) {
    stop("`fit` must be a smoothbp_fit object.")
  }
  tp <- attr(dat, "true_params")
  if (is.null(tp)) {
    stop("`dat` must have a `true_params` attribute.  ",
         "Use simulate_smoothbp() to generate data.")
  }
  stopifnot(level > 0, level < 1)

  # ---- Posterior summary at the requested level ----------------------------
  lo_p <- (1 - level) / 2
  hi_p <- 1 - lo_p

  s <- posterior::summarise_draws(
    fit$draws,
    mean,
    ~ posterior::quantile2(.x, probs = c(lo_p, hi_p))
  )
  # Column names: variable, mean, qXX, qYY  (posterior uses e.g. q2.5 / q97.5)
  q_cols   <- grep("^q[0-9]", names(s), value = TRUE)
  names(s)[names(s) == q_cols[1]] <- ".lo"
  names(s)[names(s) == q_cols[2]] <- ".hi"

  # ---- Map true_params names -> fit parameter names ------------------------
  # Always look for the (Intercept) term; also check the exact name for sigma.
  param_map <- c(
    b0      = "b0_(Intercept)",
    b1      = "b1_(Intercept)",
    delta   = "delta1_(Intercept)",
    omega   = "omega1_(Intercept)",
    rho     = "rho1_(Intercept)",
    sigma   = "sigma",
    sigma_u = "sigma_u"
  )

  # Pretty labels for the y-axis
  pretty_labels <- c(
    b0      = expression(b[0]),
    b1      = expression(b[1]),
    delta   = expression(delta[1]),
    omega   = expression(omega),
    rho     = expression(rho),
    sigma   = expression(sigma),
    sigma_u = expression(sigma[u])
  )

  # ---- Build comparison data frame ----------------------------------------
  scalar_tp <- tp[!names(tp) %in% c("u", "seed")]

  rows <- lapply(names(param_map), function(nm) {
    truth  <- scalar_tp[[nm]]
    pname  <- param_map[[nm]]
    if (is.null(truth) || !pname %in% s$variable) return(NULL)

    row <- s[s$variable == pname, ]
    data.frame(
      nm       = nm,
      pname    = pname,
      mean     = row$mean,
      lo       = row$.lo,
      hi       = row$.hi,
      truth    = truth,
      covered  = truth >= row$.lo & truth <= row$.hi,
      stringsAsFactors = FALSE
    )
  })
  cmp <- do.call(rbind, Filter(Negate(is.null), rows))

  if (nrow(cmp) == 0) {
    stop("No matching parameters found between `true_params` and the fitted model.")
  }

  # Order parameters in the conventional model order
  param_order <- intersect(names(param_map), cmp$nm)
  cmp$nm <- factor(cmp$nm, levels = rev(param_order))

  n_covered <- sum(cmp$covered)
  n_total   <- nrow(cmp)

  # ---- Build plot ----------------------------------------------------------
  ggplot2::ggplot(cmp, ggplot2::aes(y = nm)) +
    # Credible interval
    ggplot2::geom_segment(
      ggplot2::aes(x = lo, xend = hi, yend = nm, colour = covered),
      linewidth = 1.4, lineend = "round"
    ) +
    # Posterior mean
    ggplot2::geom_point(
      ggplot2::aes(x = mean, colour = covered),
      size = 3.5, shape = 19
    ) +
    # True value (black ×)
    ggplot2::geom_point(
      ggplot2::aes(x = truth),
      shape = 4, size = 4, stroke = 1.8,
      colour = "black"
    ) +
    ggplot2::scale_colour_manual(
      values = c("TRUE"  = "#2ca02c", "FALSE" = "#d62728"),
      labels = c("TRUE"  = "CI contains truth",
                 "FALSE" = "CI misses truth"),
      name = NULL
    ) +
    ggplot2::scale_y_discrete(
      labels = setNames(
        # Convert expression() labels to character for scale_y_discrete;
        # use deparse for the expression objects
        vapply(param_order, function(nm) {
          lbl <- pretty_labels[[nm]]
          if (is.null(lbl)) nm else deparse(lbl)
        }, character(1)),
        rev(param_order)
      )
    ) +
    ggplot2::labs(
      title = sprintf(
        "Parameter recovery  \u2014  %d%% posterior intervals",
        round(level * 100)
      ),
      subtitle = sprintf(
        "%d / %d intervals contain the true value  (\u2715 = truth, \u25cf = posterior mean)",
        n_covered, n_total
      ),
      x = "Value", y = NULL
    ) +
    ggplot2::theme(
      legend.position  = "bottom",
      plot.subtitle    = ggplot2::element_text(
        colour = if (n_covered == n_total) "grey40" else "#cc3333",
        size = 9
      )
    )
}
