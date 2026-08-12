# =============================================================================
# sbc_ecdf_bands.R   (v4: fully valid SBC)
#
# Simulation-based calibration for the random change-point model, assessed with
# simultaneous ECDF confidence bands (Sailynoja, Burkner & Vehtari 2022).
#
# WHY v4: earlier versions (inherited from part3_validation.R) pinned b0, b1,
# sigma_y (and rho, in the estimated-rho run) at constants in the data
# generation while the FIT estimated them under priors. That is not a valid SBC:
# the theorem requires EVERY estimated parameter's truth to be drawn from its
# prior. Because b0 is strongly correlated with the change-point location omega,
# pinning it can push the ranks of omega/sigma_re/delta off uniform even when the
# sampler is exact. v4 draws b0, b1, sigma_y and (when estimated) rho from
# proper, moderate priors and fits with those identical priors, so generation and
# fitting match on every estimated parameter.
#
# Verified against the Rust backend (run_mcmc_re):
#   * the sigma_re_om prior IS plumbed through (not the hardcoded InvGamma(1,1),
#     which only appears in the non-random-effect entry points);
#   * InvGamma convention: backend draws precision ~ Gamma(shape, 1/scale), i.e.
#     rate = scale, so a variance truth is drawn as 1/rgamma(1, shape, rate=scale)
#     and sigma = sqrt(that). Encoded in draw_sigma() below.
#
# Runs and figures (manuscripts/figures/):
#   figure_sbc_ecdf.png               rho estimated, all fits
#   figure_sbc_ecdf_delta_strata.png  delta by |delta_true|
#   figure_sbc_ecdf_converged.png     rho estimated, converged subset
#   figure_sbc_ecdf_fixedrho.png      rho fixed at truth (the clean calibration test)
#
# Run from package root AFTER rextendr::document() + devtools::load_all():
#   source("tools/sbc_ecdf_bands.R")
# =============================================================================

suppressPackageStartupMessages({
  library(devtools); library(posterior); library(ggplot2)
  library(dplyr); library(tidyr); library(patchwork)
})
# Normally run against the working tree via load_all(). Set
# SMOOTHBP_USE_INSTALLED=1 to validate the built package instead, which is what
# the release SBC should exercise.
if (nzchar(Sys.getenv("SMOOTHBP_USE_INSTALLED"))) {
  library(smoothbp)
  cat(sprintf("[SBC] using INSTALLED smoothbp %s from %s
",
              as.character(utils::packageVersion("smoothbp")), find.package("smoothbp")))
} else {
  suppressMessages(devtools::load_all(quiet = TRUE))
  cat("[SBC] using working tree via devtools::load_all()
")
}

OUT_DIR <- file.path("tools", "part3_results")
FIG_DIR <- file.path("manuscripts", "figures")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(FIG_DIR, showWarnings = FALSE, recursive = TRUE)

# ---- Config -----------------------------------------------------------------
SBC_REPS  <- 500L
SBC_J     <- 6L;  SBC_N    <- 18L
SBC_ITER  <- 3000L; SBC_WARM <- 500L; SBC_CHAINS <- 2L
SBC_L     <- 100L
TA_SBC    <- 0.99
SBC_TAU_LO <- -5; SBC_TAU_HI <- 11
DELTA_STRAT <- 0.75
ESS_MIN     <- 200L
SBC_SEED    <- 424242L
RHO_FIX     <- 3.0      # constant rho used (gen AND fit) in the fixed-rho run

# Moderate proper priors. EVERY estimated parameter is drawn from exactly these
# and the fit is given exactly these. (a, b) for invgamma is (shape, scale) with
# E[variance] = scale / (shape - 1).
PR_B0     <- c(mean = 5,  sd = 1)
PR_B1     <- c(mean = 0,  sd = 0.3)
PR_DELTA  <- c(mean = 0,  sd = 1.5)
PR_OMEGA  <- c(mean = 3,  sd = 1.0, lb = 0, ub = 10)
PR_RHO    <- c(mean = 4,  sd = 1.0, lb = 0)          # estimated-rho run only
PR_SIGMA   <- c(shape = 10, scale = 1.44)            # E[var]=0.16 -> sigma~0.4
PR_SIGMARE <- c(shape = 5,  scale = 4)               # E[var]=1.0  -> sigma_re~1
PR_SIGMAU  <- c(shape = 5,  scale = 4)               # random-intercept SD prior

SBC_PRIORS <- smoothbp_priors(
  b0          = prior_normal(PR_B0["mean"], PR_B0["sd"]),
  b1          = prior_normal(PR_B1["mean"], PR_B1["sd"]),
  deltas      = prior_normal(PR_DELTA["mean"], PR_DELTA["sd"]),
  omega       = prior_normal(PR_OMEGA["mean"], PR_OMEGA["sd"], lb = PR_OMEGA["lb"], ub = PR_OMEGA["ub"]),
  rho         = prior_normal(PR_RHO["mean"], PR_RHO["sd"], lb = PR_RHO["lb"]),
  sigma       = prior_invgamma(PR_SIGMA["shape"], PR_SIGMA["scale"]),
  sigma_re_om = prior_invgamma(PR_SIGMARE["shape"], PR_SIGMARE["scale"]),
  sigma_u     = prior_invgamma(PR_SIGMAU["shape"], PR_SIGMAU["scale"])
)

# ---- Prior draws (must match the fit priors exactly) ------------------------
rtrunc <- function(mean, sd, lb = -Inf, ub = Inf) {
  repeat { x <- rnorm(1, mean, sd); if (x > lb && x < ub) return(x) }
}
# variance ~ InvGamma(shape, scale) under the backend's rate=scale convention
draw_sigma <- function(shape, scale) sqrt(1 / rgamma(1, shape = shape, rate = scale))

# ---- Helpers (bands / panels) -----------------------------------------------
ess_of <- function(draws, v) {
  out <- tryCatch(
    posterior::summarise_draws(posterior::subset_draws(draws, variable = v), "ess_basic")$ess_basic,
    error = function(e) NA_real_)
  if (length(out) == 0) NA_real_ else out[1]
}
# The band is itself a Monte Carlo object. At M = 4000 its edge moved enough
# between runs to flip the PASS/FAIL verdict for a parameter sitting near the
# boundary, which makes the verdict a property of the band's seed rather than of
# the sampler. M is therefore raised and bands are memoised by N, since every
# parameter within a subset shares one.
.band_cache <- new.env(parent = emptyenv())
sim_band <- function(N, L, conf = 0.95, M = 20000L, grid = NULL) {
  if (N < 1) return(NULL)
  key <- paste(N, L, conf, M, sep = "_")
  if (!is.null(grid)) key <- paste(key, "custom", sep = "_")
  if (is.null(grid) && exists(key, envir = .band_cache)) return(get(key, envir = .band_cache))
  band <- .sim_band_raw(N, L, conf, M, grid)
  if (is.null(grid)) assign(key, band, envir = .band_cache)
  band
}
.sim_band_raw <- function(N, L, conf = 0.95, M = 20000L, grid = NULL) {
  if (is.null(grid)) grid <- (1:L) / (L + 1)
  thr <- grid * (L + 1) - 1
  null_counts <- function() {
    R <- matrix(sample(0:L, N * M, replace = TRUE), nrow = M)
    vapply(thr, function(t) rowSums(R <= t), numeric(M))
  }
  cal <- null_counts(); ev <- null_counts()
  cover_at <- function(gamma) {
    lo <- apply(cal, 2, quantile, probs = gamma / 2,     type = 1)
    hi <- apply(cal, 2, quantile, probs = 1 - gamma / 2, type = 1)
    inside <- rowMeans((ev >= matrix(lo, nrow(ev), length(lo), byrow = TRUE)) &
                       (ev <= matrix(hi, nrow(ev), length(hi), byrow = TRUE))) == 1
    list(cov = mean(inside), lo = lo, hi = hi)
  }
  g_lo <- 1e-4; g_hi <- 0.5
  for (it in 1:40) { g <- (g_lo + g_hi) / 2; if (cover_at(g)$cov >= conf) g_lo <- g else g_hi <- g }
  b <- cover_at(g_lo)
  data.frame(p = grid, lo = b$lo / N, hi = b$hi / N, gamma = g_lo)
}
ecdf_diff_df <- function(ranks, L, band) {
  ranks <- ranks[!is.na(ranks)]
  thr <- band$p * (L + 1) - 1
  obs <- vapply(thr, function(t) mean(ranks <= t), numeric(1))
  data.frame(p = band$p, diff = obs - band$p, lo = band$lo - band$p, hi = band$hi - band$p)
}
ecdf_panel <- function(ranks, L, title, conf = 0.95) {
  N <- sum(!is.na(ranks)); band <- sim_band(N, L, conf = conf)
  if (is.null(band)) return(list(plot = NULL, pass = NA, gamma = NA, N = N))
  dd <- ecdf_diff_df(ranks, L, band)
  inband <- all(dd$diff >= dd$lo & dd$diff <= dd$hi)
  i <- which.max(abs(dd$diff))
  max_dev <- abs(dd$diff[i])
  band_at_max <- max(abs(c(dd$lo[i], dd$hi[i])))
  p <- ggplot(dd, aes(p)) +
    geom_ribbon(aes(ymin = lo, ymax = hi), fill = "grey80") +
    geom_hline(yintercept = 0, linetype = 2, colour = "grey50") +
    geom_step(aes(y = diff), linewidth = 0.6) +
    labs(x = "Fractional rank", y = "ECDF - uniform", title = parse(text = title)) +
    theme_minimal(base_size = 11)
  list(plot = p, pass = inband, gamma = band$gamma[1], N = N,
       max_dev = max_dev, band_at_max = band_at_max)
}
# Every estimated parameter goes on the figure and into the table.
#
# Earlier versions plotted only omega, sigma_re and delta and sent b0, b1 and
# sigma_y through a console-only helper that wrote nothing to disk.  That is how
# a systematic sigma_y failure (mean fractional rank 0.39, roughly twice the band
# width) went unnoticed for several releases: reporting only the parameters that
# pass is exactly what a referee who knows SBC will look for.
SBC_PARS <- c(
  rank_omega = "omega",
  rank_sigma = "sigma[re]",
  rank_delta = "delta",
  rank_b0    = "b[0]",
  rank_b1    = "b[1]",
  rank_sigy  = "sigma[y]",
  rank_rho   = "rho"
)
SBC_ESS <- c(rank_omega = "ess_omega", rank_sigma = "ess_sigma", rank_delta = "ess_delta")

report_combined <- function(df, tag, fig_path, table_path = NULL, filt = NULL,
                            pars = SBC_PARS) {
  plots <- list(); rows <- list()
  for (col in names(pars)) {
    if (!col %in% names(df) || all(is.na(df[[col]]))) next
    d <- df
    if (!is.null(filt)) {
      # Filter on this parameter's own ESS where we have it, else on the
      # change-point ESS, so the subset is stated rather than implied.
      # SBC_ESS is a named vector, so `[[` on an absent name errors rather
      # than returning NULL; test membership explicitly.
      ec <- if (col %in% names(SBC_ESS)) SBC_ESS[[col]] else "ess_omega"
      d <- df[df$n_div == 0 & !is.na(df[[ec]]) & df[[ec]] >= filt, ]
    }
    pn <- ecdf_panel(d[[col]], SBC_L, pars[[col]])
    plots[[col]] <- pn$plot
    r <- d[[col]][!is.na(d[[col]])]
    rows[[col]] <- data.frame(
      subset = tag,
      parameter = gsub("\\[|\\]", "", pars[[col]]),
      N = pn$N,
      result = ifelse(is.na(pn$pass), "n/a", ifelse(pn$pass, "PASS", "FAIL")),
      mean_frac_rank = if (length(r)) round(mean(r) / SBC_L, 4) else NA_real_,
      # Report the magnitude alongside the verdict: a deviation close to the
      # band edge is a borderline result and should not read as a clean pass
      # or a clean failure.
      max_dev = round(pn$max_dev, 4),
      band = round(pn$band_at_max, 4),
      dev_over_band = round(pn$max_dev / pn$band_at_max, 3),
      gamma = ifelse(is.na(pn$gamma), NA_real_, round(pn$gamma, 5))
    )
  }
  tab <- do.call(rbind, rows)
  cat(sprintf("\n[SBC-ECDF] %s:\n", tag))
  print(tab[, c("parameter", "N", "result", "mean_frac_rank",
                "max_dev", "band", "dev_over_band")], row.names = FALSE)

  plots <- Filter(Negate(is.null), plots)
  if (length(plots)) {
    ggsave(fig_path,
           patchwork::wrap_plots(plots, nrow = ceiling(length(plots) / 3)),
           width = 11, height = 3.6 * ceiling(length(plots) / 3), dpi = 200)
    cat(sprintf("[SBC-ECDF] figure -> %s\n", fig_path))
  }
  if (!is.null(table_path)) {
    write.csv(tab, table_path, row.names = FALSE)
    cat(sprintf("[SBC-ECDF] table  -> %s\n", table_path))
  }
  invisible(tab)
}

# Record the regime the SBC actually validates. SBC certifies the algorithm under
# the priors and design that were run, not the package defaults, so the manuscript
# needs these numbers stated rather than left to be inferred from the figure.
write_regime <- function(df, label, path, rho_fixed) {
  regime <- data.frame(
    field = c("label", "replicates", "groups_J", "obs_per_group_N", "total_obs",
              "chains", "iter", "warmup", "thinned_draws_L", "target_accept",
              "tau_range", "rho", "prior_b0", "prior_b1", "prior_delta",
              "prior_omega", "prior_sigma_y", "prior_sigma_re",
              "fits_completed", "fits_with_divergences", "divergence_rate"),
    value = c(
      label, SBC_REPS, SBC_J, SBC_N, SBC_J * SBC_N,
      SBC_CHAINS, SBC_ITER, SBC_WARM, SBC_L, TA_SBC,
      sprintf("[%g, %g]", SBC_TAU_LO, SBC_TAU_HI),
      if (is.null(rho_fixed)) sprintf("estimated, N(%g, %g) truncated at %g",
                                      PR_RHO["mean"], PR_RHO["sd"], PR_RHO["lb"])
      else sprintf("fixed at %g (generation and fit)", rho_fixed),
      sprintf("N(%g, %g)", PR_B0["mean"], PR_B0["sd"]),
      sprintf("N(%g, %g)", PR_B1["mean"], PR_B1["sd"]),
      sprintf("N(%g, %g)", PR_DELTA["mean"], PR_DELTA["sd"]),
      sprintf("N(%g, %g) on [%g, %g]", PR_OMEGA["mean"], PR_OMEGA["sd"],
              PR_OMEGA["lb"], PR_OMEGA["ub"]),
      sprintf("variance ~ InvGamma(shape=%g, scale=%g)", PR_SIGMA["shape"], PR_SIGMA["scale"]),
      sprintf("variance ~ InvGamma(shape=%g, scale=%g)", PR_SIGMARE["shape"], PR_SIGMARE["scale"]),
      nrow(df), sum(df$n_div > 0), round(mean(df$n_div > 0), 4)
    ), stringsAsFactors = FALSE)
  write.csv(regime, path, row.names = FALSE)
  cat(sprintf("[SBC-ECDF] regime -> %s\n", path))
  invisible(regime)
}

# ---- SBC runner: fully valid generation -------------------------------------
run_sbc <- function(rho_fixed = NULL, label = "rho estimated", reparameterise = "none",
                    cache = NULL) {
  # Each arm is 500 fits. Set SBC_REUSE=1 to pick up arms that already
  # completed, so a failure in the reporting code downstream does not cost
  # another hour of sampling. Unset (the default) always refits.
  #
  # Reuse is refused unless the saved ranks carry the version that produced
  # them and it matches the version now loaded. Silently serving ranks from a
  # different build is worse than no cache at all: it is exactly how a
  # calibration table comes to describe a sampler nobody is running any more.
  this_version <- as.character(utils::packageVersion("smoothbp"))
  if (!is.null(cache) && nzchar(Sys.getenv("SBC_REUSE")) && file.exists(cache)) {
    df <- read.csv(cache, check.names = FALSE)
    saved <- if ("smoothbp_version" %in% names(df)) unique(df$smoothbp_version) else NA_character_
    if (length(saved) == 1L && !is.na(saved) && identical(saved, this_version)) {
      cat(sprintf("\n=== SBC run: %s (REUSING %d saved replicates, smoothbp %s) ===\n",
                  label, nrow(df), saved))
      return(df)
    }
    cat(sprintf("\n[SBC] ignoring cache %s: saved under smoothbp %s, now running %s\n",
                cache, if (is.na(saved[1])) "an unrecorded version" else saved[1], this_version))
  }
  cat(sprintf("\n=== SBC run: %s ===\n", label))
  set.seed(SBC_SEED)                              # identical datasets across runs
  subj <- rep(seq_len(SBC_J), each = SBC_N)
  tau  <- rep(seq(SBC_TAU_LO, SBC_TAU_HI, length.out = SBC_N), times = SBC_J)
  res <- vector("list", SBC_REPS)
  for (r in seq_len(SBC_REPS)) {
    # ---- draw EVERY estimated parameter from its prior ----
    b0    <- rnorm(1, PR_B0["mean"], PR_B0["sd"])
    b1    <- rnorm(1, PR_B1["mean"], PR_B1["sd"])
    delta <- rnorm(1, PR_DELTA["mean"], PR_DELTA["sd"])
    omega <- rtrunc(PR_OMEGA["mean"], PR_OMEGA["sd"], PR_OMEGA["lb"], PR_OMEGA["ub"])
    rho   <- if (is.null(rho_fixed)) rtrunc(PR_RHO["mean"], PR_RHO["sd"], PR_RHO["lb"]) else rho_fixed
    sigre <- draw_sigma(PR_SIGMARE["shape"], PR_SIGMARE["scale"])
    sigy  <- draw_sigma(PR_SIGMA["shape"],   PR_SIGMA["scale"])
    u_j   <- rnorm(SBC_J, 0, sigre)
    om_j  <- omega + u_j[subj]
    d     <- tau - om_j
    mu    <- b0 + b1 * d + delta * d * plogis(rho * d)
    y     <- mu + rnorm(SBC_J * SBC_N, 0, sigy)
    dat   <- data.frame(subject = factor(subj), tau = tau, y = y)

    args <- list(formula = y ~ tau, b0 = ~ 1, b1 = ~ 1,
                 omega = list(~ (1 | subject)), deltas = list(~ 1),
                 data = dat, priors = SBC_PRIORS, chains = SBC_CHAINS,
                 iter = SBC_ITER, warmup = SBC_WARM, seed = 5000L + r,
                 target_accept = TA_SBC, reparameterise = reparameterise,
                 .verbose = FALSE)
    if (!is.null(rho_fixed)) args$rho <- list(fixed(rho_fixed))
    fit <- tryCatch(do.call(smoothbp, args), error = function(e) NULL)
    if (is.null(fit)) next

    dr <- posterior::as_draws_df(fit$draws); npost <- nrow(dr)
    keep <- round(seq(1, npost, length.out = SBC_L))
    rk <- function(col, truth) if (col %in% names(dr)) sum(dr[[col]][keep] < truth, na.rm = TRUE) else NA_integer_
    res[[r]] <- data.frame(
      smoothbp_version = this_version,
      rep = r, b0_true = b0, b1_true = b1, del_true = delta, abs_delta = abs(delta),
      om_true = omega, sig_re_true = sigre, sig_y_true = sigy, rho_true = rho,
      rank_omega = rk("omega1_(Intercept)", omega),
      rank_sigma = rk("sigma_re_omega1",    sigre),
      rank_delta = rk("delta1_(Intercept)", delta),
      rank_b0    = rk("b0_(Intercept)",     b0),
      rank_b1    = rk("b1_(Intercept)",     b1),
      rank_sigy  = rk("sigma",              sigy),
      rank_rho   = if (is.null(rho_fixed)) rk("rho1_(Intercept)", rho) else NA_integer_,
      ess_omega = ess_of(fit$draws, "omega1_(Intercept)"),
      ess_sigma = ess_of(fit$draws, "sigma_re_omega1"),
      ess_delta = ess_of(fit$draws, "delta1_(Intercept)"),
      n_div = fit$n_divergent)
    if (r %% 40L == 0L) cat(sprintf("  %s %d/%d\n", label, r, SBC_REPS))
  }
  dplyr::bind_rows(res)
}
ess_report <- function(df, label) {
  cat(sprintf("\n[diagnostic] %s: divergent fits %d / %d; ESS (pre-thin):\n",
              label, sum(df$n_div > 0), nrow(df)))
  for (e in c("ess_omega", "ess_sigma", "ess_delta"))
    cat(sprintf("  %-10s median=%.0f  min=%.0f\n",
                sub("ess_", "", e), median(df[[e]], na.rm = TRUE), min(df[[e]], na.rm = TRUE)))
}

# =============================================================================
# RUN 1: rho estimated
# =============================================================================
sbc <- run_sbc(rho_fixed = NULL, label = "rho estimated",
               cache = file.path(OUT_DIR, "sbc_ranks_ecdf.csv"))
write.csv(sbc, file.path(OUT_DIR, "sbc_ranks_ecdf.csv"), row.names = FALSE)
ess_report(sbc, "rho estimated, all fits")
write_regime(sbc, "rho estimated", file.path(OUT_DIR, "sbc_regime_rhoest.csv"), NULL)
report_combined(sbc, "rho estimated, ALL fits", file.path(FIG_DIR, "figure_sbc_ecdf.png"),
                table_path = file.path(OUT_DIR, "sbc_table_rhoest_all.csv"))

pn_lo <- ecdf_panel(sbc$rank_delta[sbc$abs_delta <  DELTA_STRAT], SBC_L, sprintf("delta:~~abs(delta) < %.2f", DELTA_STRAT))
pn_hi <- ecdf_panel(sbc$rank_delta[sbc$abs_delta >= DELTA_STRAT], SBC_L, sprintf("delta:~~abs(delta) >= %.2f", DELTA_STRAT))
cat(sprintf("\n[SBC-ECDF] delta stratified: weak %s (N=%d) | clear %s (N=%d)\n",
            ifelse(pn_lo$pass, "PASS", "FAIL"), pn_lo$N, ifelse(pn_hi$pass, "PASS", "FAIL"), pn_hi$N))
ggsave(file.path(FIG_DIR, "figure_sbc_ecdf_delta_strata.png"),
       pn_lo$plot + pn_hi$plot + patchwork::plot_layout(nrow = 1), width = 8, height = 3.6, dpi = 200)

report_combined(sbc, sprintf("rho estimated, CONVERGED subset (n_div==0 & ESS>=%d)", ESS_MIN),
                file.path(FIG_DIR, "figure_sbc_ecdf_converged.png"),
                table_path = file.path(OUT_DIR, "sbc_table_rhoest_converged.csv"),
                filt = ESS_MIN)

# =============================================================================
# RUN 2: rho FIXED at truth (the clean calibration test; same datasets)
# =============================================================================
sbc_fix <- run_sbc(rho_fixed = RHO_FIX, label = "rho fixed",
                   cache = file.path(OUT_DIR, "sbc_ranks_fixedrho.csv"))
write.csv(sbc_fix, file.path(OUT_DIR, "sbc_ranks_fixedrho.csv"), row.names = FALSE)
ess_report(sbc_fix, "rho fixed at truth")
write_regime(sbc_fix, "rho fixed at truth", file.path(OUT_DIR, "sbc_regime_fixedrho.csv"), RHO_FIX)
report_combined(sbc_fix, "rho FIXED at truth, ALL fits",
                file.path(FIG_DIR, "figure_sbc_ecdf_fixedrho.png"),
                table_path = file.path(OUT_DIR, "sbc_table_fixedrho_all.csv"))
report_combined(sbc_fix, sprintf("rho FIXED, CONVERGED subset (n_div==0 & ESS>=%d)", ESS_MIN),
                file.path(FIG_DIR, "figure_sbc_ecdf_fixedrho_converged.png"),
                table_path = file.path(OUT_DIR, "sbc_table_fixedrho_converged.csv"),
                filt = ESS_MIN)

# Interval coverage, derived from the same ranks that produce the figure, so the
# manuscript's coverage table and its SBC figure cannot drift apart again.
coverage_table <- function(df, tag, path, pars = SBC_PARS) {
  cov <- function(r, lo, hi) mean(r >= lo & r <= hi, na.rm = TRUE)
  rows <- lapply(names(pars), function(col) {
    if (!col %in% names(df) || all(is.na(df[[col]]))) return(NULL)
    r <- df[[col]]
    data.frame(subset = tag, parameter = gsub("\\[|\\]", "", pars[[col]]),
               N = sum(!is.na(r)),
               cov50 = round(cov(r, SBC_L * 0.25, SBC_L * 0.75), 3),
               cov90 = round(cov(r, SBC_L * 0.05, SBC_L * 0.95), 3),
               mean_frac_rank = round(mean(r, na.rm = TRUE) / SBC_L, 3))
  })
  tab <- do.call(rbind, rows)
  cat(sprintf("\n[SBC-COVERAGE] %s:\n", tag)); print(tab[, -1], row.names = FALSE)
  write.csv(tab, path, row.names = FALSE)
  cat(sprintf("[SBC-COVERAGE] table -> %s\n", path))
  invisible(tab)
}
coverage_table(sbc_fix, "rho fixed, all fits",
               file.path(OUT_DIR, "sbc_coverage_fixedrho.csv"))

# =============================================================================
# RUN 3: rho fixed, NON-CENTRED parameterisation (same datasets)
#
# reparameterise = "omega" is a documented option and was, until 0.2.8, running
# a translation move that rescaled the random-effect columns by sigma_re when
# they always hold centred values. That broke the likelihood invariance the move
# depends on. Calibration evidence for the package should cover the option, not
# assume it inherits the default's correctness.
# =============================================================================
sbc_nc <- run_sbc(rho_fixed = RHO_FIX, label = "rho fixed, non-centred",
                  reparameterise = "omega",
                  cache = file.path(OUT_DIR, "sbc_ranks_fixedrho_nc.csv"))
write.csv(sbc_nc, file.path(OUT_DIR, "sbc_ranks_fixedrho_nc.csv"), row.names = FALSE)
ess_report(sbc_nc, "rho fixed, non-centred")
write_regime(sbc_nc, "rho fixed at truth, reparameterise = omega",
             file.path(OUT_DIR, "sbc_regime_fixedrho_nc.csv"), RHO_FIX)
report_combined(sbc_nc, "rho FIXED, NON-CENTRED, ALL fits",
                file.path(FIG_DIR, "figure_sbc_ecdf_fixedrho_nc.png"),
                table_path = file.path(OUT_DIR, "sbc_table_fixedrho_nc_all.csv"))
coverage_table(sbc_nc, "rho fixed, non-centred, all fits",
               file.path(OUT_DIR, "sbc_coverage_fixedrho_nc.csv"))


# =============================================================================
# RUN 4: RANDOM INTERCEPT (rho fixed, population change-point)
#
# The three runs above all use b0 = ~ 1, so the joint (b0, u) translation move
# added in 0.2.8 never fires in them and they say nothing about it. This run
# puts a random intercept on b0 and a single population change-point, which is
# the configuration the move actually acts on. Without it a new sampler move
# would ship on a unit test alone.
# =============================================================================
run_sbc_ri <- function(label = "random intercept", cache = NULL) {
  this_version <- as.character(utils::packageVersion("smoothbp"))
  if (!is.null(cache) && nzchar(Sys.getenv("SBC_REUSE")) && file.exists(cache)) {
    df <- read.csv(cache, check.names = FALSE)
    saved <- if ("smoothbp_version" %in% names(df)) unique(df$smoothbp_version) else NA_character_
    if (length(saved) == 1L && !is.na(saved) && identical(saved, this_version)) {
      cat(sprintf("\n=== SBC run: %s (REUSING %d saved replicates, smoothbp %s) ===\n",
                  label, nrow(df), saved))
      return(df)
    }
  }
  cat(sprintf("\n=== SBC run: %s ===\n", label))
  set.seed(SBC_SEED)
  subj <- rep(seq_len(SBC_J), each = SBC_N)
  tau  <- rep(seq(SBC_TAU_LO, SBC_TAU_HI, length.out = SBC_N), times = SBC_J)
  res <- vector("list", SBC_REPS)
  for (r in seq_len(SBC_REPS)) {
    b0    <- rnorm(1, PR_B0["mean"], PR_B0["sd"])
    b1    <- rnorm(1, PR_B1["mean"], PR_B1["sd"])
    delta <- rnorm(1, PR_DELTA["mean"], PR_DELTA["sd"])
    omega <- rtrunc(PR_OMEGA["mean"], PR_OMEGA["sd"], PR_OMEGA["lb"], PR_OMEGA["ub"])
    sigu  <- draw_sigma(PR_SIGMAU["shape"], PR_SIGMAU["scale"])
    sigy  <- draw_sigma(PR_SIGMA["shape"],  PR_SIGMA["scale"])
    u_j   <- rnorm(SBC_J, 0, sigu)
    d     <- tau - omega
    mu    <- b0 + u_j[subj] + b1 * d + delta * d * plogis(RHO_FIX * d)
    y     <- mu + rnorm(SBC_J * SBC_N, 0, sigy)
    dat   <- data.frame(subject = factor(subj), tau = tau, y = y)

    fit <- tryCatch(smoothbp(y ~ tau, b0 = ~ 1 + (1 | subject), b1 = ~ 1,
                             omega = list(~ 1), deltas = list(~ 1),
                             rho = list(fixed(RHO_FIX)), data = dat,
                             priors = SBC_PRIORS, chains = SBC_CHAINS,
                             iter = SBC_ITER, warmup = SBC_WARM, seed = 5000L + r,
                             target_accept = TA_SBC, .verbose = FALSE),
                    error = function(e) NULL)
    if (is.null(fit)) next
    dr <- posterior::as_draws_df(fit$draws)
    keep <- round(seq(1, nrow(dr), length.out = SBC_L))
    rk <- function(col, truth) if (col %in% names(dr)) sum(dr[[col]][keep] < truth, na.rm = TRUE) else NA_integer_
    res[[r]] <- data.frame(
      smoothbp_version = this_version, rep = r,
      b0_true = b0, b1_true = b1, del_true = delta, abs_delta = abs(delta),
      om_true = omega, sig_u_true = sigu, sig_y_true = sigy,
      rank_omega = rk("omega1_(Intercept)", omega),
      rank_sigma = rk("sigma_u",            sigu),
      rank_delta = rk("delta1_(Intercept)", delta),
      rank_b0    = rk("b0_(Intercept)",     b0),
      rank_b1    = rk("b1_(Intercept)",     b1),
      rank_sigy  = rk("sigma",              sigy),
      rank_rho   = NA_integer_,
      ess_omega  = ess_of(fit$draws, "omega1_(Intercept)"),
      ess_sigma  = ess_of(fit$draws, "sigma_u"),
      ess_delta  = ess_of(fit$draws, "delta1_(Intercept)"),
      ess_b0     = ess_of(fit$draws, "b0_(Intercept)"),
      n_div = fit$n_divergent)
    if (r %% 40L == 0L) cat(sprintf("  %s %d/%d\n", label, r, SBC_REPS))
  }
  dplyr::bind_rows(res)
}

sbc_ri <- run_sbc_ri(cache = file.path(OUT_DIR, "sbc_ranks_randint.csv"))
write.csv(sbc_ri, file.path(OUT_DIR, "sbc_ranks_randint.csv"), row.names = FALSE)
cat(sprintf("\n[diagnostic] random intercept: divergent fits %d / %d; b0 bulk ESS median %.0f\n",
            sum(sbc_ri$n_div > 0), nrow(sbc_ri), median(sbc_ri$ess_b0, na.rm = TRUE)))
# This arm has a random INTERCEPT, so the rank_sigma column carries sigma_u,
# not the change-point between-group SD. Label it accordingly.
RI_PARS <- SBC_PARS
RI_PARS[["rank_sigma"]] <- "sigma[u]"
RI_PARS <- RI_PARS[names(RI_PARS) != "rank_rho"]
report_combined(sbc_ri, "RANDOM INTERCEPT, ALL fits",
                file.path(FIG_DIR, "figure_sbc_ecdf_randint.png"),
                table_path = file.path(OUT_DIR, "sbc_table_randint_all.csv"),
                pars = RI_PARS)
coverage_table(sbc_ri, "random intercept, all fits",
               file.path(OUT_DIR, "sbc_coverage_randint.csv"), pars = RI_PARS)

cat("\nReading the result (now that the SBC is valid):\n",
    " - rho-fixed all parameters PASS       -> sampler is calibrated in this regime\n",
    " - rho-fixed PASS but rho-estimated FAIL only via divergent fits -> convergence issue, not bias\n",
    " - rho-fixed still FAILS               -> genuine sampler bias to fix before any calibration claim\n",
    " - centred PASS but non-centred FAIL   -> the reparameterise = 'omega' path, not the model\n", sep = "")
