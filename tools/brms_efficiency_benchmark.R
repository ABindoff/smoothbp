# =============================================================================
# brms_efficiency_benchmark.R
#
# Reproduces the benchmarking section of the manuscript: posterior agreement
# with brms, standardised 1-Wasserstein distances, and effective sample size
# per second. Split out of vignettes/brms-comparison.Rmd so the reported
# numbers come from a single script that writes its own configuration to disk.
#
# The comparison model uses a random INTERCEPT on b0, which smoothbp updates by
# conjugate Gibbs, not the random-effect NUTS blocks. It is therefore
# unaffected by the reversibility defect fixed in 0.2.8, and serves as an
# independent check that the collapsed change-point update of 0.2.8 targets the
# same posterior as before.
#
#   Rscript tools/brms_efficiency_benchmark.R
#   SMOOTHBP_LIB=/path/to/lib Rscript tools/brms_efficiency_benchmark.R
# =============================================================================

lib <- Sys.getenv("SMOOTHBP_LIB")
if (nzchar(lib)) .libPaths(c(lib, .libPaths()))
suppressPackageStartupMessages({library(smoothbp); library(posterior)})
have_brms <- requireNamespace("brms", quietly = TRUE)
cat(sprintf("[bench] smoothbp %s from %s | brms %s\n",
            as.character(packageVersion("smoothbp")), find.package("smoothbp"),
            if (have_brms) as.character(packageVersion("brms")) else "ABSENT"))

OUT_DIR <- file.path("tools", "part3_results")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

set.seed(31)
dat <- simulate_smoothbp(n_subj = 25, n_obs = 10, b0 = 5.0, b1 = -0.4,
                         delta = 1.4, omega = 3.2, rho = 4.0,
                         sigma = 0.4, sigma_u = 0.7, seed = 31L)

sbp_priors <- smoothbp_priors(
  b0 = prior_normal(0, 10), b1 = prior_normal(0, 2),
  omega = list(prior_normal(3, 2, lb = 0, ub = max(dat$tau))),
  rho = list(prior_normal(3, 2, lb = 0)),
  sigma = prior_invgamma(0.001, 0.001), sigma_u = prior_invgamma(0.001, 0.001))

fit_sbp_once <- function(seed) {
  t0 <- Sys.time()
  f <- smoothbp(y ~ tau, b0 = ~ 1 + (1 | subject), b1 = ~ 1,
                deltas = list(~ 1), omega = list(~ 1), rho = list(~ 1),
                data = dat, priors = sbp_priors, chains = 4L, iter = 2000L,
                warmup = 1000L, seed = seed, .verbose = FALSE)
  list(fit = f, secs = as.numeric(difftime(Sys.time(), t0, units = "secs")))
}

r31 <- fit_sbp_once(31L)
r32 <- fit_sbp_once(32L)          # within-sampler Monte Carlo reference
cat(sprintf("[bench] smoothbp: %.1f s (seed 31), %.1f s (seed 32)\n",
            r31$secs, r32$secs))

sbp_names  <- c("b0_(Intercept)", "b1_(Intercept)", "delta1_(Intercept)",
                "omega1_(Intercept)", "rho1_(Intercept)", "sigma", "sigma_u")
brms_names <- c("b_b0_Intercept", "b_b1_Intercept", "b_b2_Intercept",
                "b_omega_Intercept", "b_rho_Intercept", "sigma",
                "sd_subject__b0_Intercept")
pretty     <- c("b0", "b1", "delta1", "omega1", "rho1", "sigma", "sigma_u")

res <- list()
if (have_brms) {
  bf_smoothed <- brms::bf(
    y ~ b0 + b1 * (tau - omega) +
          b2 * (tau - omega) / (1 + exp(-(tau - omega) * rho)),
    b0 ~ 1 + (1 | subject), b1 ~ 1, b2 ~ 1, omega ~ 1, rho ~ 1, nl = TRUE)
  ub_omega <- max(dat$tau)
  priors_brms <- c(
    brms::prior(normal(0, 10), nlpar = "b0"),
    brms::prior(normal(0, 2),  nlpar = "b1"),
    brms::prior(normal(0, 2),  nlpar = "b2"),
    brms::prior_string("normal(3, 2)", nlpar = "omega", lb = 0, ub = ub_omega),
    brms::prior(normal(3, 2),  nlpar = "rho", lb = 0),
    brms::prior(student_t(3, 0, 10), class = "sigma"),
    brms::prior(student_t(3, 0, 10), class = "sd", nlpar = "b0"))
  init_fun <- function() list(
    b_b0 = array(rnorm(1, 5, 1)), b_b1 = array(rnorm(1, 0, 0.3)),
    b_b2 = array(rnorm(1, 0, 0.3)), b_omega = array(rnorm(1, 3, 0.3)),
    b_rho = array(rnorm(1, 3, 0.5)))

  t0 <- Sys.time()
  fit_brms <- brms::brm(bf_smoothed, data = dat, prior = priors_brms,
                        chains = 4, iter = 2000, warmup = 1000, seed = 31,
                        refresh = 0, init = init_fun,
                        control = list(adapt_delta = 0.95))
  time_brms_total <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  # Stan reports per-chain sampling times separately from compilation.
  st <- tryCatch(rstan::get_elapsed_time(fit_brms$fit), error = function(e) NULL)
  time_brms_sampling <- if (is.null(st)) NA_real_ else max(rowSums(st))
  cat(sprintf("[bench] brms: %.1f s total, %.1f s sampling only\n",
              time_brms_total, time_brms_sampling))

  d_sbp  <- posterior::as_draws_df(r31$fit$draws)
  d_sbp2 <- posterior::as_draws_df(r32$fit$draws)
  d_brm  <- posterior::as_draws_df(fit_brms)

  s_sbp <- posterior::summarise_draws(
    posterior::subset_draws(r31$fit$draws, variable = sbp_names))
  s_brm <- posterior::summarise_draws(
    posterior::subset_draws(posterior::as_draws(fit_brms), variable = brms_names))

  w1 <- function(a, b) {
    n <- min(length(a), length(b))
    qa <- quantile(a, ppoints(n), names = FALSE)
    qb <- quantile(b, ppoints(n), names = FALSE)
    mean(abs(qa - qb))
  }

  rows <- lapply(seq_along(sbp_names), function(i) {
    a <- d_sbp[[sbp_names[i]]]; b <- d_brm[[brms_names[i]]]
    a2 <- d_sbp2[[sbp_names[i]]]
    pooled <- sqrt((var(a) + var(b)) / 2)
    ra <- s_sbp[s_sbp$variable == sbp_names[i], ]
    rb <- s_brm[s_brm$variable == brms_names[i], ]
    data.frame(
      parameter   = pretty[i],
      sbp_mean    = round(mean(a), 3),   brms_mean = round(mean(b), 3),
      sbp_sd      = round(sd(a), 3),     brms_sd   = round(sd(b), 3),
      std_mean_diff = round(abs(mean(a) - mean(b)) / pooled, 3),
      w1_std        = round(w1(a, b) / pooled, 3),
      w1_std_ref    = round(w1(a, a2) / pooled, 3),
      sbp_ess     = round(ra$ess_bulk), brms_ess = round(rb$ess_bulk),
      sbp_ess_s   = round(ra$ess_bulk / r31$secs, 1),
      brms_ess_s  = round(rb$ess_bulk / ifelse(is.na(time_brms_sampling),
                                               time_brms_total, time_brms_sampling), 1))
  })
  res <- do.call(rbind, rows)
  print(res, row.names = FALSE)
  write.csv(res, file.path(OUT_DIR, "brms_benchmark.csv"), row.names = FALSE)

  meta <- data.frame(
    field = c("smoothbp_version", "brms_version", "n_subj", "n_obs", "chains",
              "iter", "warmup", "smoothbp_seconds", "brms_seconds_total",
              "brms_seconds_sampling"),
    value = c(as.character(packageVersion("smoothbp")),
              as.character(packageVersion("brms")), 25, 10, 4, 2000, 1000,
              round(r31$secs, 1), round(time_brms_total, 1),
              round(time_brms_sampling, 1)))
  write.csv(meta, file.path(OUT_DIR, "brms_benchmark_regime.csv"), row.names = FALSE)
  cat(sprintf("\n[bench] table  -> %s\n", file.path(OUT_DIR, "brms_benchmark.csv")))
} else {
  cat("[bench] brms not installed; smoothbp-only timings reported.\n")
}
