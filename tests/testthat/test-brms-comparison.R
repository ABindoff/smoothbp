# Cross-validation against brms (Stan / NUTS)
#
# These tests fit the same model with both smoothbp and brms, then check that
# population-level posterior means agree within a tolerance.  They are slow
# (~2–3 min each) and require brms, so they are skipped on CRAN and when brms
# is not installed.
#
# Two scenarios are covered:
#   1. Intercept-only model (all parameters scalar)
#   2. Covariate on omega (omega ~ 1 + treatment), exercising the
#      HMC-within-Gibbs sampler path
#
# All population-level parameters are compared.  The joint sampling of
# (b0, u, b1, b2) in a single conjugate Gibbs block breaks the b0-u
# coupling that previously inflated posterior variance for b0 and sigma_u.

# ---------------------------------------------------------------------------
# Helper: compare smoothbp vs brms posterior means
# ---------------------------------------------------------------------------
.compare_means <- function(fit_sbp, fit_brms,
                           sbp_names, brms_names,
                           tol_pct = 10,
                           skip_params = character(0)) {

  sbp_draws  <- suppressWarnings(
    posterior::as_draws_df(fit_sbp$draws)[, sbp_names]
  )
  brms_draws <- as.data.frame(
    posterior::as_draws_df(fit_brms)
  )[, brms_names]
  colnames(brms_draws) <- sbp_names

  sbp_means  <- colMeans(sbp_draws)
  brms_means <- colMeans(brms_draws)

  for (nm in sbp_names) {
    if (nm %in% skip_params) next

    delta_pct <- 100 * abs(sbp_means[[nm]] - brms_means[[nm]]) /
      (abs(brms_means[[nm]]) + 1e-8)

    testthat::expect_lt(
      delta_pct, tol_pct,
      label = sprintf(
        "%s: smoothbp=%.4f, brms=%.4f, delta=%.1f%%",
        nm, sbp_means[[nm]], brms_means[[nm]], delta_pct
      )
    )
  }
}


# ===========================================================================
# Test 1: Intercept-only model
# ===========================================================================
test_that("smoothbp matches brms on intercept-only model", {
  skip_on_cran()
  skip_if_not_installed("brms")

  set.seed(31)
  dat <- simulate_smoothbp(
    n_subj = 25, n_obs = 10,
    b0 = 5.0, b1 = -0.4, b2 = 1.4,
    omega = 3.2, rho = 4.0,
    sigma = 0.4, sigma_u = 0.7,
    seed = 31L
  )

  # --- smoothbp ---
  fit_sbp <- smoothbp(
    formula = y ~ tau,
    b0      = ~ 1 + (1 | subject),
    b1      = ~ 1,
    b2      = ~ 1,
    omega   = ~ 1,
    rho     = ~ 1,
    data    = dat,
    priors  = smoothbp_priors(
      b0    = prior_normal(0, 10),
      b1    = prior_normal(0, 2),
      b2    = prior_normal(0, 2),
      omega = prior_normal(3, 2, lb = 0, ub = max(dat$tau)),
      rho   = prior_normal(3, 2, lb = 0)
    ),
    chains = 4L, iter = 2000L, warmup = 1000L,
    seed   = 31L, .verbose = FALSE
  )

  # --- brms ---
  bf_mod <- brms::bf(
    y ~ b0 + b1 * (tau - omega) +
          b2 * (tau - omega) / (1 + exp(-(tau - omega) * rho)),
    b0 ~ 1 + (1 | subject),
    b1 ~ 1, b2 ~ 1, omega ~ 1, rho ~ 1,
    nl = TRUE
  )

  ub_om <- max(dat$tau)
  priors_brms <- c(
    brms::prior(normal(0, 10), nlpar = "b0"),
    brms::prior(normal(0, 2),  nlpar = "b1"),
    brms::prior(normal(0, 2),  nlpar = "b2"),
    brms::prior_string("normal(3, 2)", nlpar = "omega", lb = 0, ub = ub_om),
    brms::prior(normal(3, 2),  nlpar = "rho", lb = 0)
  )

  init_fun <- function() list(
    b_b0    = array(rnorm(1, 5, 1)),
    b_b1    = array(rnorm(1, 0, 0.3)),
    b_b2    = array(rnorm(1, 0, 0.3)),
    b_omega = array(rnorm(1, 3, 0.3)),
    b_rho   = array(rnorm(1, 3, 0.5))
  )

  fit_brms <- brms::brm(
    bf_mod,
    data    = dat,
    prior   = priors_brms,
    chains  = 4, iter = 2000, warmup = 1000,
    seed    = 31, refresh = 0,
    init    = init_fun,
    control = list(adapt_delta = 0.95)
  )

  # Check brms converged before comparing
  brms_rhat <- posterior::summarise_draws(
    posterior::as_draws_df(fit_brms), rhat = posterior::rhat
  )
  max_rhat <- max(brms_rhat$rhat, na.rm = TRUE)
  expect_lt(max_rhat, 1.05, label = "brms Rhat check")

  sbp_names <- c(
    "b0_(Intercept)", "b1_(Intercept)", "b2_(Intercept)",
    "omega_(Intercept)", "rho_(Intercept)", "sigma", "sigma_u"
  )
  brms_names <- c(
    "b_b0_Intercept", "b_b1_Intercept", "b_b2_Intercept",
    "b_omega_Intercept", "b_rho_Intercept", "sigma",
    "sd_subject__b0_Intercept"
  )

  .compare_means(fit_sbp, fit_brms, sbp_names, brms_names, tol_pct = 10)
})


# ===========================================================================
# Test 2: Covariate on omega (HMC-within-Gibbs path)
# ===========================================================================
test_that("smoothbp matches brms with covariate on omega", {
  skip_on_cran()
  skip_if_not_installed("brms")

  # --- Simulate data with treatment effect on omega ---
  set.seed(8147)

  n_subj   <- 30
  n_obs    <- 10
  tau_range <- c(0, 6)

  b0_true      <- 5.0
  b1_true      <- -0.4
  b2_true      <- 1.4
  omega_int    <- 3.2
  omega_trt    <- -0.8
  rho_true     <- 4.0
  sigma_true   <- 0.4
  sigma_u_true <- 0.7

  treatment <- rep(c(0, 1), each = n_subj / 2)
  u_j <- rnorm(n_subj, 0, sigma_u_true)
  .sigmoid <- function(x) 1 / (1 + exp(-x))

  rows <- vector("list", n_subj)
  for (j in seq_len(n_subj)) {
    tau_j   <- seq(tau_range[1], tau_range[2], length.out = n_obs)
    omega_j <- omega_int + omega_trt * treatment[j]
    d_j     <- tau_j - omega_j
    s_j     <- .sigmoid(d_j * rho_true)
    mu_j    <- (b0_true + u_j[j]) + b1_true * d_j + b2_true * d_j * s_j
    y_j     <- mu_j + rnorm(n_obs, 0, sigma_true)
    rows[[j]] <- data.frame(
      subject   = factor(j),
      tau       = tau_j,
      treatment = treatment[j],
      y         = y_j
    )
  }
  dat_cov <- do.call(rbind, rows)
  rownames(dat_cov) <- NULL

  # --- smoothbp (HMC-within-Gibbs for omega) ---
  fit_sbp <- smoothbp(
    formula = y ~ tau,
    b0      = ~ 1 + (1 | subject),
    b1      = ~ 1,
    b2      = ~ 1,
    omega   = ~ 1 + treatment,
    rho     = ~ 1,
    data    = dat_cov,
    priors  = smoothbp_priors(
      b0    = prior_normal(0, 10),
      b1    = prior_normal(0, 2),
      b2    = prior_normal(0, 2),
      omega = list(
        "(Intercept)" = prior_normal(3, 2, lb = 0, ub = max(dat_cov$tau)),
        "treatment"   = prior_normal(0, 2)
      ),
      rho   = prior_normal(3, 2, lb = 0)
    ),
    chains = 4L, iter = 2000L, warmup = 1000L,
    seed   = 8147L, .verbose = FALSE
  )

  # --- brms ---
  bf_cov <- brms::bf(
    y ~ b0 + b1 * (tau - omega) +
          b2 * (tau - omega) / (1 + exp(-(tau - omega) * rho)),
    b0    ~ 1 + (1 | subject),
    b1    ~ 1,
    b2    ~ 1,
    omega ~ 1 + treatment,
    rho   ~ 1,
    nl = TRUE
  )

  priors_brms <- c(
    brms::prior(normal(0, 10), nlpar = "b0"),
    brms::prior(normal(0, 2),  nlpar = "b1"),
    brms::prior(normal(0, 2),  nlpar = "b2"),
    brms::prior_string("normal(3, 2)", nlpar = "omega",
                        coef = "Intercept"),
    brms::prior(normal(0, 2),  nlpar = "omega", coef = "treatment"),
    brms::prior(normal(3, 2),  nlpar = "rho", lb = 0)
  )

  init_fun <- function() list(
    b_b0    = array(rnorm(1, 5, 1)),
    b_b1    = array(rnorm(1, 0, 0.3)),
    b_b2    = array(rnorm(1, 0, 0.3)),
    b_omega = array(c(rnorm(1, 3, 0.3), rnorm(1, -0.5, 0.3))),
    b_rho   = array(rnorm(1, 3, 0.5))
  )

  fit_brms <- brms::brm(
    bf_cov,
    data    = dat_cov,
    prior   = priors_brms,
    chains  = 4, iter = 2000, warmup = 1000,
    seed    = 8147, refresh = 0,
    init    = init_fun,
    control = list(adapt_delta = 0.95)
  )

  # Check brms converged
  brms_rhat <- posterior::summarise_draws(
    posterior::as_draws_df(fit_brms), rhat = posterior::rhat
  )
  max_rhat <- max(brms_rhat$rhat, na.rm = TRUE)
  expect_lt(max_rhat, 1.05, label = "brms Rhat check")

  sbp_names <- c(
    "b0_(Intercept)", "b1_(Intercept)", "b2_(Intercept)",
    "omega_(Intercept)", "omega_treatment",
    "rho_(Intercept)", "sigma", "sigma_u"
  )
  brms_names <- c(
    "b_b0_Intercept", "b_b1_Intercept", "b_b2_Intercept",
    "b_omega_Intercept", "b_omega_treatment",
    "b_rho_Intercept", "sigma", "sd_subject__b0_Intercept"
  )

  .compare_means(fit_sbp, fit_brms, sbp_names, brms_names, tol_pct = 10)
})
