# Tests for spike-and-slab variable selection (smoothbp_ss)

# ---------------------------------------------------------------------------
# Test 1: When b2_x has no true effect, PIP should be moderate/low
# ---------------------------------------------------------------------------

test_that("PIP is moderate/low when b2 covariate has no true effect", {
  skip_on_cran()

  dat <- simulate_smoothbp(
    n_subj = 1, n_obs = 80,
    b0 = 5, b1 = -0.4, b2 = 1.2,
    omega = 3, rho = 4,
    sigma = 0.4, sigma_u = 0,
    seed = 7710L
  )
  dat$x <- rnorm(nrow(dat))

  fit <- smoothbp_ss(
    formula = y ~ tau,
    b0 = ~ 1, b1 = ~ 1,
    b2 = ~ 1 + x,
    omega = ~ 1 + x,
    rho = ~ 1,
    data = dat,
    priors = smoothbp_priors(omega = prior_normal(3, 2, lb = 0)),
    spike = prior_spike_slab(pi = 0.5),
    chains = 2L, iter = 1500L, warmup = 750L,
    seed = 7710L, .verbose = FALSE
  )

  pips <- pip(fit)
  # Intercept always included
  expect_equal(pips["(Intercept)"], c("(Intercept)" = 1.0))
  # No true effect on x: PIP should be < 0.8
  expect_lt(pips["x"], 0.8)
})

# ---------------------------------------------------------------------------
# Test 2: When b2_x has a strong true effect, PIP should be high
# ---------------------------------------------------------------------------

test_that("PIP is high when b2 covariate has a strong true effect", {
  skip_on_cran()

  dat <- simulate_smoothbp(
    n_subj = 1, n_obs = 120,
    b0 = 5, b1 = -0.4, b2 = 1.2,
    omega = 3, rho = 4,
    sigma = 0.4, sigma_u = 0,
    seed = 8820L
  )
  dat$x <- rnorm(nrow(dat))
  # Add a true b2 covariate effect
  d <- dat$tau - 3.0
  s <- 1 / (1 + exp(-d * 4))
  dat$y <- dat$y + 0.8 * dat$x * d * s

  fit <- smoothbp_ss(
    formula = y ~ tau,
    b0 = ~ 1, b1 = ~ 1,
    b2 = ~ 1 + x,
    omega = ~ 1 + x,
    rho = ~ 1,
    data = dat,
    priors = smoothbp_priors(omega = prior_normal(3, 2, lb = 0)),
    spike = prior_spike_slab(pi = 0.5),
    chains = 2L, iter = 1500L, warmup = 750L,
    seed = 8820L, .verbose = FALSE
  )

  pips <- pip(fit)
  # Strong true effect: PIP should be > 0.8
  expect_gt(pips["x"], 0.8)
})

# ---------------------------------------------------------------------------
# Test 3: Structured zeroing — omega_x is exactly 0 when gamma_x = 0
# ---------------------------------------------------------------------------

test_that("omega covariate is exactly zero when gamma is zero", {
  skip_on_cran()

  dat <- simulate_smoothbp(
    n_subj = 1, n_obs = 60,
    b0 = 5, b1 = -0.4, b2 = 1.2,
    omega = 3, rho = 4,
    sigma = 0.4, sigma_u = 0,
    seed = 9930L
  )
  dat$x <- rnorm(nrow(dat))

  fit <- smoothbp_ss(
    formula = y ~ tau,
    b0 = ~ 1, b1 = ~ 1,
    b2 = ~ 1 + x,
    omega = ~ 1 + x,
    rho = ~ 1,
    data = dat,
    priors = smoothbp_priors(omega = prior_normal(3, 2, lb = 0)),
    spike = prior_spike_slab(pi = 0.5),
    chains = 2L, iter = 1500L, warmup = 750L,
    seed = 9930L, .verbose = FALSE
  )

  dm <- posterior::as_draws_matrix(fit$draws)
  gamma_x <- as.numeric(dm[, "gamma_x"])
  omega_x <- as.numeric(dm[, "omega_x"])

  # When gamma_x = 0, omega_x must be exactly 0
  excluded <- gamma_x == 0
  if (any(excluded)) {
    expect_true(all(omega_x[excluded] == 0),
      label = "omega_x should be exactly 0 when gamma_x = 0")
  }

  # When gamma_x = 1, omega_x should be non-constant (actually sampled)
  included <- gamma_x == 1
  if (sum(included) > 10) {
    expect_gt(sd(omega_x[included]), 0,
      label = "omega_x should vary when gamma_x = 1")
  }
})

# ---------------------------------------------------------------------------
# Test 4: Returns correct class
# ---------------------------------------------------------------------------

test_that("smoothbp_ss returns correct class", {
  skip_on_cran()

  dat <- simulate_smoothbp(
    n_subj = 1, n_obs = 40,
    b0 = 5, b1 = -0.4, b2 = 1.2,
    omega = 3, rho = 4,
    sigma = 0.4, sigma_u = 0,
    seed = 3340L
  )
  dat$x <- rnorm(nrow(dat))

  fit <- smoothbp_ss(
    formula = y ~ tau,
    b0 = ~ 1, b1 = ~ 1, b2 = ~ 1 + x,
    omega = ~ 1, rho = ~ 1,
    data = dat,
    chains = 1L, iter = 500L, warmup = 250L,
    seed = 3340L, .verbose = FALSE
  )

  expect_s3_class(fit, "smoothbp_ss_fit")
  expect_s3_class(fit, "smoothbp_fit")
  expect_true("gamma_names" %in% names(fit))
  expect_true("spike" %in% names(fit))
  expect_equal(length(fit$gamma_names), 2L)  # intercept + x

  # pip() should work
  p <- pip(fit)
  expect_true(all(p >= 0 & p <= 1))
  expect_equal(names(p), c("(Intercept)", "x"))
})
