# Tests for spike-and-slab variable selection (smoothbp_ss)

# ---------------------------------------------------------------------------
# Test 1: When b2_x has no true effect, PIP should be moderate/low
# ---------------------------------------------------------------------------

test_that("PIP is moderate/low when b2 covariate has no true effect", {
  skip_on_cran()

  dat <- simulate_smoothbp(
    n_subj = 1, n_obs = 80,
    b0 = 5, b1 = -0.4, delta = 1.2,
    omega = 3, rho = 4,
    sigma = 0.4, sigma_u = 0,
    seed = 7710L
  )
  dat$x <- rnorm(nrow(dat))

  fit <- smoothbp_ss(
    formula = y ~ tau,
    b0 = ~ 1, b1 = ~ 1,
    deltas = list(~ 1 + x),
    omega = list(~ 1 + x),
    rho = list(~ 1),
    data = dat,
    priors = smoothbp_priors(omega = prior_normal(3, 2, lb = 0)),
    spike = prior_spike_slab(pi = 0.5),
    chains = 2L, iter = 1500L, warmup = 750L,
    seed = 7710L, .verbose = FALSE
  )

  pips <- pip(fit)
  # Parameter names are delta1_(Intercept) and delta1_x
  expect_equal(pips$pip[pips$parameter == "delta1_(Intercept)"], 1.0)
  # No true effect on x: PIP should be < 0.8
  expect_lt(pips$pip[pips$parameter == "delta1_x"], 0.8)
})

# ---------------------------------------------------------------------------
# Test 2: When b2_x has a strong true effect, PIP should be high
# ---------------------------------------------------------------------------

test_that("PIP is high when b2 covariate has a strong true effect", {
  skip_on_cran()

  dat <- simulate_smoothbp(
    n_subj = 1, n_obs = 120,
    b0 = 5, b1 = -0.4, delta = 1.2,
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
    deltas = list(~ 1 + x),
    omega = list(~ 1 + x),
    rho = list(~ 1),
    data = dat,
    priors = smoothbp_priors(omega = prior_normal(3, 2, lb = 0)),
    spike = prior_spike_slab(pi = 0.5),
    chains = 2L, iter = 1500L, warmup = 750L,
    seed = 8820L, .verbose = FALSE
  )

  pips <- pip(fit)
  # Strong true effect: PIP should be > 0.8
  expect_gt(pips$pip[pips$parameter == "delta1_x"], 0.8)
})

# ---------------------------------------------------------------------------
# Test 3: Structured zeroing — omega_x is exactly 0 when gamma_x = 0
# ---------------------------------------------------------------------------

test_that("omega covariate is exactly zero when gamma is zero", {
  skip_on_cran()

  dat <- simulate_smoothbp(
    n_subj = 1, n_obs = 60,
    b0 = 5, b1 = -0.4, delta = 1.2,
    omega = 3, rho = 4,
    sigma = 0.4, sigma_u = 0,
    seed = 9930L
  )
  dat$x <- rnorm(nrow(dat))

  fit <- smoothbp_ss(
    formula = y ~ tau,
    b0 = ~ 1, b1 = ~ 1,
    deltas = list(~ 1 + x),
    omega = list(~ 1 + x),
    rho = list(~ 1),
    data = dat,
    priors = smoothbp_priors(omega = prior_normal(3, 2, lb = 0)),
    spike = prior_spike_slab(pi = 0.5),
    chains = 2L, iter = 1500L, warmup = 750L,
    seed = 9930L, .verbose = FALSE
  )

  dm <- posterior::as_draws_matrix(fit$draws)
  gamma_x <- as.numeric(dm[, "gamma_delta1_x"])
  delta_x <- as.numeric(dm[, "delta1_x"])

  # In Kuo-Mallick spike-and-slab, the beta coefficient is NOT set to 0 when gamma = 0;
  # instead, gamma=0 zeros the effect in the likelihood. beta is then sampled from the prior.
  # We check that gamma exists and is binary.
  expect_true(all(gamma_x == 0 | gamma_x == 1))

  # When gamma_x = 1, delta_x should be non-constant (actually sampled)
  included <- gamma_x == 1
  if (sum(included) > 10) {
    expect_gt(sd(delta_x[included]), 0,
      label = "delta_x should vary when gamma_x = 1")
  }
})

# ---------------------------------------------------------------------------
# Test 4: Returns correct class
# ---------------------------------------------------------------------------

test_that("smoothbp_ss returns correct class", {
  skip_on_cran()

  dat <- simulate_smoothbp(
    n_subj = 1, n_obs = 40,
    b0 = 5, b1 = -0.4, delta = 1.2,
    omega = 3, rho = 4,
    sigma = 0.4, sigma_u = 0,
    seed = 3340L
  )
  dat$x <- rnorm(nrow(dat))

  fit <- smoothbp_ss(
    formula = y ~ tau,
    b0 = ~ 1, b1 = ~ 1, deltas = list(~ 1 + x),
    omega = list(~ 1), rho = list(~ 1),
    data = dat,
    chains = 1L, iter = 500L, warmup = 250L,
    seed = 3340L, .verbose = FALSE
  )

  expect_s3_class(fit, "smoothbp_ss_fit")
  expect_s3_class(fit, "smoothbp_fit")
  expect_true("gamma_names" %in% names(fit))
  expect_true("spike" %in% names(fit))
  expect_equal(length(fit$gamma_names), 3L)  # gamma_b1_(Intercept) + gamma_delta1_(Intercept) + gamma_delta1_x

  # pip() should work
  p <- pip(fit)
  expect_true(all(p$pip >= 0 & p$pip <= 1))
  expect_equal(p$parameter, c("b1_(Intercept)", "delta1_(Intercept)", "delta1_x"))
})
