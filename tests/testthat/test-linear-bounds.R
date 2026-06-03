# Tests that finite bounds on b1 and b2 are respected by the Gibbs sampler.
#
# The conjugate Gibbs draw is treated as an independence MH proposal;
# the entire linear draw is rejected when any coefficient falls outside
# its specified bounds (covering b0, b1, and b2 simultaneously).

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

.ci_overlap <- function(lo1, hi1, lo2, hi2) lo1 <= hi2 & lo2 <= hi1

# ---------------------------------------------------------------------------
# b1 bounds — no random effects (exercises sample_linear_coefs)
# ---------------------------------------------------------------------------

test_that("b1 bounds are respected (no random effects)", {
  skip_on_cran()

  dat <- simulate_smoothbp(
    n_subj = 1, n_obs = 40,
    b0 = 5.0, b1 = -0.4, delta = 1.2,
    omega = 3.0, rho = 4.0,
    sigma = 0.4, sigma_u = 0.0,
    seed = 1101L
  )

  b1_lb <- -2.0
  b1_ub <-  0.0   # true value -0.4 is within [lb, ub]

  fit <- smoothbp(
    formula = y ~ tau,
    b0      = ~ 1,
    b1      = ~ 1,
    deltas  = list(~ 1),
    omega   = list(~ 1),
    rho     = list(~ 1),
    data    = dat,
    priors  = smoothbp_priors(
      b1    = prior_normal(0, 2, lb = b1_lb, ub = b1_ub),
      omega = prior_normal(3, 2, lb = 0)
    ),
    chains  = 2L, iter = 3000L, warmup = 1000L,
    seed    = 1101L, .verbose = FALSE
  )

  b1_draws <- as.numeric(
    posterior::as_draws_matrix(fit$draws)[, "b1_(Intercept)"]
  )

  expect_true(all(b1_draws >= b1_lb),
    label = sprintf("b1 lb: min draw = %.4f, lb = %.1f", min(b1_draws), b1_lb))
  expect_true(all(b1_draws <= b1_ub),
    label = sprintf("b1 ub: max draw = %.4f, ub = %.1f", max(b1_draws), b1_ub))
})

# ---------------------------------------------------------------------------
# b2 bounds — no random effects (exercises sample_linear_coefs)
# ---------------------------------------------------------------------------

test_that("b2 bounds are respected (no random effects)", {
  skip_on_cran()

  dat <- simulate_smoothbp(
    n_subj = 1, n_obs = 40,
    b0 = 5.0, b1 = -0.4, delta = 1.2,
    omega = 3.0, rho = 4.0,
    sigma = 0.4, sigma_u = 0.0,
    seed = 2202L
  )

  b2_lb <- 0.0
  b2_ub <- 3.0   # true value 1.2 is within [lb, ub]

  fit <- smoothbp(
    formula = y ~ tau,
    b0      = ~ 1,
    b1      = ~ 1,
    deltas  = list(~ 1),
    omega   = list(~ 1),
    rho     = list(~ 1),
    data    = dat,
    priors  = smoothbp_priors(
      deltas = prior_normal(1, 2, lb = b2_lb, ub = b2_ub),
      omega = prior_normal(3, 2, lb = 0)
    ),
    chains  = 2L, iter = 3000L, warmup = 1000L,
    seed    = 2202L, .verbose = FALSE
  )

  b2_draws <- as.numeric(
    posterior::as_draws_matrix(fit$draws)[, "delta1_(Intercept)"]
  )

  expect_true(all(b2_draws >= b2_lb),
    label = sprintf("b2 lb: min draw = %.4f, lb = %.1f", min(b2_draws), b2_lb))
  expect_true(all(b2_draws <= b2_ub),
    label = sprintf("b2 ub: max draw = %.4f, ub = %.1f", max(b2_draws), b2_ub))
})

# ---------------------------------------------------------------------------
# b1 and b2 bounds with random intercepts (exercises sample_linear_coefs_joint)
# ---------------------------------------------------------------------------

test_that("b1 and b2 bounds are respected with random intercepts", {
  skip_on_cran()

  dat <- simulate_smoothbp(
    n_subj = 20, n_obs = 8,
    b0 = 5.0, b1 = -0.4, delta = 1.2,
    omega = 3.0, rho = 4.0,
    sigma = 0.4, sigma_u = 0.6,
    seed = 3303L
  )

  b1_lb <- -2.0;  b1_ub <- 0.0    # true -0.4 inside
  b2_lb <-  0.0;  b2_ub <- 3.0    # true  1.2 inside

  fit <- smoothbp(
    formula = y ~ tau,
    b0      = ~ 1 + (1 | subject),
    b1      = ~ 1,
    deltas  = list(~ 1),
    omega   = list(~ 1),
    rho     = list(~ 1),
    data    = dat,
    priors  = smoothbp_priors(
      b1    = prior_normal(0, 2, lb = b1_lb, ub = b1_ub),
      deltas = prior_normal(1, 2, lb = b2_lb, ub = b2_ub),
      omega = prior_normal(3, 2, lb = 0)
    ),
    chains  = 2L, iter = 3000L, warmup = 1000L,
    seed    = 3303L, .verbose = FALSE
  )

  dm <- posterior::as_draws_matrix(fit$draws)
  b1_draws <- as.numeric(dm[, "b1_(Intercept)"])
  b2_draws <- as.numeric(dm[, "delta1_(Intercept)"])

  expect_true(all(b1_draws >= b1_lb),
    label = sprintf("b1 lb (RE): min = %.4f", min(b1_draws)))
  expect_true(all(b1_draws <= b1_ub),
    label = sprintf("b1 ub (RE): max = %.4f", max(b1_draws)))
  expect_true(all(b2_draws >= b2_lb),
    label = sprintf("b2 lb (RE): min = %.4f", min(b2_draws)))
  expect_true(all(b2_draws <= b2_ub),
    label = sprintf("b2 ub (RE): max = %.4f", max(b2_draws)))
})

# ---------------------------------------------------------------------------
# Posterior is not collapsed to the bound — inference still works
# ---------------------------------------------------------------------------

test_that("b1/b2 bounds do not degenerate the posterior when truth is interior", {
  skip_on_cran()

  dat <- simulate_smoothbp(
    n_subj = 20, n_obs = 10,
    b0 = 5.0, b1 = -0.4, delta = 1.2,
    omega = 3.0, rho = 4.0,
    sigma = 0.4, sigma_u = 0.5,
    seed = 4404L
  )

  # Wide bounds — should barely affect inference
  fit <- smoothbp(
    formula = y ~ tau,
    b0      = ~ 1 + (1 | subject),
    b1      = ~ 1,
    deltas  = list(~ 1),
    omega   = list(~ 1),
    rho     = list(~ 1),
    data    = dat,
    priors  = smoothbp_priors(
      b1    = prior_normal(0, 2, lb = -5, ub = 5),
      deltas = prior_normal(1, 2, lb = -5, ub = 5),
      omega = prior_normal(3, 2, lb = 0)
    ),
    chains  = 2L, iter = 3000L, warmup = 1000L,
    seed    = 4404L, .verbose = FALSE
  )

  s <- summary(fit)

  b1_row <- s[s$variable == "b1_(Intercept)", ]
  b2_row <- s[s$variable == "delta1_(Intercept)", ]

  # 95% CI should contain the true value
  expect_true(-0.4 >= b1_row$Q2.5 & -0.4 <= b1_row$Q97.5,
    label = sprintf("b1 CI covers truth: [%.3f, %.3f]", b1_row$Q2.5, b1_row$Q97.5))
  expect_true( 1.2 >= b2_row$Q2.5 &  1.2 <= b2_row$Q97.5,
    label = sprintf("b2 CI covers truth: [%.3f, %.3f]", b2_row$Q2.5, b2_row$Q97.5))
})
