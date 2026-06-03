# Test that finite bounds on b0 are respected by the Gibbs sampler.
#
# The conjugate Gibbs step now treats draws as independence MH proposals
# and rejects any draw where a b0 coefficient falls outside [lb, ub].

test_that("b0 bounds are respected in posterior draws", {
  skip_on_cran()

  dat <- simulate_smoothbp(
    n_subj = 20, n_obs = 8,
    b0 = 5.0, b1 = -0.4, delta = 1.2,
    omega = 3.0, rho = 4.0,
    sigma = 0.4, sigma_u = 0.6,
    seed = 4271L
  )

  b0_lb <- 3.0
  b0_ub <- 8.0

  fit <- smoothbp(
    formula = y ~ tau,
    b0      = ~ 1 + (1 | subject),
    b1      = ~ 1,
    deltas  = list(~ 1),
    omega   = list(~ 1),
    rho     = list(~ 1),
    data    = dat,
    priors  = smoothbp_priors(
      b0    = prior_normal(5, 10, lb = b0_lb, ub = b0_ub),
      omega = prior_normal(3, 2, lb = 0)
    ),
    chains  = 2L,
    iter    = 1500L,
    warmup  = 750L,
    seed    = 4271L,
    .verbose = FALSE
  )

  draws <- posterior::as_draws_df(fit$draws)
  b0_draws <- draws[["b0_(Intercept)"]]

  expect_true(all(b0_draws >= b0_lb),
    label = sprintf("b0 lower bound: min draw = %.4f, lb = %.1f",
                    min(b0_draws), b0_lb))
  expect_true(all(b0_draws <= b0_ub),
    label = sprintf("b0 upper bound: max draw = %.4f, ub = %.1f",
                    max(b0_draws), b0_ub))
})
