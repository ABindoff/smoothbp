library(smoothbp)

# The collapsed (Rao-Blackwellised) indicator update integrates the slab
# coefficient out before drawing gamma, instead of conditioning on whatever the
# coefficient happens to be. Under the plug-in Kuo-Mallick update an excluded
# coefficient sits wherever the diffuse slab put it, fits badly, and so rarely
# re-enters; gamma and beta lock together and gamma's ESS collapses into the
# tens. These tests pin the property that motivated the change.

test_that("inclusion indicators mix rather than sticking", {
  testthat::skip_on_cran()
  set.seed(4242)

  n <- 120
  tau <- seq(-6, 10, length.out = n)
  x <- rep(c(0, 1), length.out = n)
  # No true covariate effect on delta: gamma_delta1_x should wander, and it can
  # only wander if the indicator is not welded to its coefficient.
  d <- tau - 2
  y <- 3 + 0.2 * d + 1.5 * d * plogis(3 * d) + rnorm(n, 0, 0.4)
  dat <- data.frame(tau = tau, y = y, x = factor(x))

  fit <- smoothbp_ss(y ~ tau, b0 = ~ 1, b1 = ~ 1,
                     omega = list(~ 1), deltas = list(~ x),
                     rho = list(fixed(3)), data = dat,
                     chains = 2L, iter = 6000L, warmup = 2000L,
                     seed = 11L, .verbose = FALSE)

  g <- as.numeric(posterior::extract_variable(fit$draws, "gamma_delta1_x1"))
  pip <- mean(g)
  skip_if(pip <= 0 || pip >= 1, "indicator never moved on this data")

  # An indicator drawn independently each sweep flips with probability
  # 2*pip*(1-pip). Comparing against that makes the statistic scale-free, so it
  # works at any PIP rather than only for indicators sitting near 0.5. A welded
  # indicator reaches a small fraction of the ideal; a collapsed one gets close
  # to it, because gamma no longer has to drag a stale coefficient with it.
  flips <- mean(abs(diff(g)) > 0)
  ideal <- 2 * pip * (1 - pip)
  expect_gt(flips / ideal, 0.4)

  ess <- posterior::ess_basic(matrix(g, ncol = 1))
  expect_gt(ess, 200)
})

test_that("a strongly supported column is still included", {
  testthat::skip_on_cran()
  set.seed(808)

  n <- 120
  tau <- seq(-6, 10, length.out = n)
  x <- rep(c(0, 1), length.out = n)
  d <- tau - 2
  # Large covariate effect on the slope change: the indicator must lock on.
  y <- 3 + 0.2 * d + (1.0 + 2.5 * x) * d * plogis(3 * d) + rnorm(n, 0, 0.4)
  dat <- data.frame(tau = tau, y = y, x = factor(x))

  fit <- smoothbp_ss(y ~ tau, b0 = ~ 1, b1 = ~ 1,
                     omega = list(~ 1), deltas = list(~ x),
                     rho = list(fixed(3)), data = dat,
                     chains = 2L, iter = 4000L, warmup = 1500L,
                     seed = 12L, .verbose = FALSE)

  pips <- pip(fit)
  expect_gt(pips$pip[pips$parameter == "delta1_x1"], 0.9)
})

test_that("bounded slabs keep coefficients inside their bounds", {
  testthat::skip_on_cran()
  set.seed(1234)

  # The collapsed update draws the slab coefficient itself, so it has to respect
  # prior bounds; the plug-in update never drew it here and relied on the linear
  # block's rejection step.
  n <- 100
  tau <- seq(-5, 9, length.out = n)
  d <- tau - 2
  y <- 3 + 0.1 * d + 1.2 * d * plogis(3 * d) + rnorm(n, 0, 0.4)
  dat <- data.frame(tau = tau, y = y)

  pri <- smoothbp_priors(
    b0 = prior_normal(3, 3), b1 = prior_normal(0, 1),
    deltas = prior_normal(0, 2, lb = 0, ub = 3),
    omega = prior_normal(2, 2, lb = -5, ub = 9)
  )

  fit <- smoothbp_ss(y ~ tau, b0 = ~ 1, b1 = ~ 1,
                     omega = list(~ 1), deltas = list(~ 1),
                     rho = list(fixed(3)), data = dat, priors = pri,
                     chains = 2L, iter = 4000L, warmup = 1500L,
                     seed = 13L, .verbose = FALSE)

  del <- as.numeric(posterior::extract_variable(fit$draws, "delta1_(Intercept)"))
  expect_true(all(del >= 0 - 1e-8), label = "delta draws respect lower bound")
  expect_true(all(del <= 3 + 1e-8), label = "delta draws respect upper bound")
})
