library(smoothbp)
library(posterior)

test_that("Zero breakpoint model recovers linear parameters", {
  testthat::skip_on_cran()
  set.seed(123)
  n <- 100
  tau <- seq(0, 10, length.out = n)
  y <- 5 + 0.5 * tau + rnorm(n, sd = 0.5)
  dat <- data.frame(y = y, tau = tau)

  fit <- smoothbp(
    y ~ tau,
    deltas = list(),
    omega  = list(),
    rho    = list(),
    data = dat,
    chains = 2, iter = 1000, warmup = 500,
    .verbose = FALSE
  )

  s <- summarise_draws(fit$draws)
  expect_equal(s$mean[s$variable == "b0_(Intercept)"], 5, tolerance = 0.2)
  expect_equal(s$mean[s$variable == "b1_(Intercept)"], 0.5, tolerance = 0.1)
})

test_that("Single breakpoint recovery works", {
  testthat::skip_on_cran()
  set.seed(42)
  n <- 150
  tau <- seq(0, 10, length.out = n)
  om_true <- 4
  rho_true <- 5
  b0_true <- 10
  b1_true <- -0.2
  delta_true <- 1.5

  di <- tau - om_true
  si <- 1 / (1 + exp(-di * rho_true))
  mu <- b0_true + b1_true * di + delta_true * di * si
  y <- mu + rnorm(n, sd = 0.3)
  dat <- data.frame(y = y, tau = tau)

  fit <- smoothbp(
    y ~ tau,
    deltas = list(~ 1),
    omega  = list(~ 1),
    rho    = list(~ 1),
    data = dat,
    chains = 2, iter = 2000, warmup = 1000,
    .verbose = FALSE
  )

  s <- summarise_draws(fit$draws)
  expect_equal(s$mean[s$variable == "omega1_(Intercept)"], om_true, tolerance = 0.5)
  expect_equal(s$mean[s$variable == "delta1_(Intercept)"], delta_true, tolerance = 0.5)
})
