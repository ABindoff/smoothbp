# Validation tests for smoothbp v2 (Multi-Breakpoint)

library(smoothbp)
library(ggplot2)
library(dplyr)
library(tidyr)
library(posterior)

# ---------------------------------------------------------------------------
# 1. Zero Breakpoint Model (Linear Fallback)
# ---------------------------------------------------------------------------
test_zero_bp <- function() {
  message("\n--- Testing Zero Breakpoint Model ---")
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
    chains = 2, iter = 1000, warmup = 500
  )

  print(summarise_draws(fit$draws))
  # Should recover intercept ~5, b1 ~0.5, sigma ~0.5
}

# ---------------------------------------------------------------------------
# 2. Single Breakpoint Parity (Recovery)
# ---------------------------------------------------------------------------
test_single_bp_recovery <- function() {
  message("\n--- Testing Single Breakpoint Recovery ---")
  set.seed(42)
  n <- 150
  tau <- seq(0, 10, length.out = n)
  om_true <- 4
  rho_true <- 5
  b0_true <- 10
  b1_true <- -0.2
  delta_true <- 1.5

  # Smooth change model
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
    chains = 2, iter = 2000, warmup = 1000
  )

  sumry <- summarise_draws(fit$draws)
  print(sumry)

  # Check recovery
  # b0_ (Intercept) -> should be close to 10
  # b1_ (Intercept) -> should be close to -0.2
  # delta1_ (Intercept) -> should be close to 1.5
  # omega1_ (Intercept) -> should be close to 4
}

# ---------------------------------------------------------------------------
# 3. Model Regularization (1 BP Data -> 3 BP Model)
# ---------------------------------------------------------------------------
test_regularization <- function() {
  message("\n--- Testing Regularization (1 BP data vs 3 BP model) ---")
  set.seed(42)
  n <- 200
  tau <- seq(0, 10, length.out = n)
  om_true <- 5
  rho_true <- 4
  b0_true <- 2
  b1_true <- 0.5
  delta_true <- -1.0

  di <- tau - om_true
  si <- 1 / (1 + exp(-di * rho_true))
  mu <- b0_true + b1_true * di + delta_true * di * si
  y <- mu + rnorm(n, sd = 0.2)
  dat <- data.frame(y = y, tau = tau)

  # Fit with 3 possible breakpoints using spike-and-slab
  fit <- smoothbp_ss(
    y ~ tau,
    deltas = list(~ 1, ~ 1, ~ 1),
    omega  = list(~ 1, ~ 1, ~ 1),
    rho    = list(~ 1, ~ 1, ~ 1),
    data = dat,
    spike = prior_spike_slab(pi = 0.1, learn_pi = TRUE),
    chains = 3, iter = 2000, warmup = 1000
  )

  # Check PIPs
  pips <- subset(summarise_draws(fit$draws), grepl("^gamma_delta", variable))
  print(pips)
  # One delta should have PIP ~ 1, others should be low.
}

# ---------------------------------------------------------------------------
# 4. Large Number of Breakpoints (Scaling)
# ---------------------------------------------------------------------------
test_large_k <- function() {
  message("\n--- Testing Scaling (12 Breakpoints) ---")
  set.seed(7)
  n <- 500
  tau <- seq(0, 100, length.out = n)
  y <- sin(tau / 5) + rnorm(n, sd = 0.1) # Wavy data
  dat <- data.frame(y = y, tau = tau)

  # Fit with 12 segments
  fit <- smoothbp(
    y ~ tau,
    deltas = rep(list(~ 1), 12),
    omega  = rep(list(~ 1), 12),
    rho    = rep(list(~ 1), 12),
    data = dat,
    chains = 1, iter = 5000, warmup = 4000,
    .verbose = TRUE
  )

  message("Successfully fit model with 12 breakpoints.")
  print(dim(fit$draws))
  plot(fitted(fit)$fitted_mean ~ tau)
}

# Run tests if called directly
if (!interactive()) {
  test_zero_bp()
  test_single_bp_recovery()
  test_regularization()
  test_large_k()
}
