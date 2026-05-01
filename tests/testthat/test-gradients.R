# Numerical gradient checks for the analytical gradients used by the
# HMC-within-Gibbs steps for beta_om and beta_rho.
#
# Strategy: reimplement the forward model in R, compute the analytical
# gradient using the same calculus as the Rust code, and compare against
# central finite differences.  This validates the mathematical derivation
# independently of the compiled Rust code.
#
# A second test then runs the full sampler on a covariate-in-omega problem
# and checks that the HMC acceptance rate is not degenerate (which would
# indicate a gradient bug in the compiled code).

# -- Forward model helpers ---------------------------------------------------

sigmoid <- function(x) ifelse(x >= 0, 1 / (1 + exp(-x)), exp(x) / (1 + exp(x)))

#' Log-likelihood for the smoothbp model given all parameters.
#' Returns a scalar.
ll_smoothbp <- function(y, tau, x_b0, x_b1, x_b2, x_om, x_rho,
                        beta_b0, beta_b1, beta_b2, beta_om, beta_rho,
                        sigma) {
  omega <- as.numeric(x_om  %*% beta_om)
  rho   <- as.numeric(x_rho %*% beta_rho)
  b0  <- as.numeric(x_b0 %*% beta_b0)
  b1  <- as.numeric(x_b1 %*% beta_b1)
  b2  <- as.numeric(x_b2 %*% beta_b2)
  d   <- tau - omega
  s   <- sigmoid(d * rho)
  mu  <- b0 + d * b1 + d * s * b2
  sum(dnorm(y, mu, sigma, log = TRUE))
}

#' Analytical gradient of the log-likelihood w.r.t. beta_om.
#' Returns a vector of length ncol(x_om).
grad_ll_beta_om <- function(y, tau, x_b0, x_b1, x_b2, x_om, x_rho,
                            beta_b0, beta_b1, beta_b2, beta_om, beta_rho,
                            sigma) {
  omega <- as.numeric(x_om  %*% beta_om)
  rho   <- as.numeric(x_rho %*% beta_rho)
  b0v <- as.numeric(x_b0 %*% beta_b0)
  b1v <- as.numeric(x_b1 %*% beta_b1)
  b2v <- as.numeric(x_b2 %*% beta_b2)
  d   <- tau - omega
  s   <- sigmoid(d * rho)
  mu  <- b0v + d * b1v + d * s * b2v
  resid <- y - mu

  # dmu/domega = -(b1 + s*b2 + d*rho*s*(1-s)*b2)
  dmu_domega <- -(b1v + s * b2v + d * rho * s * (1 - s) * b2v)
  # dll/dbeta_om_k = sum_i resid_i / sigma^2 * dmu_domega_i * x_om[i,k]
  factor <- resid / (sigma^2) * dmu_domega
  as.numeric(crossprod(x_om, factor))
}

#' Analytical gradient of the log-likelihood w.r.t. beta_rho.
grad_ll_beta_rho <- function(y, tau, x_b0, x_b1, x_b2, x_om, x_rho,
                             beta_b0, beta_b1, beta_b2, beta_om, beta_rho,
                             sigma) {
  omega <- as.numeric(x_om  %*% beta_om)
  rho   <- as.numeric(x_rho %*% beta_rho)
  b0v <- as.numeric(x_b0 %*% beta_b0)
  b1v <- as.numeric(x_b1 %*% beta_b1)
  b2v <- as.numeric(x_b2 %*% beta_b2)
  d   <- tau - omega
  s   <- sigmoid(d * rho)
  mu  <- b0v + d * b1v + d * s * b2v
  resid <- y - mu

  # dmu/drho = d^2 * s * (1 - s) * b2
  dmu_drho <- d^2 * s * (1 - s) * b2v
  factor <- resid / (sigma^2) * dmu_drho
  as.numeric(crossprod(x_rho, factor))
}

#' Analytical gradient of the (unconstrained) normal log-prior.
grad_log_prior <- function(values, means, sds) {
  -(values - means) / (sds^2)
}

# -- Numerical gradient via central finite differences -----------------------

numerical_grad <- function(f, x, ..., eps = 1e-6) {
  g <- numeric(length(x))
  for (k in seq_along(x)) {
    x_plus  <- x;  x_plus[k]  <- x[k] + eps
    x_minus <- x;  x_minus[k] <- x[k] - eps
    g[k] <- (f(x_plus, ...) - f(x_minus, ...)) / (2 * eps)
  }
  g
}

# ===========================================================================
# Tests
# ===========================================================================

test_that("analytical grad_ll_beta_om matches numerical gradient", {
  set.seed(4821)
  n <- 20
  tau <- seq(0, 7, length.out = n)
  x_b0 <- cbind(1)
  x_b1 <- cbind(1)
  x_b2 <- cbind(1)
  x_om  <- cbind(1, rbinom(n, 1, 0.5))
  x_rho <- cbind(1, rnorm(n, 0, 0.3))
  y <- rnorm(n, 5, 1)

  beta_b0  <- 5.0
  beta_b1  <- -0.4
  beta_b2  <- 1.3
  beta_om  <- c(3.0, 0.5)
  beta_rho <- c(4.0, -0.3)
  sigma    <- 0.5

  analytic <- grad_ll_beta_om(
    y, tau, x_b0, x_b1, x_b2, x_om, x_rho,
    beta_b0, beta_b1, beta_b2, beta_om, beta_rho, sigma
  )

  # Wrapper for numerical_grad: takes beta_om as argument, returns scalar ll
  ll_fn <- function(bom) {
    ll_smoothbp(y, tau, x_b0, x_b1, x_b2, x_om, x_rho,
                beta_b0, beta_b1, beta_b2, bom, beta_rho, sigma)
  }
  numerical <- numerical_grad(ll_fn, beta_om)

  # Relative error should be < 1e-4 for each coordinate.
  rel_err <- abs(analytic - numerical) / pmax(abs(numerical), 1e-12)
  expect_true(
    all(rel_err < 1e-4),
    label = sprintf(
      "grad_beta_om: analytic = [%s], numerical = [%s], rel_err = [%s]",
      paste(round(analytic, 6), collapse = ", "),
      paste(round(numerical, 6), collapse = ", "),
      paste(formatC(rel_err, format = "e", digits = 2), collapse = ", ")
    )
  )
})

test_that("analytical grad_ll_beta_rho matches numerical gradient", {
  set.seed(4821)
  n <- 20
  tau <- seq(0, 7, length.out = n)
  x_b0 <- cbind(1)
  x_b1 <- cbind(1)
  x_b2 <- cbind(1)
  x_om  <- cbind(1, rbinom(n, 1, 0.5))
  x_rho <- cbind(1, rnorm(n, 0, 0.3))
  y <- rnorm(n, 5, 1)

  beta_b0  <- 5.0
  beta_b1  <- -0.4
  beta_b2  <- 1.3
  beta_om  <- c(3.0, 0.5)
  beta_rho <- c(4.0, -0.3)
  sigma    <- 0.5

  analytic <- grad_ll_beta_rho(
    y, tau, x_b0, x_b1, x_b2, x_om, x_rho,
    beta_b0, beta_b1, beta_b2, beta_om, beta_rho, sigma
  )

  ll_fn <- function(brho) {
    ll_smoothbp(y, tau, x_b0, x_b1, x_b2, x_om, x_rho,
                beta_b0, beta_b1, beta_b2, beta_om, brho, sigma)
  }
  numerical <- numerical_grad(ll_fn, beta_rho)

  rel_err <- abs(analytic - numerical) / pmax(abs(numerical), 1e-12)
  expect_true(
    all(rel_err < 1e-4),
    label = sprintf(
      "grad_beta_rho: analytic = [%s], numerical = [%s], rel_err = [%s]",
      paste(round(analytic, 6), collapse = ", "),
      paste(round(numerical, 6), collapse = ", "),
      paste(formatC(rel_err, format = "e", digits = 2), collapse = ", ")
    )
  )
})

test_that("grad_log_prior matches numerical gradient", {
  values <- c(3.0, 0.5)
  means  <- c(0.0, 0.0)
  sds    <- c(2.0, 10.0)

  analytic <- grad_log_prior(values, means, sds)

  lp_fn <- function(v) {
    sum(dnorm(v, means, sds, log = TRUE))
  }
  numerical <- numerical_grad(lp_fn, values)

  rel_err <- abs(analytic - numerical) / pmax(abs(numerical), 1e-12)
  expect_true(
    all(rel_err < 1e-4),
    label = sprintf(
      "grad_prior: analytic = [%s], numerical = [%s], rel_err = [%s]",
      paste(round(analytic, 6), collapse = ", "),
      paste(round(numerical, 6), collapse = ", "),
      paste(formatC(rel_err, format = "e", digits = 2), collapse = ", ")
    )
  )
})

test_that("gradients match at a second, distant parameter configuration", {
  set.seed(7203)
  n <- 30
  tau <- seq(0, 10, length.out = n)
  x_b0 <- cbind(1)
  x_b1 <- cbind(1)
  x_b2 <- cbind(1)
  x_om  <- cbind(1, runif(n, -1, 1), rbinom(n, 1, 0.3))
  x_rho <- cbind(1, rnorm(n))
  y <- rnorm(n, 3, 2)

  beta_b0  <- 2.0
  beta_b1  <- 0.8
  beta_b2  <- -1.2
  beta_om  <- c(5.0, -1.0, 0.3)
  beta_rho <- c(1.5, 0.7)
  sigma    <- 1.5

  # -- beta_om --
  analytic_om <- grad_ll_beta_om(
    y, tau, x_b0, x_b1, x_b2, x_om, x_rho,
    beta_b0, beta_b1, beta_b2, beta_om, beta_rho, sigma
  )
  numerical_om <- numerical_grad(
    function(bom) ll_smoothbp(y, tau, x_b0, x_b1, x_b2, x_om, x_rho,
                              beta_b0, beta_b1, beta_b2, bom, beta_rho, sigma),
    beta_om
  )
  rel_err_om <- abs(analytic_om - numerical_om) / pmax(abs(numerical_om), 1e-12)
  expect_true(all(rel_err_om < 1e-4), label = "grad_beta_om (alt config)")

  # -- beta_rho --
  analytic_rho <- grad_ll_beta_rho(
    y, tau, x_b0, x_b1, x_b2, x_om, x_rho,
    beta_b0, beta_b1, beta_b2, beta_om, beta_rho, sigma
  )
  numerical_rho <- numerical_grad(
    function(brho) ll_smoothbp(y, tau, x_b0, x_b1, x_b2, x_om, x_rho,
                               beta_b0, beta_b1, beta_b2, beta_om, brho, sigma),
    beta_rho
  )
  rel_err_rho <- abs(analytic_rho - numerical_rho) / pmax(abs(numerical_rho), 1e-12)
  expect_true(all(rel_err_rho < 1e-4), label = "grad_beta_rho (alt config)")
})

test_that("compiled HMC sampler produces non-degenerate acceptance on covariate model", {
  skip_on_cran()

  # Simulate a two-group dataset where omega varies by group.
  set.seed(3391)
  make_row <- function(subj, group, omega_g) {
    tj <- seq(0, 6, length.out = 8)
    d  <- tj - omega_g
    s  <- sigmoid(d * 4)
    mu <- 5 + (-0.4) * d + 1.3 * d * s
    data.frame(subject = subj, group = group, tau = tj,
               y = mu + rnorm(8, 0, 0.4))
  }
  dat <- do.call(rbind, c(
    lapply(1:15, function(j) make_row(paste0("A", j), "A", 2.5)),
    lapply(1:15, function(j) make_row(paste0("B", j), "B", 4.0))
  ))
  dat$subject <- factor(dat$subject)

  # Fit with covariate in omega — triggers HMC path (p_om = 2).
  fit <- smoothbp(
    formula = y ~ tau,
    b0    = ~ 1 + (1 | subject),
    b1    = ~ 1,
    b2    = ~ 1,
    omega = ~ 1 + group,
    rho   = ~ 1,
    data  = dat,
    priors = smoothbp_priors(omega = prior_normal(3, 2, lb = 0, ub = 6)),
    chains = 2L, iter = 500L, warmup = 250L,
    seed   = 3391L, .verbose = FALSE
  )

  # Extract omega draws and check they are not stuck (ESS > 0).
  om_draws <- posterior::subset_draws(
    fit$draws, variable = c("omega_(Intercept)", "omega_groupB")
  )
  ess <- posterior::summarise_draws(om_draws, "ess_bulk")$ess_bulk

  # With a correct gradient we expect ESS >> 1 even in 250 post-warmup draws.
  # A gradient bug would give acceptance ~ 0 and ESS ~ 1–2.
  expect_true(
    all(ess > 10),
    label = sprintf("omega ESS_bulk = [%s]; expect > 10 if HMC gradient is correct",
                    paste(round(ess, 1), collapse = ", "))
  )
})
