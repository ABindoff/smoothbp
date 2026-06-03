# Tests for update(), recovery_plot(), hypothesis(), and bayes_factor()
#
# Input-validation tests use a lightweight mock draws object so they run
# without the sampler.  Functional tests call the sampler with a very small
# single-subject setup (chains = 1, iter = 300) and are skipped on CRAN.

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

.mock_fit <- function(param_names = c("b1_(Intercept)", "b2_(Intercept)"),
                      n_draws = 200L) {
  set.seed(42)
  mat <- matrix(rnorm(n_draws * length(param_names)),
                nrow = n_draws,
                dimnames = list(NULL, param_names))
  draws <- posterior::as_draws_df(as.data.frame(mat))
  structure(list(draws = draws, formula = y ~ tau), class = "smoothbp_fit")
}

.minimal_sim <- function(seed = 99L) {
  simulate_smoothbp(
    n_subj = 1, n_obs = 20,
    b0 = 5, b1 = -0.3, delta = 1.2,
    omega = 3, rho = 4, sigma = 0.4, sigma_u = 0,
    seed = seed
  )
}

.minimal_fit <- function(dat, seed = 99L) {
  smoothbp(
    formula = y ~ tau,
    b0 = ~ 1, b1 = ~ 1,
    deltas = list(~ 1), omega = list(~ 1), rho = list(~ 1),
    data = dat,
    priors = smoothbp_priors(omega = prior_normal(3, 2, lb = 0)),
    chains = 1L, iter = 300L, warmup = 150L,
    seed = seed, .verbose = FALSE
  )
}

# ===========================================================================
# hypothesis() — input validation (no sampler needed)
# ===========================================================================

test_that("hypothesis errors on non-character hypotheses", {
  fit <- .mock_fit()
  expect_error(hypothesis(fit, 123),       "`hypotheses` must be a non-empty character vector")
  expect_error(hypothesis(fit, character(0)), "`hypotheses` must be a non-empty character vector")
})

test_that("hypothesis errors on point-null (==) hypotheses", {
  fit <- .mock_fit()
  expect_error(
    hypothesis(fit, "b1_(Intercept) == 0"),
    "Point-null hypotheses"
  )
})

test_that("hypothesis errors on invalid ci", {
  fit <- .mock_fit()
  expect_error(hypothesis(fit, "b1_(Intercept) > 0", ci = 0),  "ci")
  expect_error(hypothesis(fit, "b1_(Intercept) > 0", ci = 1),  "ci")
  expect_error(hypothesis(fit, "b1_(Intercept) > 0", ci = -0.5), "ci")
})

test_that("hypothesis errors on unknown parameter names", {
  fit <- .mock_fit()
  expect_error(
    hypothesis(fit, "nonexistent_param > 0"),
    "Could not evaluate hypothesis"
  )
})

test_that("hypothesis returns correct structure for a directional test", {
  fit <- .mock_fit(param_names = c("b1_(Intercept)", "b2_(Intercept)"),
                   n_draws = 500L)
  res <- hypothesis(fit, "b1_(Intercept) > 0")

  expect_s3_class(res, "smoothbp_hypothesis")
  expect_s3_class(res, "data.frame")
  expect_equal(nrow(res), 1L)
  expect_true(all(c("Hypothesis", "Estimate", "Est.Error",
                    "CI.lower", "CI.upper", "P(H)", "Evid.Ratio", "Star") %in% names(res)))
  expect_equal(res$Hypothesis, "b1_(Intercept) > 0")
  expect_true(res[["P(H)"]] >= 0 & res[["P(H)"]] <= 1)
  expect_true(res$Evid.Ratio >= 0)
})

test_that("hypothesis handles multiple hypotheses in one call", {
  fit <- .mock_fit(param_names = c("b1_(Intercept)", "b2_(Intercept)"),
                   n_draws = 500L)
  hyps <- c("b1_(Intercept) > 0",
             "b2_(Intercept) < 0",
             "b1_(Intercept) - b2_(Intercept) > 0")
  res <- hypothesis(fit, hyps)

  expect_equal(nrow(res), 3L)
  expect_equal(res$Hypothesis, hyps)
})

test_that("hypothesis handles a contrast expression (no comparison operator)", {
  fit <- .mock_fit(param_names = c("b1_(Intercept)", "b2_(Intercept)"),
                   n_draws = 500L)
  res <- hypothesis(fit, "b1_(Intercept) - b2_(Intercept)")

  expect_equal(nrow(res), 1L)
  expect_equal(res$Hypothesis, "b1_(Intercept) - b2_(Intercept)")
})

test_that("hypothesis evidence ratio is Inf when P(H) = 1", {
  # Draw all positives so P(H > 0) = 1
  draws <- posterior::as_draws_df(data.frame(`z` = abs(rnorm(200)) + 10,
                                             check.names = FALSE))
  fit <- structure(list(draws = draws, formula = y ~ tau), class = "smoothbp_fit")
  res <- hypothesis(fit, "z > 0")
  expect_true(is.infinite(res$Evid.Ratio))
  expect_equal(res$Star, "***")
})

test_that("print.smoothbp_hypothesis runs without error", {
  fit <- .mock_fit()
  res <- hypothesis(fit, "b1_(Intercept) > 0")
  expect_output(print(res), "Hypothesis tests for smoothbp_fit")
})

# ===========================================================================
# recovery_plot() — input validation (no sampler needed)
# ===========================================================================

test_that("recovery_plot errors when fit is not a smoothbp_fit", {
  dat <- .minimal_sim()
  expect_error(recovery_plot(list(), dat), "`fit` must be a smoothbp_fit object")
})

test_that("recovery_plot errors when dat has no true_params attribute", {
  fit <- .mock_fit()
  dat_no_attr <- data.frame(y = 1:5, tau = 1:5)
  expect_error(recovery_plot(fit, dat_no_attr), "true_params")
})

test_that("recovery_plot errors on invalid level", {
  fit  <- .mock_fit()
  dat  <- .minimal_sim()
  expect_error(recovery_plot(fit, dat, level = 0),   "level")
  expect_error(recovery_plot(fit, dat, level = 1.5), "level")
})

# ===========================================================================
# Functional tests — require sampler (skipped on CRAN)
# ===========================================================================

test_that("update() returns a smoothbp_fit with the same structure", {
  skip_on_cran()

  dat <- .minimal_sim(seed = 101L)
  fit <- .minimal_fit(dat, seed = 101L)

  fit2 <- update(fit, .verbose = FALSE)

  expect_s3_class(fit2, "smoothbp_fit")
  expect_equal(fit2$iter,    fit$iter)
  expect_equal(fit2$chains,  fit$chains)
  expect_equal(fit2$warmup,  fit$warmup)
  expect_equal(deparse(fit2$formula), deparse(fit$formula))
})

test_that("update() respects an overridden iter argument", {
  skip_on_cran()

  dat  <- .minimal_sim(seed = 202L)
  fit  <- .minimal_fit(dat, seed = 202L)
  fit2 <- update(fit, iter = 400L, warmup = 200L, .verbose = FALSE)

  expect_equal(fit2$iter,   400L)
  expect_equal(fit2$warmup, 200L)
})

test_that("update() respects an overridden priors argument", {
  skip_on_cran()

  dat        <- .minimal_sim(seed = 303L)
  fit        <- .minimal_fit(dat, seed = 303L)
  new_priors <- smoothbp_priors(omega = prior_normal(4, 1, lb = 0))
  fit2       <- update(fit, priors = new_priors, .verbose = FALSE)

  expect_equal(fit2$priors$omega$mean, new_priors$omega$mean)
  expect_equal(fit2$priors$omega$sd,   new_priors$omega$sd)
})

test_that("recovery_plot() returns a ggplot for simulated data", {
  skip_on_cran()

  dat <- .minimal_sim(seed = 404L)
  fit <- .minimal_fit(dat, seed = 404L)

  p <- recovery_plot(fit, dat)
  expect_s3_class(p, "gg")
  expect_s3_class(p, "ggplot")
})

test_that("recovery_plot() honours the level argument", {
  skip_on_cran()

  dat <- .minimal_sim(seed = 505L)
  fit <- .minimal_fit(dat, seed = 505L)

  p90 <- recovery_plot(fit, dat, level = 0.90)
  expect_true(grepl("90%", p90$labels$title))
})

test_that("hypothesis() works on a real smoothbp_fit", {
  skip_on_cran()

  dat <- .minimal_sim(seed = 606L)
  fit <- .minimal_fit(dat, seed = 606L)

  res <- hypothesis(fit, c(
    "b1_(Intercept) < 0",
    "delta1_(Intercept) > 0",
    "omega1_(Intercept) > 0"
  ))

  expect_s3_class(res, "smoothbp_hypothesis")
  expect_equal(nrow(res), 3L)
  expect_true(all(res[["P(H)"]] >= 0 & res[["P(H)"]] <= 1))
  expect_equal(attr(res, "n_draws"), nrow(posterior::as_draws_df(fit$draws)))
})

# ===========================================================================
# bayes_factor() — requires bridgesampling package
# ===========================================================================

test_that("bayes_factor() compares two smoothbp_fit objects", {
  skip_on_cran()
  skip_if_not_installed("bridgesampling")

  dat  <- .minimal_sim(seed = 707L)
  fit1 <- .minimal_fit(dat, seed = 707L)
  fit2 <- update(fit1,
                 priors = smoothbp_priors(omega = prior_normal(2, 1, lb = 0)),
                 .verbose = FALSE)

  bf <- bayes_factor(fit1, fit2, method = "bridgesampling")

  expect_true(is.numeric(bf$bf))
  expect_length(bf$bf, 1L)
})
