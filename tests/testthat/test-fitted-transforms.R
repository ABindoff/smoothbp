# Tests for fitted() with in-formula transforms
# Covers: scale(), log(), I(), and constant-column newdata

test_that("fitted() works with scale() in b0 formula and constant newdata", {
  set.seed(42)
  subj <- factor(rep(1:10, each = 5))
  tau0 <- rep(seq(0, 6, length.out = 5), 10)
  grp  <- rep(sample(c("A", "B"), 10, TRUE), each = 5)
  x1   <- rep(rnorm(10, 50, 10), each = 5)
  y0   <- rnorm(50, 5 + -0.3 * tau0 + 0.01 * x1, 0.5)
  d    <- data.frame(subject = subj, tau = tau0, grp = grp, y = y0, x1 = x1)

  invisible(capture.output(
    fit <- smoothbp_ss(y ~ tau,
      b0 = ~ 1 + grp + scale(x1) + (1 | subject),
      b1 = ~ 1 + grp,
      deltas = list(~ 1 + grp),
      omega  = list(~ 1 + grp),
      rho    = list(~ 1),
      data   = d,
      b1_spike = TRUE,
      chains = 1L, iter = 100L, warmup = 50L,
      seed = 1L, .verbose = FALSE
    )
  ))

  # Constant x1 column (at its mean) — previously caused NaN → 0-row matrix

  nd <- expand.grid(grp = c("A", "B"), tau = seq(0, 6, length.out = 5),
                    x1 = mean(d[["x1"]]))
  res <- fitted(fit, newdata = nd)
  expect_equal(nrow(res), nrow(nd))
  expect_true(all(is.finite(res$fitted_mean)))
})

test_that("fitted() works with log() in b0 formula", {
  set.seed(42)
  subj <- factor(rep(1:10, each = 5))
  tau0 <- rep(seq(0, 6, length.out = 5), 10)
  grp  <- rep(sample(c("A", "B"), 10, TRUE), each = 5)
  x1   <- rep(runif(10, 10, 100), each = 5)
  y0   <- rnorm(50, 5 + -0.3 * tau0 + 0.5 * log(x1), 0.5)
  d    <- data.frame(subject = subj, tau = tau0, grp = grp, y = y0, x1 = x1)

  invisible(capture.output(
    fit <- smoothbp_ss(y ~ tau,
      b0 = ~ 1 + grp + log(x1) + (1 | subject),
      b1 = ~ 1 + grp,
      deltas = list(~ 1 + grp),
      omega  = list(~ 1 + grp),
      rho    = list(~ 1),
      data   = d,
      b1_spike = TRUE,
      chains = 1L, iter = 100L, warmup = 50L,
      seed = 1L, .verbose = FALSE
    )
  ))

  nd <- expand.grid(grp = c("A", "B"), tau = seq(0, 6, length.out = 5), x1 = 50)
  res <- fitted(fit, newdata = nd)
  expect_equal(nrow(res), nrow(nd))
  expect_true(all(is.finite(res$fitted_mean)))
})

test_that("$data stores the original training data frame", {
  set.seed(42)
  subj <- factor(rep(1:10, each = 5))
  tau0 <- rep(seq(0, 6, length.out = 5), 10)
  grp  <- rep(sample(c("A", "B"), 10, TRUE), each = 5)
  x1   <- rep(rnorm(10, 50, 10), each = 5)
  y0   <- rnorm(50, 5 + -0.3 * tau0 + 0.01 * x1, 0.5)
  d    <- data.frame(subject = subj, tau = tau0, grp = grp, y = y0, x1 = x1)

  invisible(capture.output(
    fit <- smoothbp_ss(y ~ tau,
      b0 = ~ 1 + grp + scale(x1) + (1 | subject),
      b1 = ~ 1,
      deltas = list(~ 1),
      omega  = list(~ 1),
      rho    = list(~ 1),
      data   = d,
      chains = 1L, iter = 100L, warmup = 50L,
      seed = 1L, .verbose = FALSE
    )
  ))

  expect_s3_class(fit[["data"]], "data.frame")
  expect_equal(nrow(fit[["data"]]), nrow(d))
  expect_true(all(c("subject", "tau", "grp", "y", "x1") %in% names(fit[["data"]])))
})

test_that("train_model_frames stores scale attributes from training data", {
  set.seed(42)
  subj <- factor(rep(1:10, each = 5))
  tau0 <- rep(seq(0, 6, length.out = 5), 10)
  x1   <- rep(rnorm(10, 50, 10), each = 5)
  y0   <- rnorm(50, 5 + -0.3 * tau0 + 0.01 * x1, 0.5)
  d    <- data.frame(subject = subj, tau = tau0, y = y0, x1 = x1)

  invisible(capture.output(
    fit <- smoothbp(y ~ tau,
      b0 = ~ 1 + scale(x1) + (1 | subject),
      b1 = ~ 1,
      deltas = list(~ 1),
      omega  = list(~ 1),
      rho    = list(~ 1),
      data   = d,
      chains = 1L, iter = 100L, warmup = 50L,
      seed = 1L, .verbose = FALSE
    )
  ))

  tmf <- fit[["train_model_frames"]]
  expect_type(tmf, "list")
  expect_s3_class(tmf$b0, "data.frame")

  # The terms should carry predvars with scale(x1, center = ..., scale = ...)
  pv <- attr(terms(tmf$b0), "predvars")
  expect_true(!is.null(pv))
  pv_str <- deparse(pv)
  expect_true(grepl("center", pv_str))
  expect_true(grepl("scale", pv_str))
})
