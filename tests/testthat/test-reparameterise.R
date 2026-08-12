library(smoothbp)

# Regression test for the omega_translation_step non-centred bug.
#
# state$beta_om always holds CENTRED random-effect values; omega_vec() feeds them
# straight into the likelihood. omega_translation_step used to rescale them by
# sigma_re whenever reparameterise = "omega", which broke the likelihood
# invariance of the (omega_bar + c, u_s - c) move while still accepting it with
# probability one. The two parameterisations must target the same posterior, so
# their marginals have to agree on the same data.

test_that("centred and non-centred parameterisations agree on the same posterior", {
  testthat::skip_on_cran()
  set.seed(20260811)

  J <- 8L; N <- 14L
  subj <- rep(seq_len(J), each = N)
  tau  <- rep(seq(-4, 10, length.out = N), times = J)
  omega_true <- 3
  sigma_re   <- 0.9
  u <- rnorm(J, 0, sigma_re)
  d <- tau - (omega_true + u[subj])
  y <- 5 + 0.1 * d + 1.2 * d * plogis(3 * d) + rnorm(J * N, 0, 0.35)
  dat <- data.frame(subject = factor(subj), tau = tau, y = y)

  pri <- smoothbp_priors(
    b0     = prior_normal(5, 2),
    b1     = prior_normal(0, 0.5),
    deltas = prior_normal(0, 2),
    omega  = prior_normal(3, 1.5, lb = -4, ub = 10),
    sigma_re_om = prior_invgamma(3, 2)
  )

  common <- list(
    formula = y ~ tau, b0 = ~ 1, b1 = ~ 1,
    omega = list(~ (1 | subject)), deltas = list(~ 1),
    rho = list(fixed(3)), data = dat, priors = pri,
    chains = 2L, iter = 8000L, warmup = 3000L,
    target_accept = 0.95, .verbose = FALSE
  )

  fit_c  <- do.call(smoothbp, c(common, list(seed = 1L, reparameterise = "none")))
  fit_nc <- do.call(smoothbp, c(common, list(seed = 1L, reparameterise = "omega")))

  pars <- c("omega1_(Intercept)", "sigma_re_omega1", "delta1_(Intercept)", "sigma")

  for (p in pars) {
    a <- as.numeric(posterior::extract_variable(fit_c$draws, p))
    b <- as.numeric(posterior::extract_variable(fit_nc$draws, p))

    # Compare on the scale of the posterior's own spread: a broken invariance
    # shifts the location by many posterior SDs, so this is a loose but decisive
    # threshold rather than a Monte Carlo tolerance.
    pooled_sd <- sqrt((var(a) + var(b)) / 2)
    expect_lt(abs(mean(a) - mean(b)) / pooled_sd, 0.25,
              label = paste0("standardised mean difference for ", p))
    expect_lt(abs(sd(a) - sd(b)) / pooled_sd, 0.25,
              label = paste0("standardised SD difference for ", p))
  }
})

test_that("the omega translation move preserves subject-level change-points", {
  testthat::skip_on_cran()
  set.seed(99)

  # sigma_re is large relative to the omega prior SD, so the translation step is
  # doing real work: if it shifted the RE columns by anything other than c, the
  # subject-level omegas (omega_bar + u_s) would drift and the fit would not
  # recover the generating values under either parameterisation.
  J <- 6L; N <- 16L
  subj <- rep(seq_len(J), each = N)
  tau  <- rep(seq(-5, 11, length.out = N), times = J)
  u <- c(-1.6, -0.9, -0.1, 0.4, 1.1, 1.8)
  d <- tau - (4 + u[subj])
  y <- 2 + 0.2 * d + 1.5 * d * plogis(3 * d) + rnorm(J * N, 0, 0.3)
  dat <- data.frame(subject = factor(subj), tau = tau, y = y)

  pri <- smoothbp_priors(
    b0 = prior_normal(2, 3), b1 = prior_normal(0, 1), deltas = prior_normal(0, 3),
    omega = prior_normal(4, 2, lb = -5, ub = 11), sigma_re_om = prior_invgamma(3, 3)
  )

  for (rp in c("none", "omega")) {
    fit <- smoothbp(y ~ tau, b0 = ~ 1, b1 = ~ 1, omega = list(~ (1 | subject)),
                    deltas = list(~ 1), rho = list(fixed(3)), data = dat,
                    priors = pri, chains = 2L, iter = 8000L, warmup = 3000L,
                    seed = 7L, target_accept = 0.95, reparameterise = rp,
                    .verbose = FALSE)
    om <- as.numeric(posterior::extract_variable(fit$draws, "omega1_(Intercept)"))
    expect_gt(mean(om), 4 - 1.5, label = paste0("omega lower, reparameterise=", rp))
    expect_lt(mean(om), 4 + 1.5, label = paste0("omega upper, reparameterise=", rp))
  }
})
