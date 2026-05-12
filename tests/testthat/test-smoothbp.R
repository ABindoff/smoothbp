# Parameter recovery test
#
# Simulates data from the known model using simulate_smoothbp(), fits the
# model, then asserts that the 95% posterior interval contains the true value
# for every scalar population-level parameter.
#
# Runtime: ~30–60 s.  Skipped on CRAN.

test_that("smoothbp recovers parameters on simulated data", {
  skip_on_cran()

  dat <- simulate_smoothbp(
    n_subj  = 30, n_obs = 8,
    b0 = 5.0, b1 = -0.5, b2 = 1.5,
    omega = 3.0, rho = 4.0,
    sigma = 0.5, sigma_u = 1.0,
    seed = 7L
  )

  tp <- attr(dat, "true_params")

  fit <- smoothbp(
    formula = y ~ tau,
    b0      = ~ 1 + (1 | subject),
    b1      = ~ 1,
    deltas  = list(~ 1),
    omega   = list(~ 1),
    rho     = list(~ 1),
    data    = dat,
    priors  = smoothbp_priors(omega = prior_normal(3, 2, lb = 0)),
    chains  = 2L,
    iter    = 1500L,
    warmup  = 750L,
    seed    = 7L,
    .verbose = FALSE
  )

  s <- summary(fit)

  # Map true_params names -> fit parameter names
  param_map <- c(
    b0      = "b0_(Intercept)",
    b1      = "b1_(Intercept)",
    b2      = "delta1_(Intercept)",
    omega   = "omega1_(Intercept)",
    rho     = "rho1_(Intercept)",
    sigma   = "sigma",
    sigma_u = "sigma_u"
  )

  for (nm in names(param_map)) {
    pname <- param_map[[nm]]
    truth <- tp[[nm]]
    row   <- s[s$variable == pname, ]

    # Skip parameters not present in the model (e.g. sigma_u in fixed-only fit)
    if (nrow(row) == 0) next

    expect_true(
      truth >= row$Q2.5 & truth <= row$Q97.5,
      label = sprintf(
        "%s (%s): true=%.3f, 95%% CI=[%.3f, %.3f]",
        nm, pname, truth, row$Q2.5, row$Q97.5
      )
    )
  }
})
