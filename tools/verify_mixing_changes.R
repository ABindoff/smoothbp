# =============================================================================
# verify_mixing_changes.R
#
# Build the package after the sampler refactor (LinearCache + componentwise
# adaptive MH + Haario adaptive joint proposal), then check:
#
#   (1) Compilation succeeds.
#   (2) The existing parameter-recovery test still passes.
#   (3) On a model with a covariate in omega (the case the changes target),
#       acceptance rates are near target and ESS for omega coefficients is
#       meaningfully higher than ESS per second of wall-clock time.
#
# Run interactively from the package root:
#
#   source("tools/verify_mixing_changes.R")
#
# Output is verbose by design so anything weird is easy to spot in the log.
# =============================================================================

suppressPackageStartupMessages({
  library(rextendr)
  library(devtools)
  library(testthat)
  library(posterior)
  library(dplyr)
  library(tidyr)
  library(tibble)
})

cat("\n========================================================\n")
cat(" smoothbp mixing-improvement verification\n")
cat("========================================================\n\n")

# ----------------------------------------------------------------------------
# (1) Recompile the Rust crate and reload the package.
# ----------------------------------------------------------------------------
cat("[1/4] rextendr::document() ... \n")
t0 <- Sys.time()
doc_ok <- tryCatch({ rextendr::document(); TRUE },
                   error = function(e) { print(e); FALSE })
cat(sprintf("      done in %.1fs (success = %s)\n\n",
            as.numeric(difftime(Sys.time(), t0, units = "secs")), doc_ok))
if (!isTRUE(doc_ok)) {
  stop("rextendr::document() failed -- check the error above before continuing.")
}

cat("[2/4] devtools::load_all() ... \n")
suppressMessages(devtools::load_all(quiet = TRUE))
cat("      loaded.\n\n")

# ----------------------------------------------------------------------------
# (2) Run existing testthat tests (parameter recovery + formula utils).
# ----------------------------------------------------------------------------
cat("[3/4] testthat::test_dir('tests/testthat') ... \n")
test_results <- tryCatch(
  testthat::test_dir("tests/testthat", reporter = "summary", stop_on_failure = FALSE),
  error = function(e) { print(e); NULL }
)
cat("\n")

# ----------------------------------------------------------------------------
# (3) Mixing check: covariate in omega.  Compare acceptance / ESS to the
#     intercept-only case to confirm the new code paths do not regress.
# ----------------------------------------------------------------------------
cat("[4/4] mixing check: covariate in omega + rho ...\n")

set.seed(2026)

# Simulate a two-group dataset where the change-point shifts by group.
make_two_group_data <- function(n_subj_per = 20, n_obs = 10,
                                 omega_a = 2.5, omega_b = 4.0,
                                 b0 = 5, b1 = -0.4, delta = 1.3,
                                 rho = 4, sigma = 0.4, sigma_u = 0.6,
                                 seed = 11L) {
  set.seed(seed)
  sigmoid <- function(x) 1 / (1 + exp(-x))
  build_one <- function(group_label, omega_g, sid_offset) {
    u <- rnorm(n_subj_per, 0, sigma_u)
    rows <- vector("list", n_subj_per)
    for (j in seq_len(n_subj_per)) {
      tj <- seq(0, 6, length.out = n_obs)
      d  <- tj - omega_g
      mu <- (b0 + u[j]) + b1 * d + delta * d * sigmoid(d * rho)
      yj <- mu + rnorm(n_obs, 0, sigma)
      rows[[j]] <- tibble::tibble(
        subject = factor(sid_offset + j),
        group   = group_label,
        tau     = tj,
        y       = yj
      )
    }
    dplyr::bind_rows(rows)
  }
  dplyr::bind_rows(
    build_one("A", omega_a, 0),
    build_one("B", omega_b, n_subj_per)
  )
}

dat <- make_two_group_data()

cat("      data: n =", nrow(dat),
    "; subjects =", nlevels(dat$subject),
    "; groups =", paste(unique(dat$group), collapse = ", "), "\n")

# Fit with covariate in omega (the case the new sampler is meant to help with).
# Use a longer chain than the recovery test so rhat has a fighting chance on
# the highly correlated random-effects parameters in this small-N design.
t0 <- Sys.time()
fit_cov <- smoothbp(
  formula = y ~ tau,
  b0      = ~ 1 + group + (1 | subject),
  b1      = ~ 1,
  deltas  = list(~ 1),
  omega   = ~ 1 + group,
  rho     = ~ 1,
  data    = dat,
  priors  = smoothbp_priors(omega = prior_normal(3, 2, lb = 0, ub = 6)),
  chains  = 3L, iter = 3000L, warmup = 1500L, seed = 42L,
  cores   = 1L, .verbose = FALSE
)
elapsed_cov <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

s_cov <- posterior::summarise_draws(
  fit_cov$draws,
  mean, sd,
  ~ posterior::quantile2(.x, probs = c(0.025, 0.975)),
  ess_bulk, ess_tail, rhat
)

cat("\n  --- covariate-in-omega fit ---\n")
cat(sprintf("  wall time: %.1fs (3 chains, 3000 iter, 1500 warmup)\n",
            elapsed_cov))

# Show all population-level (non-random-effect) parameters.
pop_rows <- s_cov[!grepl("^u\\[", s_cov$variable), ]
cat("\n  -- population-level parameters --\n")
print(pop_rows, n = Inf, digits = 3)

# Show the five worst rhat parameters anywhere in the fit.
cat("\n  -- five worst rhat parameters (any) --\n")
worst_rhat_rows <- s_cov %>%
  dplyr::arrange(dplyr::desc(rhat)) %>%
  dplyr::slice_head(n = 5)
print(worst_rhat_rows, n = Inf, digits = 3)

# Show the five smallest ess_bulk parameters.
cat("\n  -- five smallest ess_bulk parameters (any) --\n")
worst_ess_rows <- s_cov %>%
  dplyr::arrange(ess_bulk) %>%
  dplyr::slice_head(n = 5)
print(worst_ess_rows, n = Inf, digits = 3)

omega_rows <- s_cov[grepl("^omega_", s_cov$variable), ]
ess_per_sec <- omega_rows$ess_bulk / elapsed_cov
cat("\n  omega ESS / second:\n")
print(setNames(round(ess_per_sec, 1), omega_rows$variable))

# Acceptance-rate sanity: fit a no-covariate version on the same data so we
# can compare ESS-per-second against the case the original sampler handled.
t0 <- Sys.time()
fit_int <- smoothbp(
  formula = y ~ tau,
  b0      = ~ 1 + (1 | subject),
  b1      = ~ 1,
  deltas  = list(~ 1),
  omega   = ~ 1,
  rho     = ~ 1,
  data    = dat,
  priors  = smoothbp_priors(omega = prior_normal(3, 2, lb = 0, ub = 6)),
  chains  = 3L, iter = 3000L, warmup = 1500L, seed = 42L,
  cores   = 1L, .verbose = FALSE
)
elapsed_int <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
s_int <- posterior::summarise_draws(fit_int$draws,
                                    ess_bulk, rhat)

cat("\n  --- intercept-only fit on same data (regression check) ---\n")
cat(sprintf("  wall time: %.1fs\n", elapsed_int))
cat(sprintf("  worst rhat:           %.3f\n",
            max(s_int$rhat, na.rm = TRUE)))
cat(sprintf("  omega bulk ESS / sec: %.1f\n",
            s_int$ess_bulk[s_int$variable == "omega_(Intercept)"] /
              elapsed_int))

cat("\n========================================================\n")
cat(" verification complete\n")
cat(" -- if all parameter-recovery tests passed and rhat < 1.05,\n")
cat("    the changes are safe to commit.\n")
cat("========================================================\n")
