# =============================================================================
# sbc_recovery_diagnostic.R
#
# Localises the b0 / b1 / sigma_y SBC failures: are their POSTERIORS actually
# biased or mis-dispersed (a real issue), or do they cover correctly while only
# the ranks fail (a harness/extraction subtlety)? Recovery + interval coverage
# answers this directly, which ranks alone cannot.
#
# rho is FIXED at truth (the clean, divergence-free regime). Every estimated
# parameter is drawn from its prior and the fit uses those same priors, so this
# is a valid SBC setup. ~150 reps is plenty to see a systematic bias.
#
# Run from package root AFTER devtools::load_all():
#   source("tools/sbc_recovery_diagnostic.R")
# Then also inspect warnings():  warnings()
# =============================================================================

suppressPackageStartupMessages({ library(devtools); library(posterior) })
suppressMessages(devtools::load_all(quiet = TRUE))

REPS <- 150L
J <- 6L; N <- 18L; ITER <- 3000L; WARM <- 500L; CH <- 2L; TA <- 0.99
TAU_LO <- -5; TAU_HI <- 11; RHO_FIX <- 3.0
set.seed(20260617L)

# Priors (identical to sbc_ecdf_bands.R; plain scalars, no named vectors).
P <- list(b0 = c(5, 1), b1 = c(0, 0.3), delta = c(0, 1.5),
          omega = c(3, 1, 0, 10), sigma = c(10, 1.44), sigre = c(5, 4))
PRIORS <- smoothbp_priors(
  b0 = prior_normal(P$b0[1], P$b0[2]), b1 = prior_normal(P$b1[1], P$b1[2]),
  deltas = prior_normal(P$delta[1], P$delta[2]),
  omega = prior_normal(P$omega[1], P$omega[2], lb = P$omega[3], ub = P$omega[4]),
  sigma = prior_invgamma(P$sigma[1], P$sigma[2]),
  sigma_re_om = prior_invgamma(P$sigre[1], P$sigre[2]))

rtrunc   <- function(m, s, lb = -Inf, ub = Inf) { repeat { x <- rnorm(1, m, s); if (x > lb && x < ub) return(x) } }
draw_sd  <- function(shape, scale) sqrt(1 / rgamma(1, shape = shape, rate = scale))

# Tested columns -> the truth they should recover.
COLS <- c(b0 = "b0_(Intercept)", b1 = "b1_(Intercept)", sigma = "sigma",
          omega = "omega1_(Intercept)", delta = "delta1_(Intercept)",
          sigre = "sigma_re_omega1")

subj <- rep(seq_len(J), each = N)
tau  <- rep(seq(TAU_LO, TAU_HI, length.out = N), times = J)
rows <- vector("list", REPS); printed_names <- FALSE

for (r in seq_len(REPS)) {
  truth <- c(
    b0 = rnorm(1, P$b0[1], P$b0[2]), b1 = rnorm(1, P$b1[1], P$b1[2]),
    delta = rnorm(1, P$delta[1], P$delta[2]),
    omega = rtrunc(P$omega[1], P$omega[2], P$omega[3], P$omega[4]),
    sigma = draw_sd(P$sigma[1], P$sigma[2]), sigre = draw_sd(P$sigre[1], P$sigre[2]))
  u  <- rnorm(J, 0, truth["sigre"]); d <- tau - (truth["omega"] + u[subj])
  mu <- truth["b0"] + truth["b1"] * d + truth["delta"] * d * plogis(RHO_FIX * d)
  y  <- mu + rnorm(J * N, 0, truth["sigma"])
  dat <- data.frame(subject = factor(subj), tau = tau, y = y)

  fit <- tryCatch(smoothbp(y ~ tau, b0 = ~ 1, b1 = ~ 1, omega = list(~ (1 | subject)),
                           deltas = list(~ 1), rho = list(fixed(RHO_FIX)), data = dat,
                           priors = PRIORS, chains = CH, iter = ITER, warmup = WARM,
                           seed = 5000L + r, target_accept = TA, .verbose = FALSE),
                  error = function(e) NULL)
  if (is.null(fit)) next
  dm <- posterior::as_draws_matrix(fit$draws)
  if (!printed_names) { cat("draws columns:\n"); print(colnames(dm)); printed_names <- TRUE }

  row <- list(rep = r)
  for (nm in names(COLS)) {
    col <- COLS[[nm]]
    if (!col %in% colnames(dm)) next
    x <- as.numeric(dm[, col]); q <- quantile(x, c(.05, .25, .5, .75, .95), names = FALSE)
    row[[paste0(nm, "_truth")]] <- unname(truth[nm])
    row[[paste0(nm, "_mean")]]  <- mean(x)
    row[[paste0(nm, "_q05")]] <- q[1]; row[[paste0(nm, "_q25")]] <- q[2]
    row[[paste0(nm, "_q75")]] <- q[4]; row[[paste0(nm, "_q95")]] <- q[5]
  }
  rows[[r]] <- as.data.frame(row)
  if (r %% 30L == 0L) cat(sprintf("  recovery %d/%d\n", r, REPS))
}
df <- do.call(rbind, rows)

cat("\n[recovery] bias and interval coverage (rho fixed, valid SBC):\n")
cat(sprintf("  %-8s %8s %8s %8s %8s %8s\n", "param", "bias", "bias/sd", "cov50", "cov90", "n"))
prior_sd <- c(b0 = P$b0[2], b1 = P$b1[2], delta = P$delta[2], omega = P$omega[2],
              sigma = NA, sigre = NA)
for (nm in names(COLS)) {
  tr <- df[[paste0(nm, "_truth")]]; mn <- df[[paste0(nm, "_mean")]]
  if (is.null(tr)) next
  ok <- !is.na(tr) & !is.na(mn)
  bias  <- mean(mn[ok] - tr[ok])
  cov50 <- mean(tr[ok] >= df[[paste0(nm, "_q25")]][ok] & tr[ok] <= df[[paste0(nm, "_q75")]][ok])
  cov90 <- mean(tr[ok] >= df[[paste0(nm, "_q05")]][ok] & tr[ok] <= df[[paste0(nm, "_q95")]][ok])
  cat(sprintf("  %-8s %8.3f %8s %8.2f %8.2f %8d\n", nm, bias,
              ifelse(is.na(prior_sd[nm]), "-", sprintf("%.2f", bias / prior_sd[nm])),
              cov50, cov90, sum(ok)))
}
cat("\nReading it:\n",
    " - good coverage (cov50~0.5, cov90~0.9) but failing ranks -> harness/extraction subtlety\n",
    " - biased centre or wrong-width intervals (esp. scaling with prior tightness) -> real issue\n", sep = "")
