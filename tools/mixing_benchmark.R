# =============================================================================
# mixing_benchmark.R
#
# Produces the random change-point mixing table reported in the manuscript
# (Section "Convergence and Calibration for Random Change-points").
#
# This exists because the published table credited tools/part3_validation.R,
# which runs 4 chains x 1000 post-warmup draws, while the table described
# 4 chains x 4000. The two are not the same run and the numbers were not
# reproducible from the cited script. Everything the table reports is now
# generated here, from one configuration stated in full, and written to disk.
#
# Run from the package root:
#   Rscript tools/mixing_benchmark.R
# or against a specific installed build:
#   SMOOTHBP_LIB=/path/to/lib Rscript tools/mixing_benchmark.R
# =============================================================================

lib <- Sys.getenv("SMOOTHBP_LIB")
if (nzchar(lib)) .libPaths(c(lib, .libPaths()))
suppressPackageStartupMessages({library(smoothbp); library(posterior)})
cat(sprintf("[mixing] smoothbp %s from %s\n",
            as.character(packageVersion("smoothbp")), find.package("smoothbp")))

OUT_DIR <- file.path("tools", "part3_results")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

# ---- Configuration (the "low prior fraction" regime) ------------------------
J          <- 20L     # groups
N_OBS      <- 15L     # observations per group
SIGMA_RE   <- 0.30    # between-group SD of the change-point
TRUE_OMEGA <- 3.0; TRUE_DELTA <- 1.5; TRUE_RHO <- 3.0; TRUE_SIGMA_Y <- 0.4
CHAINS     <- 4L
ITER       <- 5000L   # 4000 post-warmup per chain -> 16000 draws total
WARMUP     <- 1000L
TARGET_ACC <- 0.9
SEED_DATA  <- 2024L
SEED_FIT   <- 42L

set.seed(SEED_DATA)
u_j  <- rnorm(J, 0, SIGMA_RE)
subj <- rep(seq_len(J), each = N_OBS)
tau  <- rep(seq(0.5, 5.5, length.out = N_OBS), times = J)
d    <- tau - (TRUE_OMEGA + u_j[subj])
y    <- 5.0 + TRUE_DELTA * d * plogis(TRUE_RHO * d) + rnorm(J * N_OBS, 0, TRUE_SIGMA_Y)
dat  <- data.frame(subject = factor(subj), tau = tau, y = y)

t0 <- proc.time()["elapsed"]
fit <- smoothbp(y ~ tau, omega = list(~ 1 + (1 | subject)), deltas = list(~ 1),
                data = dat, priors = smoothbp_priors(omega = prior_normal(3.0, 1.5, lb = 0)),
                chains = CHAINS, iter = ITER, warmup = WARMUP, seed = SEED_FIT,
                target_accept = TARGET_ACC, .verbose = FALSE)
elapsed <- unname(proc.time()["elapsed"] - t0)

vars <- posterior::variables(fit$draws)
subj_vars <- grep("^omega1_subject", vars, value = TRUE)
if (!length(subj_vars)) subj_vars <- grep("^omega1_.*[0-9]$", vars, value = TRUE)

s <- posterior::summarise_draws(fit$draws)
pick <- function(v) s[s$variable == v, , drop = FALSE]

subj_rows <- s[s$variable %in% subj_vars, , drop = FALSE]
rows <- list(
  data.frame(parameter = "Subject omega_j (mean)",
             ess_bulk = round(mean(subj_rows$ess_bulk)),
             rhat = round(max(subj_rows$rhat), 2),
             n_params = nrow(subj_rows)),
  data.frame(parameter = "Population omega",
             ess_bulk = round(pick("omega1_(Intercept)")$ess_bulk),
             rhat = round(pick("omega1_(Intercept)")$rhat, 2), n_params = 1L),
  data.frame(parameter = "sigma_re",
             ess_bulk = round(pick("sigma_re_omega1")$ess_bulk),
             rhat = round(pick("sigma_re_omega1")$rhat, 2), n_params = 1L),
  data.frame(parameter = "delta",
             ess_bulk = round(pick("delta1_(Intercept)")$ess_bulk),
             rhat = round(pick("delta1_(Intercept)")$rhat, 2), n_params = 1L),
  data.frame(parameter = "sigma",
             ess_bulk = round(pick("sigma")$ess_bulk),
             rhat = round(pick("sigma")$rhat, 2), n_params = 1L)
)
tab <- do.call(rbind, rows)
tab$ess_per_sec <- round(tab$ess_bulk / elapsed, 1)

blk <- fit$n_divergent_by_block
cat(sprintf("\nsmoothbp %s | J=%d, n=%d, sigma_re=%.2f | %d chains x %d post-warmup = %d draws\n",
            as.character(packageVersion("smoothbp")), J, N_OBS, SIGMA_RE,
            CHAINS, ITER - WARMUP, CHAINS * (ITER - WARMUP)))
cat(sprintf("target_accept=%.2f | %.1f s | divergences: total %d (subj %s, omega %s, rho %s)\n\n",
            TARGET_ACC, elapsed, fit$n_divergent,
            if (is.null(blk)) "-" else blk$subj,
            if (is.null(blk)) "-" else blk$om,
            if (is.null(blk)) "-" else blk$rho))
print(tab, row.names = FALSE)

meta <- data.frame(
  field = c("version", "J", "n_obs", "sigma_re", "chains", "iter", "warmup",
            "post_warmup_draws", "target_accept", "seconds", "divergences_total",
            "divergences_subject", "divergences_omega", "divergences_rho"),
  value = c(as.character(packageVersion("smoothbp")), J, N_OBS, SIGMA_RE, CHAINS,
            ITER, WARMUP, CHAINS * (ITER - WARMUP), TARGET_ACC, round(elapsed, 1),
            fit$n_divergent,
            if (is.null(blk)) NA else blk$subj,
            if (is.null(blk)) NA else blk$om,
            if (is.null(blk)) NA else blk$rho))
write.csv(tab, file.path(OUT_DIR, "mixing_table.csv"), row.names = FALSE)
write.csv(meta, file.path(OUT_DIR, "mixing_table_regime.csv"), row.names = FALSE)
cat(sprintf("\n[mixing] table  -> %s\n", file.path(OUT_DIR, "mixing_table.csv")))
cat(sprintf("[mixing] regime -> %s\n", file.path(OUT_DIR, "mixing_table_regime.csv")))
