library(smoothbp)
library(dplyr)
set.seed(42)
n_tickers <- 10
n_months  <- 25
dat_sim   <- expand.grid(
  month  = 1:n_months,
  ticker = paste0("T", formatC(1:n_tickers, width = 2, flag = "0"))
)
dat_sim$price <- rnorm(nrow(dat_sim), 10, 0.5)

my_spike <- prior_spike_slab(pi = 0.25, learn_pi = TRUE, a = 1, b = 5)

priors  = smoothbp_priors(
  sigma_re_om = prior_invgamma(2, 1),
  omega       = prior_normal(mean = 12.5, sd = 10, lb = 1, ub = 25)
)

formula = price ~ month
b0      = ~ (1 | ticker)
deltas  = replicate(8, ~ ticker, simplify = FALSE)
omega   = replicate(8, ~ (1 | ticker), simplify = FALSE)
rho     = replicate(8, ~ 1,            simplify = FALSE)

# Build dm and pv as in smoothbp_ss
dm <- smoothbp:::.build_design_matrices(b0, ~1, deltas, omega, rho, dat_sim)

priors_effective <- priors
priors_effective$b1 <- priors$b1
priors_effective$deltas <- my_spike$slab

# Build X_full exactly like Rust
tau <- dat_sim$month
n <- nrow(dat_sim)

# Initialize om and rho exactly as Rust does
# For omega, pv$om[[k]]$mean is 12.5 for all columns.
# x_om has 11 columns. beta_om = rep(12.5, 11)
# om_vec = x_om %*% beta_om
# Since x_om has intercept + 1 dummy per row, sum of x_om for any row is 2.
# So om_vec = 25.0 for all rows!
om_vec <- rep(25.0, n)

# rho_vec = x_rho %*% beta_rho. x_rho is ~1, beta_rho is 3.
rho_vec <- rep(3.0, n)

p_b0 <- ncol(dm$X_b0)
p_b1 <- ncol(dm$X_b1)
p_total <- p_b0 + p_b1 + sum(sapply(dm$X_deltas, ncol))

x_full <- matrix(0, nrow = n, ncol = p_total)
x_full[, 1:p_b0] <- dm$X_b0
x_full[, (p_b0+1):(p_b0+p_b1)] <- dm$X_b1 * (tau - om_vec)

offset <- p_b0 + p_b1
for (k in 1:length(dm$X_deltas)) {
  pk <- ncol(dm$X_deltas[[k]])
  di <- tau - om_vec
  si <- 1 / (1 + exp(-di * rho_vec))
  d_design <- dm$X_deltas[[k]] * (di * si)
  x_full[, (offset+1):(offset+pk)] <- d_design
  offset <- offset + pk
}

# precision matrix
prec_prior <- c(
  rep(1 / 10^2, p_b0),
  rep(1 / 2^2, p_b1),
  rep(1 / 5^2, p_total - p_b0 - p_b1)
)

precision <- t(x_full) %*% x_full + diag(prec_prior)
print("Eigenvalues of precision matrix:")
print(summary(eigen(precision)$values))

tryCatch({
  chol(precision)
  print("Cholesky succeeded!")
}, error = function(e) print(e))
