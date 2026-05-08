library(smoothbp)
library(dplyr)
library(posterior)

dat_market <- data.frame(
  month  = rep(1:15, 3),
  ticker = rep(c("NVDA", "MSFT", "AAPL"), each = 15),
  price  = c(
    13.50, 16.50, 14.60, 19.52, 23.19, 27.75, 27.72, 37.80, 42.27, 46.69, 49.32, 43.47, 40.75, 46.74, 49.49,
    232, 255, 240, 241.47, 243.65, 281.63, 300.15, 321.49, 333.39, 328.87, 321.56, 309.77, 331.71, 372.49, 369.67,
    153, 148, 130, 142.01, 145.30, 162.55, 167.26, 174.96, 191.46, 193.91, 185.69, 169.23, 168.79, 188.00, 190.55
  )
)
dat_market$ticker <- factor(dat_market$ticker, levels = c("AAPL", "MSFT", "NVDA"))

my_spike <- prior_spike_slab(
  pi = 0.2,
  learn_pi = TRUE,
  a = 1, b = 9
)

# Create a list of 8 priors, spaced out across the time domain
# Time domain is 1 to 15 in dat_market, so space them from 2 to 14
om_priors <- lapply(seq(2, 14, length.out = 8), function(m) {
  prior_normal(mean = m, sd = 10, lb = 1, ub = 15)
})

message("Fitting hierarchical SS model...")
fit_ss_hier <- smoothbp_ss(
  formula = price ~ month, b0 = ~ (1|ticker),
  deltas = replicate(8, ~ ticker, simplify = FALSE), 
  omega  = replicate(8, ~ (1 | ticker), simplify = FALSE),
  rho    = replicate(8, ~ 1, simplify = FALSE),
  data   = dat_market,
  spike  = my_spike,
  priors = smoothbp_priors(
    sigma_re_om = prior_invgamma(2, 1),
    omega = om_priors
  ),
  chains = 2, iter = 500, warmup = 200,
  cores = 1
)

draws_ss <- as_draws_df(fit_ss_hier$draws)
print(summary(draws_ss$pi))
print(summary(draws_ss$sigma_re_omega1))
message("SUCCESS!")
