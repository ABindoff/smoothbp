library(smoothbp)
library(dplyr)
library(ggplot2)

set.seed(42)

# Simulate 5 tickers responding to a shared market event
# Event at month 10
n_tickers <- 5
months <- 1:20
market_omega <- 10
lags <- c(0, 0.2, -0.2, 0.5, -0.1) # Ticker-specific lags

sim_data <- do.call(rbind, lapply(1:n_tickers, function(i) {
  ticker_name <- paste0("T", i)
  ticker_omega <- market_omega + lags[i]
  
  # Sigmoidal transition
  # y = 10 (pre) -> 20 (post)
  tau <- months
  mu <- 10 + 10 * (1 / (1 + exp(-(tau - ticker_omega) * 2)))
  y <- mu + rnorm(length(tau), sd = 0.2)
  
  data.frame(ticker = ticker_name, month = tau, y = y)
}))

# Fit model WITHOUT hierarchical shrinkage
fit_fixed <- smoothbp(
  formula = y ~ month,
  omega = list(~ ticker),
  data = sim_data,
  iter = 1000, warmup = 500, chains = 2
)

# Fit model WITH hierarchical shrinkage
fit_hier <- smoothbp(
  formula = y ~ month,
  omega = list(~ ticker),
  data = sim_data,
  hierarchical = "omega",
  iter = 1000, warmup = 500, chains = 2
)

# Compare results
# We expect sigma_re_omega1 to be small (around 0.2-0.3)
# and we expect the ticker offsets to be shrunk toward zero in fit_hier compared to fit_fixed
fixed_draws <- posterior::as_draws_df(fit_fixed$draws)
hier_draws <- posterior::as_draws_df(fit_hier$draws)

cat("\nFixed Effects (No Shrinkage):\n")
print(summary(fixed_draws[, grep("omega1", names(fixed_draws))]))

cat("\nHierarchical Effects (Adaptive Shrinkage):\n")
print(summary(hier_draws[, grep("omega1", names(hier_draws))]))

cat("\nLearned Market Synchronization (sigma_re_omega1):\n")
print(summary(hier_draws$sigma_re_omega1))
