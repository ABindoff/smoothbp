library(smoothbp)
library(dplyr)
library(ggplot2)

test_that("Hierarchical shrinkage works", {
  testthat::skip_on_cran()
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
    tau <- months
    mu <- 10 + 10 * (1 / (1 + exp(-(tau - ticker_omega) * 2)))
    y <- mu + rnorm(length(tau), sd = 0.2)
    
    data.frame(ticker = ticker_name, month = tau, y = y)
  }))

  # Fit model WITH hierarchical shrinkage (using deprecated argument intentionally)
  fit_hier <- suppressWarnings(smoothbp(
    formula = y ~ month,
    omega = list(~ ticker),
    data = sim_data,
    hierarchical = "omega",
    iter = 1000, warmup = 500, chains = 2,
    .verbose = FALSE
  ))
  
  expect_s3_class(fit_hier, "smoothbp_fit")
  expect_true("sigma_re_omega1" %in% posterior::variables(fit_hier$draws))
})
