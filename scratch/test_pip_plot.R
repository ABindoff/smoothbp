library(smoothbp)
library(ggplot2)

# Load the previously fitted model if available, or just re-run a small one.
# For speed, let's just use the logic on a mock object if we can, 
# but better to run a real small model.

set.seed(123)
n <- 100
dat <- data.frame(
  x = seq(1, 10, length.out = n),
  y = rnorm(n, mean = 5 + 2 * (seq(1, 10, length.out = n) > 5) * (seq(1, 10, length.out = n) - 5), sd = 0.5)
)

# Small model
fit <- smoothbp_ss(
  formula = y ~ x,
  deltas = list(~ 1),
  omega = list(~ 1),
  data = dat,
  iter = 500, warmup = 250, chains = 2, cores = 1
)

pips <- pip(fit)
print(pips)

p <- plot(pips)
ggsave("pip_plot.png", p, width = 8, height = 4)
message("Plot saved to pip_plot.png")
