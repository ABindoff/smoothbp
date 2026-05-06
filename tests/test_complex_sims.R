# Complex simulation tests for smoothbp
# 1. Breakpoints per group
# 2. Breakpoints conditional on covariates

library(smoothbp)
library(dplyr)
library(ggplot2)
library(posterior)

# ---------------------------------------------------------------------------
# Test 1: Different numbers of breakpoints per group
# Group A: 0 BP
# Group B: 1 BP (at omega=5)
# Group C: 2 BP (at omega=3 and omega=7)
# ---------------------------------------------------------------------------
test_group_breakpoints <- function() {
  message("\n--- Testing Different Breakpoints per Group ---")
  set.seed(123)
  
  n_per_group <- 100
  tau <- seq(0, 10, length.out = n_per_group)
  
  # Group A: 0 BP
  yA <- 5 + 0.5 * tau + rnorm(n_per_group, sd = 0.2)
  
  # Group B: 1 BP at 5
  omB <- 5; rhoB <- 5; b0B <- 5; b1B <- 0.5; deltaB <- -1.0
  diB <- tau - omB; siB <- 1/(1+exp(-diB*rhoB))
  yB <- b0B + b1B*diB + deltaB*diB*siB + rnorm(n_per_group, sd = 0.2)
  
  # Group C: 2 BP at 3 and 7
  omC1 <- 3; omC2 <- 7; rhoC <- 5; b0C <- 5; b1C <- 0.5
  deltaC1 <- -0.8; deltaC2 <- 1.2
  diC1 <- tau - omC1; siC1 <- 1/(1+exp(-diC1*rhoC))
  diC2 <- tau - omC2; siC2 <- 1/(1+exp(-diC2*rhoC))
  yC <- b0C + b1C*diC1 + deltaC1*diC1*siC1 + deltaC2*diC2*siC2 + rnorm(n_per_group, sd = 0.2)
  
  dat <- data.frame(
    y     = c(yA, yB, yC),
    tau   = rep(tau, 3),
    group = rep(c("A", "B", "C"), each = n_per_group)
  )
  
  message("Fitting smoothbp_ss with 2 candidate breakpoints...")
  # We use Group-level interaction for deltas to allow varying BP counts
  fit <- smoothbp_ss(
    formula = y ~ tau,
    b0      = ~ group,
    b1      = ~ group,
    deltas  = list(~ group, ~ group),
    omega   = list(~ 1, ~ 1),
    rho     = list(~ 1, ~ 1),
    data    = dat,
    priors  = smoothbp_priors(
      omega = list(prior_normal(3, 1, lb = 0), prior_normal(7, 1, lb = 0))
    ),
    chains = 2, iter = 2000, warmup = 1000
  )
  
  # Check PIPs for the interaction terms
  pips <- pip(fit)
  message("Posterior Inclusion Probabilities:")
  print(pips)
  
  # We expect gamma_delta1_groupB and gamma_delta1_groupC to be high, 
  # but gamma_delta1_groupA to be low? 
  # Actually, smoothbp_ss currently applies the spike to the entire coefficient vector 
  # of the delta submodel if specified.
  
  # Let's visualize the fit
  pred <- fitted(fit)
  dat_plot <- dat %>% mutate(y_fit = pred$fitted_mean)
  
  p <- ggplot(dat_plot, aes(x = tau, y = y, color = group)) +
    geom_point(alpha = 0.3) +
    geom_line(aes(y = y_fit), size = 1) +
    facet_wrap(~group) +
    theme_minimal() +
    labs(title = "Test 1: Different BP counts per group")
  print(p)
}

# ---------------------------------------------------------------------------
# Test 2: Breakpoints conditional on a continuous covariate
# delta = 1.0 * X (where X is a covariate)
# When X=0, delta=0 -> No breakpoint
# When X=1, delta=1 -> Strong breakpoint
# ---------------------------------------------------------------------------
test_covariate_breakpoints <- function() {
  message("\n--- Testing Breakpoints Conditional on Covariate ---")
  set.seed(456)
  
  n_subjects <- 50
  n_obs      <- 10
  tau        <- seq(0, 10, length.out = n_obs)
  
  # Covariate x ~ Uniform(0, 1)
  x_vals <- runif(n_subjects, 0, 1)
  
  rows <- list()
  for (i in 1:n_subjects) {
    xi <- x_vals[i]
    om <- 5; rho <- 5; b0 <- 10; b1 <- -0.5
    delta <- 2.0 * xi # magnitude scales with xi
    
    di <- tau - om
    si <- 1/(1+exp(-di*rho))
    yi <- b0 + b1*di + delta*di*si + rnorm(n_obs, sd = 0.1)
    
    rows[[i]] <- data.frame(
      subject = i,
      tau     = tau,
      y       = yi,
      x       = xi
    )
  }
  dat <- do.call(rbind, rows)
  
  message("Fitting smoothbp with delta ~ x...")
  fit <- smoothbp(
    formula = y ~ tau,
    b0      = ~ 1 + (1 | subject),
    b1      = ~ 1,
    deltas  = list(~ x), # delta magnitude is linear in x
    omega   = list(~ 1),
    rho     = list(~ 1),
    data    = dat,
    chains  = 2, iter = 2000, warmup = 1000
  )
  
  message("Summary of fixed effects:")
  print(summary(fit, effects = "fixed"))
  
  # Check if delta1_x is close to 2.0
  
  # Visualize
  # Plot for a few values of x
  new_dat <- expand.grid(
    tau = seq(0, 10, length.out = 50),
    x   = c(0, 0.5, 1.0)
  )
  pred <- fitted(fit, newdata = new_dat)
  new_dat$y_fit <- pred$fitted_mean
  
  p <- ggplot(new_dat, aes(x = tau, y = y_fit, color = factor(x))) +
    geom_line(size = 1) +
    theme_minimal() +
    labs(title = "Test 2: BP magnitude proportional to covariate X",
         subtitle = "At X=0, the breakpoint effectively disappears",
         color = "X value")
  print(p)
}

if (!interactive()) {
  test_group_breakpoints()
  test_covariate_breakpoints()
}
