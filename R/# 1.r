# 1. Compile the updated Rust backend and refresh documentation
rextendr::document()

# 2. Run the validation tests
source("tests/test_validation.R")

# 3. Try a multi-breakpoint model
fit <- smoothbp(
  formula = value ~ tau,
  deltas = list(~ 1, ~ 1), # Specify 2 breakpoints
  data = my_data
)
summary(fit)
