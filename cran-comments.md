## Submission Summary
This is the initial release of the `smoothbp` package. `smoothbp` provides a Metropolis-within-Gibbs sampler for Bayesian hierarchical piecewise regression with multiple logistic-smoothed change-points.

## Test environments
* Local Windows 11, R 4.6.0
* (Recommended to run R-hub checks before actual submission)

## R CMD check results
0 errors | 1 warning | 0 notes

* Warning: checking Rust compilation ... WARNING Downloads Rust crates
  * This is expected as the package uses the `extendr` framework. We have ensured that `Cargo.lock` is included to provide reproducible builds. If required by CRAN, we can vendor all dependencies in a subsequent submission.

## Note on New Submission
* This is a new release.
* The package has been rigorously tested against `brms` (Stan) to ensure parameter recovery and posterior consistency.
* All vignettes have been verified to build correctly and provide comprehensive examples for users in medicine, ecology, and finance.
