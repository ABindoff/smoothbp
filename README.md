# smoothbp <img src="man/figures/smoothbp_hexlogo.svg" align="right" height="139" />

Fits smoothed hierarchical piecewise regression with **multiple change-points** using an optimised Metropolis-within-Gibbs sampler implemented in Rust.

## Features

- **Multi-Breakpoint Models**: Support for an arbitrary number of change-points.
- **Spike-and-Slab Regularization**: Automatic selection of the number of breakpoints using `smoothbp_ss()`.
- **Flexible Predictors**: Change-point locations (`omega`), slope changes (`delta`), and transition sharpness (`rho`) can all be conditioned on covariates.
- **Hierarchical Structure**: Random intercepts for all parameters.
- **High Performance**: Core MCMC engine written in Rust.

## Installation

```r
# Requires Rtools45 on Windows for Rust compilation
pak::pkg_install("ABindoff/smoothbp")
```
