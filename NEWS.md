# smoothbp 0.2.6

* Fixed installation on platforms where `libR` is not in the default linker
  or dynamic-library search path (M1/ARM macOS, Alpine musl Linux). The
  `cargo run --bin document` wrapper-generation step now runs only when
  `NOT_CRAN` is set (i.e. during development), since the generated files
  (`R/extendr-wrappers.R`, `src/entrypoint.c`) are pre-built and committed
  in the source package.

# smoothbp 0.2.5

* Added `derivative()` generic and methods for `smoothbp_fit` and
  `smoothbp_ss_fit`. Computes the posterior d-th derivative of the
  conditional mean with respect to tau at each row of a user-supplied
  data frame, with full credible-interval propagation. Orders 1--4 are
  supported via central finite differences; provide a subject column to
  condition on subject-level change-point timing or omit it for
  population-level derivatives.

# smoothbp 0.2.4

* Removed the experimental `re_fraction` argument (added in 0.2.3 dev,
  never released to CRAN): it performed a binary centred/non-centred
  per-group switch, not the partial non-centring its name implied.
  True per-group partial non-centring is deferred to a future release.
  Use `reparameterise = "omega"` for full non-centring; diagnose with
  fibr's `smoothbp_advisor()`.

# smoothbp 0.2.3

* Fixed severe MCMC convergence issue (low ESS, divergences) in random change-points models by correcting adaptation of subject-level parameters in the NUTS step and introducing a joint Gibbs translation step (`omega_translation_step`) for population-level and subject-level intercepts.

# smoothbp 0.2.2

* Fixed build and installation failures on Fedora Linux and macOS check systems by conditionally skipping compiling and running the Cargo standalone wrapper generation (`document` target) at install time on CRAN.

# smoothbp 0.2.1

* Fixed a build failure on Windows where the offline Rust vendor directory was
  extracted to the wrong location, causing Cargo to be unable to resolve
  `extendr-api` as a dependency during `R CMD INSTALL`.

# smoothbp 0.2.0

* Initial CRAN release.
