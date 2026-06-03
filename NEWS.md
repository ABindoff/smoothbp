# smoothbp 0.2.3

* Fixed severe MCMC convergence issue (low ESS, divergences) in random change-points models by correcting adaptation of subject-level parameters in the NUTS step and introducing a joint Gibbs translation step (`omega_translation_step`) for population-level and subject-level intercepts.
* Added per-group partial-NC via fibr integration (`re_fraction` argument).

# smoothbp 0.2.2

* Fixed build and installation failures on Fedora Linux and macOS check systems by conditionally skipping compiling and running the Cargo standalone wrapper generation (`document` target) at install time on CRAN.

# smoothbp 0.2.1

* Fixed a build failure on Windows where the offline Rust vendor directory was
  extracted to the wrong location, causing Cargo to be unable to resolve
  `extendr-api` as a dependency during `R CMD INSTALL`.

# smoothbp 0.2.0

* Initial CRAN release.
