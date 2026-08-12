# smoothbp 0.2.8

## Sampler correctness

* Fixed the NUTS proposal rule in the random-effects sampler. The doubling
  loop updated its proposal from the newly built subtree without checking
  whether that subtree had terminated, omitting the `s' = 1` guard of
  Hoffman and Gelman (2014, Algorithm 3). A subtree that ends on a U-turn
  contains states the reverse trajectory could not reach, so proposing
  from it breaks reversibility; because U-turn termination is the ordinary
  way the loop ends, this affected a large share of transitions rather
  than a rare tail.

  The effect is not subtle. Sampling a standard normal with the affected
  code returns a variance of 1.24 rather than 1.0, over-dispersed in the
  direction the bug predicts, since the wrongly admitted proposals come
  from beyond the U-turn. In fitted models the over-dispersed
  random effects degrade the fit and the residual scale absorbs it, which
  is how this surfaced: `sigma` was systematically over-estimated in
  simulation-based calibration while `omega`, `delta` and `sigma_re`
  looked calibrated.

  **Any random change-point or random-slope fit produced by 0.2.x should
  be re-run.** Only the random-effects sampler is affected; the
  non-hierarchical entry points use fixed-length HMC with a single
  Metropolis correction and are unaffected, which simulation-based
  calibration confirms.

* Fixed the change-point translation move under `reparameterise = "omega"`.
  `omega_translation_step()` rescaled the random-effect columns by
  `sigma_re`, but those columns always hold centred values. Every other
  sampling step converts back on write-back, and `omega_vec()` feeds them
  straight into the likelihood. The move therefore shifted the
  subject-level change-points instead of leaving them invariant, while
  still being accepted with probability one. Results computed with
  `reparameterise = "omega"` and a random change-point should be
  regarded as unreliable and re-run. The default,
  `reparameterise = "none"`, was never affected.

## Sampler efficiency

* The change-point update is now collapsed (Rao-Blackwellised): the linear
  block (`b0`, `b1`, the slope changes) is integrated out analytically
  before `omega` is drawn, then resampled from its full conditional given
  the new `omega`. Previously `omega` was drawn conditional on a single
  draw of those coefficients and vice versa, so the chain traversed the
  strong `omega`-`delta` dependence one axis at a time. On a
  single-change-point model this raised bulk ESS for `omega` from a median
  of 394 to 5588 out of 5000 posterior draws, cut the worst R-hat from
  1.090 to 1.006, and cost about 14% more time per fit. Posterior
  marginals are unchanged; only the efficiency differs.

  Note that the collapse necessarily runs in this direction. `omega`
  cannot be integrated out of the `delta` update, since it enters through
  `d = t - omega` and the transition weight `s(d; rho)` and that integral
  has no closed form. The linear block can be integrated out of the
  `omega` update because it is conditionally Gaussian, and that breaks the
  same dependence.

  Models that place finite bounds on a linear coefficient
  (`prior_normal(lb=, ub=)` on `b0`, `b1` or `deltas`) keep the previous
  conditional update, because the marginal likelihood of a box-truncated
  Gaussian is not available in closed form.

* Added a joint translation move for the population intercept and the
  group-level random intercepts, `b0 -> b0 + c` and `u_j -> u_j - c`. Only
  the sum `b0 + u_j` is identified within a group, so updating the fixed
  coefficients conditional on the random intercepts and vice versa could
  not traverse the ridge between them: `b0` reached a bulk effective
  sample size of 91 out of 4000 draws on the vignette's random-intercept
  model while every other parameter mixed freely. The move raises that to
  3558 at no measurable cost in run time. The shift leaves the mean
  function exactly invariant, so `c` is drawn from its prior-only
  conditional and accepted outright; it is skipped when invariance would
  not hold (no intercept column, or observations outside every group).

  Note that the package documentation previously described the fixed and
  group-level intercepts as being updated in a single joint block. They
  were not, which is what the low effective sample size reflected.

* The spike-and-slab inclusion indicators now use a collapsed
  (Rao-Blackwellised) update: the slab coefficient is integrated out
  analytically before the indicator is drawn, then resampled given the
  new indicator. Previously each indicator was drawn at the current
  coefficient, so an excluded coefficient sat wherever the diffuse slab
  put it, fitted badly, and rarely re-entered. The indicator and its
  coefficient locked together and indicator ESS fell into the tens. This
  targets the same posterior as before (it is a better sampler of it,
  not a different model), so posterior inclusion probabilities keep their
  meaning.

* The collapsed update also honours bounded slabs (`prior_normal(lb=, ub=)`
  on `deltas` or `b1`), which the previous indicator update ignored, and
  uses the learned `sigma_re` rather than the fixed-effect prior as the
  slab for random-effect columns.

## Validation

* `tools/sbc_ecdf_bands.R` now puts every estimated parameter on the SBC
  figure and writes PASS/FAIL, interval-coverage and run-regime tables to
  `tools/part3_results/`. Previously `b0`, `b1` and `sigma` were checked
  only on the console and left no artefact, which is why a systematic
  `sigma` miscalibration went unnoticed.

# smoothbp 0.2.7

* Fixed installation on any platform where R was not built as a shared
  library (`--enable-R-shlib`). The `cargo run --bin document` step
  — which compiled and ran a binary that links against `libR` — has been
  removed from `src/Makevars.in` and `src/Makevars.win.in`. The generated
  wrapper files (`R/extendr-wrappers.R`, `src/entrypoint.c`) are
  pre-built and committed in the source package; regeneration during
  installation is neither necessary nor portable.

# smoothbp 0.2.6

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
