## Submission (0.2.8)

This release fixes a correctness defect in the sampler used for models with
random change-points or random slopes. It is not a resubmission; 0.2.7 is on
CRAN and passing.

### The defect

The no-U-turn sampler used for the random-effect blocks updated its proposal
from each newly built sub-trajectory without checking whether that
sub-trajectory had terminated, omitting the `s' = 1` guard of Hoffman and
Gelman (2014, Algorithm 3). A sub-trajectory that ends on a U-turn contains
states the reverse trajectory could not have reached, so proposing from it
breaks reversibility. Because U-turn termination is the ordinary way the
doubling loop ends, this affected a large share of transitions rather than a
rare tail.

Sampling a standard normal with the affected code returns a variance of 1.24
rather than 1.0. In fitted models the over-dispersed random effects degrade
the fit and the residual scale absorbs the slack, which is how it surfaced:
simulation-based calibration showed the residual standard deviation
systematically over-estimated while the other parameters looked calibrated.

Only the random-effects sampler was affected. The non-hierarchical entry
points use fixed-length HMC with a single Metropolis correction, and
simulation-based calibration confirms they were and remain calibrated. Users
of earlier 0.2.x releases who fitted random change-point or random-slope
models are told in NEWS.md to re-run those fits.

A second, narrower fix corrects the change-point translation move under
`reparameterise = "omega"`, which rescaled centred random-effect values as
though they were non-centred. The default, `reparameterise = "none"`, was
never affected.

### Other changes

- The change-point update is now collapsed (Rao-Blackwellised): the linear
  block is integrated out analytically before the change-point is drawn.
  Same posterior, roughly an order of magnitude more effective samples for
  the change-point location.
- The spike-and-slab inclusion indicators use the analogous collapsed update,
  and now honour bounded slabs.
- No user-facing API changes; no changes to function signatures or defaults.

## R CMD check results

0 errors | 0 warnings | 0 notes

## Test environments

* Local Windows 11, R 4.6.1
* win-builder: R-devel
* win-builder: R-release

## Test suite note

Tests that require MCMC sampling are wrapped in `skip_on_cran()`. The Rust
unit tests (`cargo test`) cover the sampler mathematics directly: the
collapsed change-point energy is checked against finite differences and
against the n-by-n Gaussian marginal likelihood it is derived from, and the
no-U-turn sampler is checked against a target with known moments.

## Downstream dependencies

None.

## Changes since last CRAN release (0.2.7)

See NEWS.md.
