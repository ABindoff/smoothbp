## R CMD check results

0 errors | 0 warnings | 0 notes

## Test environments

* Local Windows 11, R 4.5.0
* win-builder: R-devel (2026-06-12 r90141 ucrt) — OK
* win-builder: R-release — OK

## Test suite note

One test (`smoothbp recovers parameters on simulated data`) is wrapped in
`skip_on_cran()`. It is a stochastic coverage test requiring ~60 s of MCMC
sampling, unsuitable for CRAN infrastructure.

## Downstream dependencies

None. smoothbp 0.2.4 has not previously been on CRAN; 0.2.2 is the current
CRAN version. No reverse dependencies exist.

## Changes since last CRAN release (0.2.2)

See NEWS.md. Key changes:

- 0.2.3: Fixed per-subject NUTS adaptation bug in random change-point models.
  Adaptation state was shared across subjects, causing divergences and poor
  mixing when subject-level change-point distributions differed.
- 0.2.4: Removed the experimental `re_fraction` argument introduced in 0.2.3
  development; it was superseded by the full non-centred reparameterisation
  (`reparameterise = "omega"`) and never released to CRAN.

## SystemRequirements note

The package requires Rust (via the extendr framework). Rust source and an
offline vendor directory are included in the tarball; no network access is
required at install time. Compilation takes approximately 10–15 minutes.
