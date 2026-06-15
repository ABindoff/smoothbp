## Resubmission (0.2.5)

This resubmission addresses two issues flagged in the CRAN incoming pre-checks
for 0.2.4:

1. Version increment: 0.2.4 was already in CRAN's incoming queue from a prior
   resubmission fixing a Rust compilation timeout on r-devel-linux-x86_64-
   fedora-clang. The version has been incremented to 0.2.5 to allow processing.
   The 0.2.5 release also adds the new `derivative()` method (see NEWS.md).

2. Non-standard top-level files: LaTeX build artefacts
   (smoothbp_manuscript.aux/.log/.out/.pdf) were present at the package root
   from local manuscript compilation. These are now excluded via .Rbuildignore.

3. Parallel make: `src/Makevars.in` and `src/Makevars.win.in` now set
   `MAKEFLAGS=""` immediately before each `cargo` invocation. This prevents
   cargo from inheriting the jobserver file-descriptor tokens that `make -jN`
   places in MAKEFLAGS, which caused "jobserver unavailable" warnings (and
   potential hangs) on CRAN build hosts that use parallel make.

The Rust release profile remains `lto = "thin"` (the fix from the 0.2.4
fedora-clang resubmission).

## R CMD check results

0 errors | 0 warnings | 0 notes

## Test environments

* Local Windows 11, R 4.6.0
* win-builder: R-devel — OK
* win-builder: R-release — OK
* macOS builder (mac.r-project.org): R-release — OK (0 errors, 0 warnings,
  0 notes on pre-built vignettes)

## Test suite note

One test (`smoothbp recovers parameters on simulated data`) is wrapped in
`skip_on_cran()`. It is a stochastic coverage test requiring ~60 s of MCMC
sampling, unsuitable for CRAN infrastructure.

## Downstream dependencies

None. No reverse dependencies exist.

## Changes since last CRAN release (0.2.2)

See NEWS.md. Key changes:

- 0.2.3: Fixed per-subject NUTS adaptation bug in random change-point models.
- 0.2.4: Removed the experimental `re_fraction` argument.
- 0.2.5: Added `derivative()` method; excluded manuscript build artefacts.
