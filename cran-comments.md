## Resubmission (0.2.7)

This resubmission fixes a portability issue raised by Prof Ripley after
the 0.2.6 submission.

### Problem

`src/Makevars.in` and `src/Makevars.win.in` ran `cargo run --bin document`
during package installation. This binary links against `libR`, but R is not
built as a shared library by default (`./configure && make` does not produce
`libR.so`/`libR.dylib`). Any user building from source without
`--enable-R-shlib` — including all standard Linux installations and CRAN's
own M1mac and musl platforms — would fail to install the package.

The 0.2.6 fix (guarding the step with `NOT_CRAN`) was insufficient: users
with `NOT_CRAN` set, or those on any platform without a shared `libR`,
would still be affected.

### Fix

The `cargo run --bin document` block has been removed entirely from both
Makevars templates. The generated wrapper files (`R/extendr-wrappers.R`,
`src/entrypoint.c`) are pre-built and committed in the source package;
they do not need to be regenerated at install time. Wrapper regeneration
during development is handled separately by `rextendr::document()`.

### Other fixes carried forward from 0.2.5/0.2.6

- Version increments required by rapid CRAN iteration.
- `.Rbuildignore`: LaTeX artefacts and old tarballs excluded.
- `MAKEFLAGS=""` before each `cargo build` (parallel-make fix).
- New `derivative()` method (see NEWS.md).

## R CMD check results

0 errors | 0 warnings | 0 notes

## Test environments

* Local Windows 11, R 4.6.0
* win-builder: R-devel — 1 NOTE ("Days since last update: N", expected)
* win-builder: R-release — OK

## Test suite note

One test (`smoothbp recovers parameters on simulated data`) is wrapped in
`skip_on_cran()`. It requires ~60 s of MCMC sampling.

## Downstream dependencies

None.

## Changes since last CRAN release (0.2.2)

See NEWS.md.
