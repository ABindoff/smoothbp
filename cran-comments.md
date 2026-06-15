## Resubmission (0.2.6)

This resubmission fixes installation failures on M1/ARM macOS and Alpine
musl Linux that were present in 0.2.5 (and reported against 0.2.4 in the
CRAN additional-issues checks).

### Root cause

`src/Makevars.in` and `src/Makevars.win.in` unconditionally ran
`cargo run --bin document` to regenerate the extendr R-wrapper code.
This binary links against `libR`, which is not on the default linker or
dynamic-library search path on M1mac (causing "Library not loaded:
libR.dylib" at run time) and Alpine musl (causing "cannot find -lR" at
link time).

### Fix

The `cargo run --bin document` step is now guarded by `NOT_CRAN`. When
`NOT_CRAN` is unset (all CRAN build environments), the step is skipped;
the pre-built `R/extendr-wrappers.R` and `src/entrypoint.c` committed in
the source package are used instead. The step still runs in development
builds where `NOT_CRAN` is set, so wrapper regeneration after Rust API
changes is unaffected.

### Other fixes carried forward from 0.2.5

1. Version increment from 0.2.4 (CRAN incoming-queue duplicate).
2. `.Rbuildignore`: LaTeX build artefacts and old source tarballs excluded.
3. `MAKEFLAGS=""` set before each `cargo` invocation (parallel-make fix
   per private note from Prof Ripley).
4. New `derivative()` method (see NEWS.md).

## R CMD check results

0 errors | 0 warnings | 0 notes

## Test environments

* Local Windows 11, R 4.6.0
* win-builder: R-devel — 1 NOTE ("Days since last update: 1", expected)
* win-builder: R-release — 1 NOTE (same)

## Test suite note

One test (`smoothbp recovers parameters on simulated data`) is wrapped in
`skip_on_cran()`. It requires ~60 s of MCMC sampling.

## Downstream dependencies

None.

## Changes since last CRAN release (0.2.2)

See NEWS.md.
