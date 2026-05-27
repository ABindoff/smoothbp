## Resubmission

This is a resubmission of version 0.2.1, addressing the feedback received from the CRAN volunteers regarding the initial submission of this version.

### Volunteer Review Feedback Fixes:

1. **DESCRIPTION Formatting & References**:
   - Wrapped the software name `'Rust'` in single quotes in the `Description` field to comply with naming guidelines.
   - Added method references in the description field in the required format: `Bacon and Watts (1971) <doi:10.2307/2334389>` and `Kuo and Mallick (1998) <https://www.jstor.org/stable/25053023>` with no spaces after `doi:`/`https:`.

2. **Missing `\value` Tags in S3/Generics Documentation**:
   - Added `@return` roxygen tags detailing the output structure (class) and semantic meaning for the S3 methods and generic functions: `print.smoothbp_fit`, `summary.smoothbp_fit`, `as.data.frame.smoothbp_fit`, `fitted.smoothbp_fit`, `log_lik`, `bridge_sampler.smoothbp_fit`, and `bayes_factor.smoothbp_fit`.
   - Regenerated all `.Rd` files cleanly using `devtools::document()`.

3. **Compiler and Installation Cleanliness (`tools/config.R` / `Makevars.win.in`)**:
   - Removed the dynamic creation of `src/.cargo/config.toml` from the R script `tools/config.R`.
   - Instead, the Windows linker and AR tool paths detected at build time are passed via Makefile placeholders (`@LINKER@` and `@AR@`) to `src/Makevars.win.in`.
   - Updated `src/Makevars.win.in` to use a safe `CARGOTMP` inside R's `tempdir()` (converted to Windows short paths) using `:=` (simply expanded) to ensure Rscript runs exactly once. Linker configuration is written and cleaned up dynamically within `tempdir()` during the compilation step, leaving no `.cargo` or build files in the package directory.

---

## Test environments
* local Windows 11 install, R 4.6.0
* ubuntu 22.04 (on GitHub Actions), R-release
* macOS (on GitHub Actions), R-release

## R CMD check results
0 errors | 0 warnings | 1 note

### NOTE: CRAN incoming feasibility
Maintainer: 'Aidan D Bindoff <aidan.bindoff@utas.edu.au>'
New submission
