# CRAN submission comments

## Test environments

- Windows 11 x64, R 4.6.0 (local)
- R-hub (Windows, Ubuntu, macOS) — to be added before submission

## R CMD check results

0 errors | 0 warnings | 1 note

### Note: non-API call to R (`R_NamespaceRegistry`)

```
File 'smoothbp/libs/x64/smoothbp.dll':
  Found non-API call to R: 'R_NamespaceRegistry'
```

This call originates from the `extendr-api` Rust crate (v0.8.1), which is
used to interface between R and the compiled Rust sampler.  It is present in
extendr's generated wrapper code, not in the package's own Rust source.  The
extendr project is aware of the remaining non-API usage and is tracking its
removal in upcoming releases.  The call is read-only (namespace registry
inspection) and poses no stability risk with current R versions.

## Package notes

- **SystemRequirements**: Cargo (Rust's package manager) and rustc ≥ 1.65.0
  must be installed.  The Makevars files handle compilation automatically on
  platforms where Cargo is available.  The package links only against
  standard system libraries (`ws2_32`, `advapi32`, `userenv`, `bcrypt`,
  `ntdll` on Windows; none beyond `libR` on Unix).

- The package bundles a pre-configured `.cargo/config.toml` in `src/.cargo/`
  to pin registry settings for offline/reproducible builds.  This directory
  is excluded from the installed package via `.Rbuildignore`.

- **Vignette build time**: the `brms-comparison` vignette fits Stan models
  and takes several minutes.  It is guarded by `eval = have_brms` and will
  build in environments where `brms` is installed.
