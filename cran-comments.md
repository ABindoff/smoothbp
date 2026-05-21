## Resubmission

This is a patch resubmission (0.2.1) fixing a Windows installation failure
in 0.2.0 where the offline Rust vendor directory was not resolved correctly
when `CARGO_HOME` is placed outside the build tree (required on Windows to
avoid paths with spaces).

## Test environments
* local Windows install, R 4.6.0
* ubuntu 22.04 (on GitHub Actions), R-release
* macOS (on GitHub Actions), R-release

## R CMD check results
0 errors | 0 warnings | 3 notes

### NOTE: hidden files and directories in src/vendor/

The `src/vendor/` directory contains the offline-vendored Rust crate
dependencies (`.cargo-checksum.json`, `.cargo_vcs_info.json`, `.github/`,
etc.). These dotfiles are required by Cargo's directory-source mechanism
to verify crate integrity during offline builds. They cannot be removed
without breaking the offline compilation. This is a standard pattern for
CRAN packages that bundle Rust code (e.g. arcgisbinding, prqlr, savvy).

### NOTE: non-portable file paths in src/vendor/zerocopy-derive/

Four file paths in the vendored `zerocopy-derive` crate exceed 100 bytes.
These files are upstream test fixtures; they are not compiled or installed.
They are stored inside `src/rust/vendor.tar.xz` in the distributed package,
so portability of path length is not a practical concern.

### NOTE: CRAN incoming feasibility
Maintainer: 'Aidan Bindoff <aidan.bindoff@utas.edu.au>'
This is a resubmission.
