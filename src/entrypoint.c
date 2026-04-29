// We need to forward routine registration from C to Rust
// to avoid the linker removing the static library.
// Note: register_extendr_panic_hook() was internalised in extendr-api 0.7;
// it is no longer a separate exported symbol.

void R_init_smoothbp_extendr(void *dll);

void R_init_smoothbp(void *dll) {
    R_init_smoothbp_extendr(dll);
}
