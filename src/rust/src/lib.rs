use extendr_api::prelude::*;
use nalgebra::{DMatrix, DVector};
use rayon::prelude::*;

mod model;
mod sampler;

use model::{ModelData, Priors};
use sampler::run_chain;

// Helper: build DMatrix from a flat column-major slice + dimensions
fn flat_to_dmatrix(data: &[f64], nrow: usize, ncol: usize) -> DMatrix<f64> {
    DMatrix::from_column_slice(nrow, ncol, data)
}

/// Run Metropolis-within-Gibbs sampler for the smooth change-point model.
///
/// @param y           Response vector (n).
/// @param tau         Time variable (n).
/// @param x_b0        Design matrix for b0 as flat column-major vector.
/// @param p_b0        Number of columns in x_b0.
/// @param x_b1        Design matrix for b1.
/// @param p_b1        Number of columns in x_b1.
/// @param x_b2        Design matrix for b2.
/// @param p_b2        Number of columns in x_b2.
/// @param x_om        Design matrix for omega.
/// @param p_om        Number of columns in x_om.
/// @param x_rho       Design matrix for rho.
/// @param p_rho       Number of columns in x_rho.
/// @param group_b0    0-based group indices for b0 random intercept (-1 = no RE).
/// @param n_groups_b0 Number of RE groups (0 = no random effect).
/// @param prior_mean  Prior means concatenated in the order
///                    `b0`, `b1`, `b2`, `omega`, `rho`.
/// @param prior_sd    Prior SDs (same order).
/// @param prior_lb    Lower bounds (-Inf for unconstrained).
/// @param prior_ub    Upper bounds (+Inf for unconstrained).
/// @param sigma_shape Shape of InvGamma prior on residual variance.
/// @param sigma_scale Scale of InvGamma prior on residual variance.
/// @param sigma_u_shape Shape of InvGamma prior on RE variance.
/// @param sigma_u_scale Scale of InvGamma prior on RE variance.
/// @param step_om     Initial MH step size for omega coefficients.
/// @param step_rho    Initial MH step size for rho coefficients.
/// @param chains      Number of independent chains.
/// @param iter        Total iterations per chain (warmup + sampling).
/// @param warmup      Number of warmup iterations discarded.
/// @param seed        Integer random seed.
/// @param verbose     Print progress to the R console.
/// @param n_cores     Number of threads for parallel chain execution.
///                    1 = sequential (with progress bar); > 1 = parallel
///                    (progress bar suppressed; chains run concurrently).
/// @return List with one matrix per chain (rows = post-warmup draws, cols = parameters).
/// @export
#[extendr]
fn run_mcmc(
    y: &[f64],
    tau: &[f64],
    x_b0: &[f64], p_b0: i32,
    x_b1: &[f64], p_b1: i32,
    x_b2: &[f64], p_b2: i32,
    x_om: &[f64],  p_om: i32,
    x_rho: &[f64], p_rho: i32,
    group_b0: &[i32],
    n_groups_b0: i32,
    prior_mean: &[f64],
    prior_sd: &[f64],
    prior_lb: &[f64],
    prior_ub: &[f64],
    sigma_shape: f64,
    sigma_scale: f64,
    sigma_u_shape: f64,
    sigma_u_scale: f64,
    step_om: f64,
    step_rho: f64,
    chains: i32,
    iter: i32,
    warmup: i32,
    seed: i32,
    verbose: bool,
    n_cores: i32,
) -> List {
    let n = y.len();
    let p_b0 = p_b0 as usize;
    let p_b1 = p_b1 as usize;
    let p_b2 = p_b2 as usize;
    let p_om = p_om as usize;
    let p_rho = p_rho as usize;

    let data = ModelData {
        y: DVector::from_column_slice(y),
        tau: DVector::from_column_slice(tau),
        x_b0:  flat_to_dmatrix(x_b0,  n, p_b0),
        x_b1:  flat_to_dmatrix(x_b1,  n, p_b1),
        x_b2:  flat_to_dmatrix(x_b2,  n, p_b2),
        x_om:  flat_to_dmatrix(x_om,  n, p_om),
        x_rho: flat_to_dmatrix(x_rho, n, p_rho),
        group_b0: group_b0.to_vec(),
        n_groups_b0: n_groups_b0 as usize,
        n,
    };

    let priors = Priors {
        mean: prior_mean.to_vec(),
        sd: prior_sd.to_vec(),
        lb: prior_lb.to_vec(),
        ub: prior_ub.to_vec(),
        sigma_shape,
        sigma_scale,
        sigma_u_shape,
        sigma_u_scale,
        p_b0,
        p_b1,
        p_b2,
        p_om,
        p_rho,
    };

    let n_chains  = chains as usize;
    let n_iter    = iter as usize;
    let n_warmup  = warmup as usize;
    let base_seed = seed as u64;
    let n_cores   = (n_cores as usize).max(1);
    let parallel  = n_cores > 1 && n_chains > 1;

    // -----------------------------------------------------------------------
    // Step 1: run all chains and collect DMatrix<f64> results.
    //
    // DMatrix<f64> is Send, so Rayon worker threads may produce it.
    // The R API (rprintln!, RMatrix) is NOT thread-safe and must only be
    // called from the main R thread — see Step 2.
    // -----------------------------------------------------------------------

    let matrices: Vec<DMatrix<f64>> = if parallel {
        // Announce parallel run from the main thread before handing off.
        if verbose {
            rprintln!(
                "Running {} chains in parallel on {} thread(s)...",
                n_chains, n_cores
            );
        }

        // Build a scoped thread pool so the number of threads is controlled
        // per-call without touching the global Rayon pool.
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(n_cores)
            .build()
            .expect("Failed to build Rayon thread pool");

        pool.install(|| {
            (0..n_chains)
                .into_par_iter()
                .map(|c| {
                    let chain_seed = base_seed.wrapping_add(c as u64 * 1_000_003);
                    // Pass a no-op progress closure: calling rprintln! from a
                    // worker thread would invoke Rprintf on a non-R thread,
                    // which is undefined behaviour.
                    run_chain(
                        &data, &priors,
                        n_iter, n_warmup,
                        step_om, step_rho,
                        chain_seed,
                        false,           // verbose = false inside worker
                        c, n_chains,
                        &|_, _, _, _, _| {},
                    )
                })
                .collect()
        })

    } else {
        // Sequential path: full two-colour progress bar per chain.
        //
        // '~' = warmup portion, '=' = sampling portion, ' ' = remaining.
        // This closure is only ever called from the main R thread.
        let progress_fn = move |chain_id: usize, n_chains: usize,
                                 iter: usize, n_iter: usize, in_warmup: bool| {
            let safe_n      = n_iter.max(1);
            let pct         = (iter * 100) / safe_n;
            let done        = iter * 20 / safe_n;
            let warmup_end  = n_warmup * 20 / safe_n;
            let warmup_fill = warmup_end.min(done);
            let sample_fill = done.saturating_sub(warmup_fill);
            let empty       = 20usize.saturating_sub(done);
            let bar: String = "~".repeat(warmup_fill)
                + &"=".repeat(sample_fill)
                + &" ".repeat(empty);
            let phase = if iter == n_iter { "done    " }
                        else if in_warmup  { "warmup  " }
                        else               { "sampling" };
            rprintln!(
                "  Chain {}/{} [{}] {:3}%  ({})  iter {}/{}",
                chain_id + 1, n_chains, bar, pct, phase, iter, n_iter
            );
        };

        (0..n_chains)
            .map(|c| {
                if verbose {
                    rprintln!("Chain {}/{}", c + 1, n_chains);
                }
                let chain_seed = base_seed.wrapping_add(c as u64 * 1_000_003);
                run_chain(
                    &data, &priors,
                    n_iter, n_warmup,
                    step_om, step_rho,
                    chain_seed,
                    verbose, c, n_chains,
                    &progress_fn,
                )
            })
            .collect()
    };

    if verbose && parallel {
        rprintln!("All {} chains complete.", n_chains);
    }

    // -----------------------------------------------------------------------
    // Step 2: convert DMatrix results to R matrices on the main thread.
    // -----------------------------------------------------------------------
    let chain_results: Vec<Robj> = matrices
        .into_iter()
        .map(|draws| {
            let n_post   = draws.nrows();
            let n_params = draws.ncols();
            let flat: Vec<f64> = draws.iter().cloned().collect();
            RMatrix::new_matrix(n_post, n_params, |r, c| flat[c * n_post + r]).into()
        })
        .collect();

    let n_params_total = p_b0 + (n_groups_b0 as usize) + p_b1 + p_b2 + p_om + p_rho + 2;
    list!(
        draws    = chain_results,
        iter     = iter,
        warmup   = warmup,
        chains   = chains,
        n_params = n_params_total as i32
    )
}

// Macro to register exports with R
extendr_module! {
    mod smoothbp;
    fn run_mcmc;
}
