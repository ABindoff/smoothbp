use nalgebra::{DMatrix, DVector};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand::Rng;
use rand_distr::{Normal, Gamma, Distribution};

use crate::model::{ModelData, Priors, State, log_truncated_normal_prior, sigmoid};



// ---------------------------------------------------------------------------
// LinearCache: precomputed parts of the mean function that do NOT depend on
// beta_om or beta_rho.  Refreshed once per outer iteration after the linear
// coefficient block is sampled.  Used by the MH steps for omega and rho so
// each proposal does O(n) work instead of O(n * (p_b0 + p_b1 + p_b2)).
//
// mu_i = b0_fixed_i + d_i * b1_vals_i + d_i * s_i * b2_vals_i + re_contrib_i
//   where d_i = tau_i - omega_i and s_i = sigmoid(d_i * rho_i).
// ---------------------------------------------------------------------------

struct LinearCache {
    b0_fixed:   DVector<f64>,   // X_b0 * beta_b0
    b1_vals:    DVector<f64>,   // X_b1 * beta_b1   (constant inputs to per-i b1 contribution)
    b2_vals:    DVector<f64>,   // X_b2 * beta_b2
    re_contrib: DVector<f64>,   // per-observation random-intercept value (zero if none)
}

impl LinearCache {
    fn build(state: &State, data: &ModelData) -> Self {
        let b0_fixed = &data.x_b0 * &state.beta_b0;
        let b1_vals  = &data.x_b1 * &state.beta_b1;
        let b2_vals  = &data.x_b2 * &state.beta_b2;
        let mut re_contrib = DVector::<f64>::zeros(data.n);
        if data.n_groups_b0 > 0 {
            for i in 0..data.n {
                let g = data.group_b0[i];
                if g >= 0 {
                    re_contrib[i] = state.u_b0[g as usize];
                }
            }
        }
        LinearCache { b0_fixed, b1_vals, b2_vals, re_contrib }
    }

    /// Log-likelihood given cached linear parts and the current omega/rho vectors.
    /// omega and rho are the n-length per-observation linear predictors
    /// (X_om * beta_om and X_rho * beta_rho respectively).
    fn log_likelihood(
        &self,
        data: &ModelData,
        omega: &DVector<f64>,
        rho:   &DVector<f64>,
        sigma: f64,
    ) -> f64 {
        let n = data.n;
        let sigma2 = sigma * sigma;
        let log_norm = 0.5 * (std::f64::consts::TAU * sigma2).ln();
        let mut ll = 0.0f64;
        for i in 0..n {
            let di = data.tau[i] - omega[i];
            let si = sigmoid(di * rho[i]);
            let mu = self.b0_fixed[i]
                + di * self.b1_vals[i]
                + di * si * self.b2_vals[i]
                + self.re_contrib[i];
            let r = data.y[i] - mu;
            ll -= 0.5 * r * r / sigma2 + log_norm;
        }
        ll
    }
}

// ---------------------------------------------------------------------------
// AdaptProposal: state for the adaptive Metropolis proposal on either beta_om
// or beta_rho.
//
// Three phases inside warmup:
//   Componentwise: random-walk MH one coordinate at a time, each coordinate
//     tuned to ~44% acceptance (1D optimum).  Used for the first portion of
//     warmup so each scale settles before we estimate covariance.  Skipped
//     entirely when p == 1 (the joint and componentwise updates coincide).
//   Adaptive: joint random-walk MH with proposal covariance Sigma_t = scale *
//     (Cov_t + eps*I), where Cov_t is the running sample covariance of post-
//     componentwise iterates (Welford-updated) and `scale` is tuned to ~23.4%
//     joint acceptance.  Initialised at scale = 2.4^2 / d (Roberts & Rosenthal).
//   Frozen: post-warmup, no further updates to step sizes, Cov, or scale.
//     The proposal Cholesky from the end of warmup is used unchanged for the
//     entire sampling phase, so the chain is a valid time-homogeneous Markov
//     chain.
//
// For p == 1 the implementation degenerates to the original scalar adaptive
// random-walk MH (target rate 0.234).
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
enum ProposalPhase {
    Componentwise,
    Adaptive,
    Frozen,
}

struct AdaptProposal {
    p: usize,
    phase: ProposalPhase,

    // Componentwise (Phase A) and 1D scalar fall-back: per-coordinate step,
    // accept and attempt counters reset every tune_window.
    comp_steps:    Vec<f64>,
    comp_accepts:  Vec<u32>,
    comp_attempts: Vec<u32>,

    // Adaptive joint (Phase B) and frozen (Phase C):
    joint_scale:    f64,
    joint_accepts:  u32,
    joint_attempts: u32,

    // Welford running mean and "M2" matrix (sum of (x - mean)(x - new_mean)^T)
    // accumulated over post-componentwise iterates.  Cov = M2 / (n - 1).
    n_seen: usize,
    mean:   DVector<f64>,
    m2:     DMatrix<f64>,

    // Cholesky factor L such that L L^T = scale * (Cov + eps*I).
    // Lazily refreshed during Phase B; held fixed in Phase C.
    proposal_chol: Option<DMatrix<f64>>,
}

impl AdaptProposal {
    fn new(p: usize, init_step: f64) -> Self {
        // Initial joint scale follows Roberts & Rosenthal (2001): 2.4^2 / d.
        let init_joint_scale = if p > 0 {
            init_step * init_step * 2.4 * 2.4 / (p as f64)
        } else {
            init_step * init_step
        };
        AdaptProposal {
            p,
            phase: ProposalPhase::Componentwise,
            comp_steps:    vec![init_step; p],
            comp_accepts:  vec![0; p],
            comp_attempts: vec![0; p],
            joint_scale:    init_joint_scale,
            joint_accepts:  0,
            joint_attempts: 0,
            n_seen: 0,
            mean:   DVector::<f64>::zeros(p),
            m2:     DMatrix::<f64>::zeros(p, p),
            proposal_chol: None,
        }
    }

    /// Welford-style update of running mean and M2 with one new sample x.
    fn observe(&mut self, x: &DVector<f64>) {
        if self.p == 0 { return; }
        self.n_seen += 1;
        let n = self.n_seen as f64;
        let delta1 = x - &self.mean;
        self.mean += &delta1 / n;
        let delta2 = x - &self.mean;
        // m2 += delta1 * delta2^T  (outer product)
        for i in 0..self.p {
            for j in 0..self.p {
                self.m2[(i, j)] += delta1[i] * delta2[j];
            }
        }
    }

    /// Recompute the proposal Cholesky from the running covariance, scaled by
    /// `joint_scale` and ridge-regularised by `eps` on the diagonal.  Returns
    /// false if Cholesky failed (proposal_chol left unchanged).
    fn refresh_chol(&mut self) -> bool {
        if self.p == 0 || self.n_seen < 2 { return false; }
        let n = self.n_seen as f64;
        let mut sigma = (&self.m2) / (n - 1.0);
        sigma *= self.joint_scale;
        // Tikhonov ridge for numerical stability and to keep the proposal
        // non-degenerate while the chain is still exploring.
        let eps = 1e-6_f64.max(self.joint_scale * 1e-8);
        for i in 0..self.p {
            sigma[(i, i)] += eps;
        }
        if let Some(chol) = sigma.cholesky() {
            self.proposal_chol = Some(chol.l());
            true
        } else {
            // Fall back to a heavier ridge once before giving up.
            let mut sigma2 = (&self.m2) / (n - 1.0);
            sigma2 *= self.joint_scale;
            for i in 0..self.p {
                sigma2[(i, i)] += 1e-3;
            }
            if let Some(chol) = sigma2.cholesky() {
                self.proposal_chol = Some(chol.l());
                true
            } else {
                false
            }
        }
    }

    fn is_componentwise(&self) -> bool { self.phase == ProposalPhase::Componentwise }
}

// Multiplicative scaling used by the per-window adaptation.  Same shape as
// the original `adapt_step` but parameterised by target rate.
fn adapt_scalar(step: f64, accept_rate: f64, target: f64) -> f64 {
    let factor = (accept_rate / target).clamp(0.5, 2.0);
    (step * factor).max(1e-8)
}

// Same shape but for a multiplicative scale factor (joint Haario scale).
fn adapt_log_scale(scale: f64, accept_rate: f64, target: f64) -> f64 {
    let factor = (accept_rate / target).clamp(0.5, 2.0);
    (scale * factor).max(1e-12)
}

// ---------------------------------------------------------------------------
// Entry point: run one chain, return (n_post × n_params) matrix
// ---------------------------------------------------------------------------

// progress_fn(chain_id, n_chains, iter, n_iter, in_warmup) -- called every
// report_every iterations.  Passed as a closure from lib.rs where the extendr
// context is available.
//
// `unused_assignments` is allowed because the cache placeholders below are
// always overwritten on the first iteration of the loop; the placeholder
// initialisation exists only to satisfy the borrow checker.
#[allow(unused_assignments)]
pub fn run_chain(
    data: &ModelData,
    priors: &Priors,
    n_iter: usize,
    n_warmup: usize,
    step_om_init: f64,
    step_rho_init: f64,
    seed: u64,
    verbose: bool,
    chain_id: usize,
    n_chains: usize,
    progress_fn: &dyn Fn(usize, usize, usize, usize, bool),
) -> DMatrix<f64> {
    let mut rng = StdRng::seed_from_u64(seed);

    let mut state = init_state(data, priors, &mut rng);
    let n_post = n_iter - n_warmup;
    let n_params = state.n_params();

    // draws stored row-major: rows = posterior samples, cols = parameters
    let mut draws = DMatrix::<f64>::zeros(n_post, n_params);

    // Adaptive proposal state for omega and rho.
    let mut adapt_om  = AdaptProposal::new(priors.p_om,  step_om_init);
    let mut adapt_rho = AdaptProposal::new(priors.p_rho, step_rho_init);

    // Phase boundaries within warmup.
    //
    // For multivariate proposals we spend the first 30% of warmup doing
    // componentwise tuning and accumulating samples for covariance estimation,
    // then switch to adaptive joint MH for the remainder of warmup, then
    // freeze at end of warmup.  For 1D proposals (p == 1) we skip the
    // componentwise -> joint distinction (they are identical) and stay in
    // Componentwise the whole warmup, freezing at end of warmup.
    //
    // If the warmup is too short to estimate a covariance reliably we also
    // stay in Componentwise.
    let use_adaptive_joint_om  = priors.p_om  >= 2 && n_warmup >= 200;
    let use_adaptive_joint_rho = priors.p_rho >= 2 && n_warmup >= 200;
    let phase_a_end_om  = if use_adaptive_joint_om  { (n_warmup as f64 * 0.30) as usize } else { n_warmup };
    let phase_a_end_rho = if use_adaptive_joint_rho { (n_warmup as f64 * 0.30) as usize } else { n_warmup };

    let tune_window = 100usize;

    // Cached log-likelihood + log-priors that survive across MH steps.
    let mut ll     = 0.0f64;
    let mut lp_om  = 0.0f64;
    let mut lp_rho = 0.0f64;

    // Cached current omega and rho linear predictors plus the LinearCache.
    // These are unconditionally rebuilt at the top of every iteration after
    // the linear coefficient block, so the initial values would be discarded
    // on iter 0 -- start them as zero-length placeholders to keep the borrow
    // checker happy without doing redundant initial work.
    let mut omega_cur: DVector<f64> = DVector::<f64>::zeros(0);
    let mut rho_cur:   DVector<f64> = DVector::<f64>::zeros(0);
    let mut cache: LinearCache = LinearCache {
        b0_fixed:   DVector::<f64>::zeros(0),
        b1_vals:    DVector::<f64>::zeros(0),
        b2_vals:    DVector::<f64>::zeros(0),
        re_contrib: DVector::<f64>::zeros(0),
    };

    let report_every = (n_iter / 10).max(1);

    for iter in 0..n_iter {
        // ---------------------------------------------------------------
        // Progress reporting
        // ---------------------------------------------------------------
        if verbose && iter % report_every == 0 {
            progress_fn(chain_id, n_chains, iter, n_iter, iter < n_warmup);
        }
        // ---------------------------------------------------------------
        // Block 1a: Sample random intercepts via non-centred reparameterisation.
        // Sample z_j ~ N(0,1) using the conjugate full conditional, then set
        // u_j = sigma_u * z_j.  This breaks the strong posterior correlation
        // between the overall intercept (beta_b0) and the group effects (u_j).
        // ---------------------------------------------------------------
        if data.n_groups_b0 > 0 {
            sample_random_effects_nc(data, &mut state, &mut rng);
        }

        // ---------------------------------------------------------------
        // Block 1b: Sample linear coefficients (Gibbs, exact conjugate)
        // ---------------------------------------------------------------
        sample_linear_coefs(data, priors, &mut state, &mut rng);

        // After both updates the linear cache, omega/rho vectors, ll and
        // log-priors all need refreshing.
        cache     = LinearCache::build(&state, data);
        // omega/rho only depend on the betas we have not yet touched in this
        // iteration, so they are still current; rebuild defensively in case
        // beta_om/beta_rho have been overwritten elsewhere (cheap).
        omega_cur = &data.x_om  * &state.beta_om;
        rho_cur   = &data.x_rho * &state.beta_rho;
        ll = cache.log_likelihood(data, &omega_cur, &rho_cur, state.sigma);
        lp_om  = state.log_prior_om(priors);
        lp_rho = state.log_prior_rho(priors);

        // Determine the phase for this iteration.
        let phase_om = phase_for(iter, n_warmup, phase_a_end_om,  use_adaptive_joint_om);
        let phase_rho = phase_for(iter, n_warmup, phase_a_end_rho, use_adaptive_joint_rho);
        adapt_om.phase  = phase_om;
        adapt_rho.phase = phase_rho;

        // ---------------------------------------------------------------
        // Block 2a: MH for beta_om
        // ---------------------------------------------------------------
        mh_step_om(
            data, priors, &mut state,
            &mut ll, &mut lp_om,
            &cache, &mut omega_cur, &rho_cur,
            &mut adapt_om,
            &mut rng,
        );

        // ---------------------------------------------------------------
        // Block 2b: MH for beta_rho
        // ---------------------------------------------------------------
        mh_step_rho(
            data, priors, &mut state,
            &mut ll, &mut lp_rho,
            &cache, &omega_cur, &mut rho_cur,
            &mut adapt_rho,
            &mut rng,
        );

        // ---------------------------------------------------------------
        // Block 3a: Sample sigma (Gibbs, InvGamma conjugate)
        // ---------------------------------------------------------------
        sample_sigma(data, priors, &mut state, &mut rng);
        // sigma changed but cache, omega_cur, rho_cur are still valid.
        ll = cache.log_likelihood(data, &omega_cur, &rho_cur, state.sigma);

        // ---------------------------------------------------------------
        // Block 3b: Sample sigma_u (Gibbs, InvGamma conjugate).
        // ---------------------------------------------------------------
        if data.n_groups_b0 > 0 {
            sample_sigma_u(priors, &mut state, &mut rng);
        }

        // ---------------------------------------------------------------
        // Adaptation of step sizes / proposal covariance during warmup.
        // ---------------------------------------------------------------
        if iter < n_warmup {
            // Per-window adaptation of componentwise scales.
            if (iter + 1) % tune_window == 0 {
                tune_componentwise(&mut adapt_om);
                tune_componentwise(&mut adapt_rho);
                tune_joint(&mut adapt_om);
                tune_joint(&mut adapt_rho);
            }

            // Welford accumulation: feed the current draw into the running
            // mean/cov estimator once we have entered the joint-adaptation
            // window for that parameter (we want covariance from samples
            // collected after the componentwise scales settled).
            if use_adaptive_joint_om && iter >= phase_a_end_om {
                adapt_om.observe(&state.beta_om);
                // Refresh proposal occasionally during Phase B.
                if (iter + 1) % tune_window == 0 {
                    adapt_om.refresh_chol();
                }
            }
            if use_adaptive_joint_rho && iter >= phase_a_end_rho {
                adapt_rho.observe(&state.beta_rho);
                if (iter + 1) % tune_window == 0 {
                    adapt_rho.refresh_chol();
                }
            }
        }

        // ---------------------------------------------------------------
        // Store post-warmup draws
        // ---------------------------------------------------------------
        if iter >= n_warmup {
            let row = iter - n_warmup;
            let draw = state.to_vec();
            for (col, &val) in draw.iter().enumerate() {
                draws[(row, col)] = val;
            }
        }
    }

    if verbose {
        progress_fn(chain_id, n_chains, n_iter, n_iter, false);
    }

    // Suppress unused-variable warnings for the per-iteration `ll` cache; it
    // is intentionally maintained but not read after the final iteration.
    let _ = (ll, lp_om, lp_rho);

    draws
}

fn phase_for(iter: usize, n_warmup: usize, phase_a_end: usize, use_joint: bool) -> ProposalPhase {
    if iter >= n_warmup {
        ProposalPhase::Frozen
    } else if !use_joint || iter < phase_a_end {
        ProposalPhase::Componentwise
    } else {
        ProposalPhase::Adaptive
    }
}

fn tune_componentwise(adapt: &mut AdaptProposal) {
    // Only tune when in or just leaving componentwise phase; harmless if
    // counters are zero.
    let target = if adapt.p > 1 { 0.44 } else { 0.234 };
    for k in 0..adapt.p {
        if adapt.comp_attempts[k] > 0 {
            let rate = adapt.comp_accepts[k] as f64 / adapt.comp_attempts[k] as f64;
            adapt.comp_steps[k] = adapt_scalar(adapt.comp_steps[k], rate, target);
        }
        adapt.comp_accepts[k]  = 0;
        adapt.comp_attempts[k] = 0;
    }
}

fn tune_joint(adapt: &mut AdaptProposal) {
    if adapt.joint_attempts > 0 {
        let rate = adapt.joint_accepts as f64 / adapt.joint_attempts as f64;
        adapt.joint_scale = adapt_log_scale(adapt.joint_scale, rate, 0.234);
    }
    adapt.joint_accepts  = 0;
    adapt.joint_attempts = 0;
}

// ---------------------------------------------------------------------------
// Initialise state from prior means with small jitter
// ---------------------------------------------------------------------------

fn init_state(data: &ModelData, priors: &Priors, rng: &mut StdRng) -> State {
    let jitter = Normal::new(0.0, 0.1).unwrap();

    let mut init_vec = |range: std::ops::Range<usize>| -> DVector<f64> {
        DVector::from_iterator(
            range.len(),
            range.map(|i| {
                // Clamp to [lb, ub] after jitter
                let v = priors.mean[i] + jitter.sample(rng);
                v.max(priors.lb[i] + 1e-6).min(priors.ub[i] - 1e-6)
            }),
        )
    };

    let beta_b0 = init_vec(priors.b0_range());
    let beta_b1 = init_vec(priors.b1_range());
    let beta_b2 = init_vec(priors.b2_range());
    let beta_om = init_vec(priors.om_range());
    let beta_rho = init_vec(priors.rho_range());

    let u_b0 = DVector::zeros(data.n_groups_b0);

    State {
        beta_b0,
        u_b0,
        beta_b1,
        beta_b2,
        beta_om,
        beta_rho,
        sigma: 1.0,
        sigma_u: 1.0,
    }
}

// ---------------------------------------------------------------------------
// Block 1a: Non-centred random intercepts  (Gibbs, exact conjugate)
//
// Reparameterise: u_j = sigma_u * z_j,  z_j ~ N(0, 1)  (prior).
//
// Given z_j, the model is y_i = fixed_i + sigma_u * z_{group_i} + eps_i.
// The full conditional for z_j:
//   precision: tau_j = n_j * sigma_u^2 / sigma^2 + 1
//   mean:      mu_j  = (sigma_u / sigma^2) * sum_k r_{j,k} / tau_j
// ---------------------------------------------------------------------------

fn sample_random_effects_nc(
    dat