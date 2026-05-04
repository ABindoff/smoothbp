use nalgebra::{DMatrix, DVector};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand::Rng;
use rand_distr::{Normal, Gamma, Distribution};

use crate::model::{ModelData, Priors, State, log_truncated_normal_prior, sigmoid};


// ---------------------------------------------------------------------------
// LinearCache: precomputed parts of the mean function that do NOT depend on
// beta_om or beta_rho.  Refreshed once per outer iteration after the linear
// coefficient block is sampled.  Used by the HMC steps for omega and rho so
// each gradient/energy evaluation does O(n) work instead of
// O(n * (p_b0 + p_b1 + p_b2)).
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

    /// Gradient of the log-likelihood with respect to beta_om.
    ///
    /// ∂mu_i/∂omega_i = -(b1_i + s_i*b2_i + d_i*rho_i*s_i*(1-s_i)*b2_i)
    /// ∂ll/∂beta_om_k = Σ_i  r_i/σ² * (∂mu_i/∂omega_i) * x_om[i,k]
    ///
    /// Note: ∂omega_i/∂beta_om_k = x_om[i,k], and the chain rule gives
    /// ∂ll/∂beta_om_k = Σ_i (∂ll/∂mu_i)(∂mu_i/∂omega_i)(∂omega_i/∂beta_om_k).
    fn grad_beta_om(
        &self,
        data: &ModelData,
        omega: &DVector<f64>,
        rho:   &DVector<f64>,
        sigma: f64,
        p_om:  usize,
    ) -> DVector<f64> {
        let n = data.n;
        let inv_sigma2 = 1.0 / (sigma * sigma);
        let mut grad = DVector::<f64>::zeros(p_om);
        for i in 0..n {
            let di = data.tau[i] - omega[i];
            let ri = rho[i];
            let si = sigmoid(di * ri);
            let b1i = self.b1_vals[i];
            let b2i = self.b2_vals[i];
            let mu = self.b0_fixed[i]
                + di * b1i
                + di * si * b2i
                + self.re_contrib[i];
            let resid = data.y[i] - mu;

            // ∂mu_i/∂omega_i (note the sign: omega_i enters as tau_i - omega_i)
            let dmu_domega = -(b1i + si * b2i + di * ri * si * (1.0 - si) * b2i);

            // scalar factor for this observation
            let factor = resid * inv_sigma2 * dmu_domega;
            for k in 0..p_om {
                grad[k] += factor * data.x_om[(i, k)];
            }
        }
        grad
    }

    /// Gradient of the log-likelihood with respect to beta_rho.
    ///
    /// ∂mu_i/∂rho_i = d_i² * s_i * (1 - s_i) * b2_i
    /// ∂ll/∂beta_rho_k = Σ_i  r_i/σ² * (∂mu_i/∂rho_i) * x_rho[i,k]
    fn grad_beta_rho(
        &self,
        data: &ModelData,
        omega: &DVector<f64>,
        rho:   &DVector<f64>,
        sigma: f64,
        p_rho: usize,
    ) -> DVector<f64> {
        let n = data.n;
        let inv_sigma2 = 1.0 / (sigma * sigma);
        let mut grad = DVector::<f64>::zeros(p_rho);
        for i in 0..n {
            let di = data.tau[i] - omega[i];
            let ri = rho[i];
            let si = sigmoid(di * ri);
            let b2i = self.b2_vals[i];
            let b1i = self.b1_vals[i];
            let mu = self.b0_fixed[i]
                + di * b1i
                + di * si * b2i
                + self.re_contrib[i];
            let resid = data.y[i] - mu;

            // ∂mu_i/∂rho_i = d_i² * s_i * (1 - s_i) * b2_i
            let dmu_drho = di * di * si * (1.0 - si) * b2i;

            let factor = resid * inv_sigma2 * dmu_drho;
            for k in 0..p_rho {
                grad[k] += factor * data.x_rho[(i, k)];
            }
        }
        grad
    }
}

// ---------------------------------------------------------------------------
// Gradient of the truncated-normal log-prior w.r.t. the parameter vector.
// For each coordinate: ∂logp/∂θ_k = -(θ_k - μ_k) / σ_k².
// Returns a zero vector if the point is outside bounds (though callers
// should not evaluate gradients at out-of-bounds points).
// ---------------------------------------------------------------------------

fn grad_truncated_normal_prior(
    values: &[f64],
    means:  &[f64],
    sds:    &[f64],
    lbs:    &[f64],
    ubs:    &[f64],
) -> DVector<f64> {
    let p = values.len();
    let mut g = DVector::<f64>::zeros(p);
    for i in 0..p {
        if values[i] < lbs[i] || values[i] > ubs[i] {
            return DVector::<f64>::zeros(p);
        }
        g[i] = -(values[i] - means[i]) / (sds[i] * sds[i]);
    }
    g
}

// ---------------------------------------------------------------------------
// HmcAdapt: dual-averaging step-size adaptation and optional diagonal mass
// matrix estimation for the HMC-within-Gibbs steps on beta_om / beta_rho.
//
// During warmup the step size epsilon is tuned via Nesterov dual averaging
// (Algorithm 5 from Hoffman & Gelman 2014, i.e. the NUTS paper) targeting
// a user-specified acceptance probability.  Epsilon is initialised via the
// "find reasonable epsilon" heuristic (Algorithm 4) and floored at
// EPSILON_FLOOR to prevent collapse.
//
// A diagonal mass matrix (inverse variances) is estimated from warmup draws
// via Welford online accumulation and refreshed at a mid-warmup window
// boundary.
//
// Post-warmup, epsilon and the mass matrix are frozen so the chain is a
// valid time-homogeneous Markov chain.  Divergent transitions (|delta_H| >
// DIVERGENCE_THRESHOLD) are counted post-warmup and reported to R.
// ---------------------------------------------------------------------------

/// Minimum epsilon to prevent dual-averaging from collapsing to zero.
const EPSILON_FLOOR: f64 = 1e-6;

/// |delta_H| threshold above which a transition is flagged as divergent.
const DIVERGENCE_THRESHOLD: f64 = 1000.0;

/// Maximum boundary reflections per coordinate per leapfrog step.
const MAX_REFLECTIONS: usize = 20;

struct HmcAdapt {
    p: usize,

    // Leapfrog trajectory length (number of steps).  Randomised per
    // proposal as Uniform(l_min, l_max) to avoid periodic-orbit resonance.
    l_min: usize,
    l_max: usize,

    // Current step size.
    epsilon: f64,

    // Dual-averaging state (Hoffman & Gelman 2014, section 3.2).
    target_accept: f64,
    mu:         f64,       // log(10 * epsilon_0) -- shrinkage target
    log_eps_bar: f64,
    h_bar:      f64,
    gamma:      f64,       // 0.05 (default)
    t0:         f64,       // 10.0  (default)
    kappa:      f64,       // 0.75  (default)
    da_count:   usize,     // number of dual-averaging updates so far

    // Diagonal mass matrix: inv_mass[k] = 1/M_kk (used to scale momentum).
    // Initialised to 1.0 (identity).  Updated from Welford variance estimates
    // during warmup.
    inv_mass: Vec<f64>,

    // Welford accumulators for diagonal variance of the position.
    welford_n:    usize,
    welford_mean: DVector<f64>,
    welford_m2:   DVector<f64>,

    // Whether adaptation is still active.
    adapting: bool,

    // Divergence counter: post-warmup transitions with |delta_H| > threshold.
    n_divergent: usize,
}

impl HmcAdapt {
    fn new(p: usize, init_epsilon: f64, target_accept: f64, l_min: usize, l_max: usize) -> Self {
        HmcAdapt {
            p,
            l_min,
            l_max,
            epsilon: init_epsilon,
            target_accept,
            mu: (10.0 * init_epsilon).ln(),
            log_eps_bar: 0.0,
            h_bar: 0.0,
            gamma: 0.05,
            t0: 10.0,
            kappa: 0.75,
            da_count: 0,
            inv_mass: vec![1.0; p],
            welford_n: 0,
            welford_mean: DVector::<f64>::zeros(p),
            welford_m2:   DVector::<f64>::zeros(p),
            adapting: true,
            n_divergent: 0,
        }
    }

    /// Initialise epsilon via the "find reasonable epsilon" heuristic
    /// (Hoffman & Gelman 2014, Algorithm 4).  Takes a single leapfrog step
    /// and doubles or halves epsilon until the log acceptance probability
    /// crosses ln(0.5).  `one_step_log_alpha(eps)` should perform one
    /// leapfrog step at step size eps from the current state and return
    /// the log Metropolis acceptance ratio (h0 - h1).
    fn find_reasonable_epsilon<F>(&mut self, mut one_step_log_alpha: F)
    where
        F: FnMut(f64) -> f64,
    {
        let mut eps = self.epsilon;
        let log_half = (0.5f64).ln();
        let log_alpha = one_step_log_alpha(eps);

        // If the initial step gives NaN/Inf, shrink aggressively.
        if !log_alpha.is_finite() {
            for _ in 0..20 {
                eps *= 0.1;
                if eps < EPSILON_FLOOR { eps = EPSILON_FLOOR; break; }
                let la = one_step_log_alpha(eps);
                if la.is_finite() { break; }
            }
            self.epsilon = eps.max(EPSILON_FLOOR);
            self.mu = (10.0 * self.epsilon).ln();
            return;
        }

        // Determine direction: if log_alpha > ln(0.5), accept_prob > 0.5, increase eps.
        let a: f64 = if log_alpha > log_half { 1.0 } else { -1.0 };

        for _ in 0..50 {
            let la = one_step_log_alpha(eps);
            if !la.is_finite() || (a * la) < (a * log_half) {
                break;
            }
            eps *= 2.0f64.powf(a);
            if eps < EPSILON_FLOOR { eps = EPSILON_FLOOR; break; }
        }

        self.epsilon = eps.max(EPSILON_FLOOR);
        self.mu = (10.0 * self.epsilon).ln();
    }

    /// Update dual averaging with the acceptance probability from one HMC step.
    fn update_epsilon(&mut self, accept_prob: f64) {
        if !self.adapting { return; }
        // Guard against NaN (divergent trajectories).
        let ap = if accept_prob.is_nan() { 0.0 } else { accept_prob.clamp(0.0, 1.0) };
        self.da_count += 1;
        let m = self.da_count as f64;
        let w = 1.0 / (m + self.t0);
        self.h_bar = (1.0 - w) * self.h_bar + w * (self.target_accept - ap);
        let log_eps = self.mu - (m.sqrt() / self.gamma) * self.h_bar;
        self.epsilon = log_eps.exp().max(EPSILON_FLOOR);
        let mk = m.powf(-self.kappa);
        self.log_eps_bar = mk * log_eps + (1.0 - mk) * self.log_eps_bar;
    }

    /// Record a transition energy error; count as divergent if |delta_H| > threshold.
    fn record_energy_error(&mut self, delta_h: f64) {
        if !self.adapting && (delta_h.abs() > DIVERGENCE_THRESHOLD || delta_h.is_nan()) {
            self.n_divergent += 1;
        }
    }

    /// Accumulate a position sample for mass matrix estimation.
    fn observe(&mut self, q: &DVector<f64>) {
        if !self.adapting { return; }
        self.welford_n += 1;
        let n = self.welford_n as f64;
        for k in 0..self.p {
            let delta = q[k] - self.welford_mean[k];
            self.welford_mean[k] += delta / n;
            let delta2 = q[k] - self.welford_mean[k];
            self.welford_m2[k] += delta * delta2;
        }
    }

    /// Refresh the diagonal mass matrix from Welford variance estimates.
    fn refresh_mass_matrix(&mut self) {
        if self.welford_n < 20 { return; }
        let n = self.welford_n as f64;
        for k in 0..self.p {
            let var_k = self.welford_m2[k] / (n - 1.0);
            self.inv_mass[k] = var_k.max(1e-8);
        }
    }

    /// Freeze adaptation: set epsilon to the smoothed dual-averaging estimate.
    fn freeze(&mut self) {
        self.epsilon = self.log_eps_bar.exp().max(EPSILON_FLOOR);
        self.adapting = false;
    }

    /// Sample the number of leapfrog steps for this proposal.
    fn sample_l(&self, rng: &mut StdRng) -> usize {
        if self.l_min == self.l_max {
            self.l_min
        } else {
            rng.gen_range(self.l_min..=self.l_max)
        }
    }
}
// ---------------------------------------------------------------------------
// Scalar adaptive random-walk MH state: retained for p == 1 parameters
// where HMC overhead is unnecessary (single-dimensional targets are
// efficiently sampled with a tuned scalar random walk).
// ---------------------------------------------------------------------------

struct ScalarAdapt {
    step: f64,
    accepts:  u32,
    attempts: u32,
}

impl ScalarAdapt {
    fn new(init_step: f64) -> Self {
        ScalarAdapt { step: init_step, accepts: 0, attempts: 0 }
    }

    fn tune(&mut self, target: f64) {
        if self.attempts > 0 {
            let rate = self.accepts as f64 / self.attempts as f64;
            let factor = (rate / target).clamp(0.5, 2.0);
            self.step = (self.step * factor).max(1e-8);
        }
        self.accepts  = 0;
        self.attempts = 0;
    }
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
    target_accept: f64,
    seed: u64,
    verbose: bool,
    chain_id: usize,
    n_chains: usize,
    progress_fn: &dyn Fn(usize, usize, usize, usize, bool),
) -> (DMatrix<f64>, usize) {
    let mut rng = StdRng::seed_from_u64(seed);

    let mut state = init_state(data, priors, &mut rng);
    let n_post = n_iter - n_warmup;
    let n_params = state.n_params();

    // draws stored row-major: rows = posterior samples, cols = parameters
    let mut draws = DMatrix::<f64>::zeros(n_post, n_params);

    // -----------------------------------------------------------------------
    // Adaptation state for omega and rho.
    //
    // Strategy selection:
    //   p == 0  → nothing to sample
    //   p == 1  → scalar adaptive random-walk MH (efficient for 1D)
    //   p >= 2  → HMC-within-Gibbs with dual-averaging step-size adaptation
    //             and diagonal mass matrix estimation
    // -----------------------------------------------------------------------
    let use_hmc_om  = priors.p_om  >= 2;
    let use_hmc_rho = priors.p_rho >= 2;

    // HMC adaptation state (only used when p >= 2).
    let mut hmc_om  = if use_hmc_om  {
        Some(HmcAdapt::new(priors.p_om,  step_om_init,  target_accept, 5, 15))
    } else { None };
    let mut hmc_rho = if use_hmc_rho {
        Some(HmcAdapt::new(priors.p_rho, step_rho_init, target_accept, 5, 15))
    } else { None };

    // Scalar adaptive MH state (only used when p == 1).
    let mut scalar_om  = if priors.p_om  == 1 { Some(ScalarAdapt::new(step_om_init))  } else { None };
    let mut scalar_rho = if priors.p_rho == 1 { Some(ScalarAdapt::new(step_rho_init)) } else { None };

    let tune_window = 100usize;

    // Mass matrix refresh: at 60% of warmup, refresh the diagonal mass
    // matrix from accumulated Welford samples and reset the dual-averaging
    // step-size adaptation so it can re-converge with the new metric.
    let mass_refresh_iter_om  = (n_warmup as f64 * 0.60) as usize;
    let mass_refresh_iter_rho = (n_warmup as f64 * 0.60) as usize;

    // Cached current omega and rho linear predictors plus the LinearCache.
    let mut omega_cur: DVector<f64> = DVector::<f64>::zeros(0);
    let mut rho_cur:   DVector<f64> = DVector::<f64>::zeros(0);
    let mut cache: LinearCache = LinearCache {
        b0_fixed:   DVector::<f64>::zeros(0),
        b1_vals:    DVector::<f64>::zeros(0),
        b2_vals:    DVector::<f64>::zeros(0),
        re_contrib: DVector::<f64>::zeros(0),
    };

    let report_every = (n_iter / 10).max(1);

    // -----------------------------------------------------------------------
    // Find reasonable initial epsilon for HMC blocks (Algorithm 4, H&G 2014).
    // Requires one Gibbs sweep first to get a sensible starting state.
    // -----------------------------------------------------------------------
    {
        // Quick initial Gibbs sweep to populate state.
        if data.n_groups_b0 > 0 { sample_random_effects_nc(data, &mut state, &mut rng); }
        sample_linear_coefs(data, priors, &mut state, &mut rng);
        let init_cache = LinearCache::build(&state, data);
        let init_omega = &data.x_om  * &state.beta_om;
        let init_rho   = &data.x_rho * &state.beta_rho;

        if let Some(ref mut h) = hmc_om {
            let p = priors.p_om;
            let r_idx = priors.om_range();
            let pr_mean: Vec<f64> = priors.mean[r_idx.clone()].to_vec();
            let pr_sd:   Vec<f64> = priors.sd[r_idx.clone()].to_vec();
            let pr_lb:   Vec<f64> = priors.lb[r_idx.clone()].to_vec();
            let pr_ub:   Vec<f64> = priors.ub[r_idx.clone()].to_vec();
            let q0 = state.beta_om.clone();
            let sigma = state.sigma;
            let inv_mass = h.inv_mass.clone();
            h.find_reasonable_epsilon(|eps| {
                let normal_dist = Normal::new(0.0, 1.0f64).unwrap();
                let mut mom = DVector::<f64>::from_iterator(
                    p, (0..p).map(|k| normal_dist.sample(&mut rng) / inv_mass[k].sqrt()));
                let kinetic0: f64 = (0..p).map(|k| mom[k]*mom[k]*inv_mass[k]).sum::<f64>() * 0.5;
                let ll0 = init_cache.log_likelihood(data, &init_omega, &init_rho, sigma);
                let lp0 = log_truncated_normal_prior(q0.as_slice(), &pr_mean, &pr_sd, &pr_lb, &pr_ub);
                let h0_val = -ll0 - lp0 + kinetic0;
                // Single leapfrog step
                let grad_ll = init_cache.grad_beta_om(data, &init_omega, &init_rho, sigma, p);
                let grad_pr = grad_truncated_normal_prior(q0.as_slice(), &pr_mean, &pr_sd, &pr_lb, &pr_ub);
                let grad_lp: DVector<f64> = &grad_ll + &grad_pr;
                let mut q = q0.clone();
                for k in 0..p { mom[k] += 0.5 * eps * grad_lp[k]; }
                for k in 0..p {
                    q[k] += eps * mom[k] * inv_mass[k];
                    for _b in 0..MAX_REFLECTIONS {
                        if q[k] < pr_lb[k] { q[k] = 2.0*pr_lb[k]-q[k]; mom[k]=-mom[k]; }
                        else if q[k] > pr_ub[k] { q[k] = 2.0*pr_ub[k]-q[k]; mom[k]=-mom[k]; }
                        else { break; }
                    }
                }
                let omega_p = &data.x_om * &q;
                let gl2 = init_cache.grad_beta_om(data, &omega_p, &init_rho, sigma, p);
                let gp2 = grad_truncated_normal_prior(q.as_slice(), &pr_mean, &pr_sd, &pr_lb, &pr_ub);
                let grad2: DVector<f64> = &gl2 + &gp2;
                for k in 0..p { mom[k] += 0.5 * eps * grad2[k]; }
                let ll1 = init_cache.log_likelihood(data, &omega_p, &init_rho, sigma);
                let lp1 = log_truncated_normal_prior(q.as_slice(), &pr_mean, &pr_sd, &pr_lb, &pr_ub);
                let kin1: f64 = (0..p).map(|k| mom[k]*mom[k]*inv_mass[k]).sum::<f64>() * 0.5;
                let h1_val = -ll1 - lp1 + kin1;
                h0_val - h1_val  // log acceptance ratio
            });
        }

        if let Some(ref mut h) = hmc_rho {
            let p = priors.p_rho;
            let r_idx = priors.rho_range();
            let pr_mean: Vec<f64> = priors.mean[r_idx.clone()].to_vec();
            let pr_sd:   Vec<f64> = priors.sd[r_idx.clone()].to_vec();
            let pr_lb:   Vec<f64> = priors.lb[r_idx.clone()].to_vec();
            let pr_ub:   Vec<f64> = priors.ub[r_idx.clone()].to_vec();
            let q0 = state.beta_rho.clone();
            let sigma = state.sigma;
            let inv_mass_rho = h.inv_mass.clone();
            h.find_reasonable_epsilon(|eps| {
                let normal_dist = Normal::new(0.0, 1.0f64).unwrap();
                let mut mom = DVector::<f64>::from_iterator(
                    p, (0..p).map(|k| normal_dist.sample(&mut rng) / inv_mass_rho[k].sqrt()));
                let kinetic0: f64 = (0..p).map(|k| mom[k]*mom[k]*inv_mass_rho[k]).sum::<f64>() * 0.5;
                let ll0 = init_cache.log_likelihood(data, &init_omega, &init_rho, sigma);
                let lp0 = log_truncated_normal_prior(q0.as_slice(), &pr_mean, &pr_sd, &pr_lb, &pr_ub);
                let h0_val = -ll0 - lp0 + kinetic0;
                let grad_ll = init_cache.grad_beta_rho(data, &init_omega, &init_rho, sigma, p);
                let grad_pr = grad_truncated_normal_prior(q0.as_slice(), &pr_mean, &pr_sd, &pr_lb, &pr_ub);
                let grad_lp: DVector<f64> = &grad_ll + &grad_pr;
                let mut q = q0.clone();
                for k in 0..p { mom[k] += 0.5 * eps * grad_lp[k]; }
                for k in 0..p {
                    q[k] += eps * mom[k] * inv_mass_rho[k];
                    for _b in 0..MAX_REFLECTIONS {
                        if q[k] < pr_lb[k] { q[k] = 2.0*pr_lb[k]-q[k]; mom[k]=-mom[k]; }
                        else if q[k] > pr_ub[k] { q[k] = 2.0*pr_ub[k]-q[k]; mom[k]=-mom[k]; }
                        else { break; }
                    }
                }
                let rho_p = &data.x_rho * &q;
                let gl2 = init_cache.grad_beta_rho(data, &init_omega, &rho_p, sigma, p);
                let gp2 = grad_truncated_normal_prior(q.as_slice(), &pr_mean, &pr_sd, &pr_lb, &pr_ub);
                let grad2: DVector<f64> = &gl2 + &gp2;
                for k in 0..p { mom[k] += 0.5 * eps * grad2[k]; }
                let ll1 = init_cache.log_likelihood(data, &init_omega, &rho_p, sigma);
                let lp1 = log_truncated_normal_prior(q.as_slice(), &pr_mean, &pr_sd, &pr_lb, &pr_ub);
                let kin1: f64 = (0..p).map(|k| mom[k]*mom[k]*inv_mass_rho[k]).sum::<f64>() * 0.5;
                let h1_val = -ll1 - lp1 + kin1;
                h0_val - h1_val
            });
        }
    }

    for iter in 0..n_iter {
        // ---------------------------------------------------------------
        // Progress reporting
        // ---------------------------------------------------------------
        if verbose && iter % report_every == 0 {
            progress_fn(chain_id, n_chains, iter, n_iter, iter < n_warmup);
        }
        // ---------------------------------------------------------------
        // Block 1: Sample linear coefficients (+ random intercepts if present)
        // When RE are present, sample (beta_b0, u, beta_b1, beta_b2) jointly
        // to break the b0-u coupling that slows mixing in component-wise Gibbs.
        // ---------------------------------------------------------------
        if data.n_groups_b0 > 0 {
            sample_linear_coefs_joint(data, priors, &mut state, &mut rng);
        } else {
            sample_linear_coefs(data, priors, &mut state, &mut rng);
        }

        // Rebuild cache and linear predictors after the Gibbs blocks.
        cache     = LinearCache::build(&state, data);
        omega_cur = &data.x_om  * &state.beta_om;
        rho_cur   = &data.x_rho * &state.beta_rho;

        // ---------------------------------------------------------------
        // Block 2a: Update beta_om (HMC for p>=2, scalar MH for p==1)
        // ---------------------------------------------------------------
        if use_hmc_om {
            hmc_step_om(
                data, priors, &mut state,
                &cache, &mut omega_cur, &rho_cur,
                hmc_om.as_mut().unwrap(),
                &mut rng,
            );
        } else if priors.p_om == 1 {
            scalar_mh_step_om(
                data, priors, &mut state,
                &cache, &mut omega_cur, &rho_cur,
                scalar_om.as_mut().unwrap(),
                &mut rng,
            );
        }

        // ---------------------------------------------------------------
        // Block 2b: Update beta_rho (HMC for p>=2, scalar MH for p==1)
        // ---------------------------------------------------------------
        if use_hmc_rho {
            hmc_step_rho(
                data, priors, &mut state,
                &cache, &omega_cur, &mut rho_cur,
                hmc_rho.as_mut().unwrap(),
                &mut rng,
            );
        } else if priors.p_rho == 1 {
            scalar_mh_step_rho(
                data, priors, &mut state,
                &cache, &omega_cur, &mut rho_cur,
                scalar_rho.as_mut().unwrap(),
                &mut rng,
            );
        }

        // ---------------------------------------------------------------
        // Block 3a: Sample sigma (Gibbs, InvGamma conjugate)
        // ---------------------------------------------------------------
        sample_sigma(data, priors, &mut state, &mut rng);

        // ---------------------------------------------------------------
        // Block 3b: Sample sigma_u (Gibbs, InvGamma conjugate).
        // ---------------------------------------------------------------
        if data.n_groups_b0 > 0 {
            sample_sigma_u(priors, &mut state, &mut rng);
        }

        // ---------------------------------------------------------------
        // Adaptation during warmup
        // ---------------------------------------------------------------
        if iter < n_warmup {
            // Scalar MH: tune step sizes every window.
            if (iter + 1) % tune_window == 0 {
                if let Some(ref mut s) = scalar_om  { s.tune(0.234); }
                if let Some(ref mut s) = scalar_rho { s.tune(0.234); }
            }

            // HMC: accumulate Welford samples for mass matrix.
            if let Some(ref mut h) = hmc_om  { h.observe(&state.beta_om);  }
            if let Some(ref mut h) = hmc_rho { h.observe(&state.beta_rho); }

            // Mass matrix refresh at the mid-warmup window boundary.
            if iter == mass_refresh_iter_om {
                if let Some(ref mut h) = hmc_om {
                    h.refresh_mass_matrix();
                    // Reset dual averaging so epsilon can re-adapt to new metric.
                    h.da_count = 0;
                    h.h_bar = 0.0;
                    h.mu = (10.0 * h.epsilon).ln();
                    h.log_eps_bar = h.epsilon.ln();
                }
            }
            if iter == mass_refresh_iter_rho {
                if let Some(ref mut h) = hmc_rho {
                    h.refresh_mass_matrix();
                    h.da_count = 0;
                    h.h_bar = 0.0;
                    h.mu = (10.0 * h.epsilon).ln();
                    h.log_eps_bar = h.epsilon.ln();
                }
            }
        }

        // Freeze adaptation at end of warmup.
        if iter + 1 == n_warmup {
            if let Some(ref mut h) = hmc_om  { h.freeze(); }
            if let Some(ref mut h) = hmc_rho { h.freeze(); }
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

    let n_div = hmc_om.as_ref().map_or(0, |h| h.n_divergent)
        + hmc_rho.as_ref().map_or(0, |h| h.n_divergent);
    (draws, n_div)
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

    let gamma = vec![true; priors.p_b2];

    State {
        beta_b0,
        u_b0,
        beta_b1,
        beta_b2,
        beta_om,
        beta_rho,
        sigma: 1.0,
        sigma_u: 1.0,
        gamma,
        pi: 0.5,
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
    data: &ModelData,
    state: &mut State,
    rng: &mut StdRng,
) {
    let sigma2    = state.sigma   * state.sigma;
    let sigma_u2  = state.sigma_u * state.sigma_u;
    let n_groups  = data.n_groups_b0;

    // Fixed-part fitted values (no random effects)
    let omega = state.omega_vec(&data.x_om);
    let rho   = state.rho_vec(&data.x_rho);
    let d: DVector<f64> = data.tau.zip_map(&omega, |t, w| t - w);
    let s: DVector<f64> = d.zip_map(&rho, |di, ri| sigmoid(di * ri));

    let b0_fixed   = &data.x_b0 * &state.beta_b0;
    let b1_vals    = &data.x_b1 * &state.beta_b1;
    let b2_vals    = &data.x_b2 * &state.beta_b2;
    let b1_contrib: DVector<f64> = d.zip_map(&b1_vals, |di, b| di * b);
    let b2_contrib: DVector<f64> = d.zip_map(&s, |di, si| di * si).zip_map(&b2_vals, |ds, b| ds * b);
    let fixed_fit  = b0_fixed + b1_contrib + b2_contrib;

    // Sufficient statistics per group: sum of residuals and count
    let mut sum_r = vec![0.0f64; n_groups];
    let mut n_j   = vec![0u32;   n_groups];
    for i in 0..data.n {
        let g = data.group_b0[i] as usize;
        sum_r[g] += data.y[i] - fixed_fit[i];
        n_j[g]   += 1;
    }

    let normal = Normal::new(0.0, 1.0f64).unwrap();
    for j in 0..n_groups {
        let nj     = n_j[j] as f64;
        let tau_j  = nj * sigma_u2 / sigma2 + 1.0;
        let v_j    = 1.0 / tau_j;
        let mu_j   = v_j * state.sigma_u / sigma2 * sum_r[j];
        let z_j    = mu_j + v_j.sqrt() * normal.sample(rng);
        state.u_b0[j] = state.sigma_u * z_j;
    }
}

// ---------------------------------------------------------------------------
// Bounded-coefficient helper
//
// Sample a single linear coefficient from its 1-D truncated-normal full
// conditional via rejection sampling.
//
// x_col   : effective design column for this coefficient (length n).
//           e.g. for b1_k: d[i] * X_b1[i,k]
// y_resid : residuals with THIS coefficient's contribution already removed,
//           i.e. y_resid[i] = y[i] - (all other contributions)
// Returns a draw from TN(post_mean, post_sd, lb, ub).
// ---------------------------------------------------------------------------
fn sample_bounded_coef(
    x_col:      &[f64],
    y_resid:    &[f64],
    prior_mean: f64,
    prior_sd:   f64,
    lb:         f64,
    ub:         f64,
    sigma2:     f64,
    rng:        &mut StdRng,
) -> f64 {
    let prior_prec  = 1.0 / (prior_sd * prior_sd);
    let data_prec: f64 = x_col.iter().map(|&xi| xi * xi).sum::<f64>() / sigma2;
    let post_prec  = prior_prec + data_prec;
    let post_var   = 1.0 / post_prec;
    let data_cross: f64 = x_col.iter().zip(y_resid.iter())
        .map(|(&xi, &ri)| xi * ri)
        .sum::<f64>() / sigma2;
    let post_mean = post_var * (prior_prec * prior_mean + data_cross);
    let post_sd   = post_var.sqrt();

    let distr = Normal::new(post_mean, post_sd).unwrap();
    for _ in 0..50_000 {
        let v = distr.sample(rng);
        if (!lb.is_finite() || v >= lb) && (!ub.is_finite() || v <= ub) {
            return v;
        }
    }
    // Extremely tight truncation (should not occur with sensible priors).
    // Return the posterior mean clamped to the feasible region.
    let safe_lb = if lb.is_finite() { lb } else { f64::NEG_INFINITY };
    let safe_ub = if ub.is_finite() { ub } else { f64::INFINITY };
    post_mean.max(safe_lb).min(safe_ub)
}

// ---------------------------------------------------------------------------
// Block 1b: Sample (beta_b0, beta_b1, beta_b2) jointly | u, omega, rho, sigma
// (Normal-normal conjugate via Cholesky.)
// ---------------------------------------------------------------------------

fn sample_linear_coefs(
    data: &ModelData,
    priors: &Priors,
    state: &mut State,
    rng: &mut StdRng,
) {
    let n = data.n;
    let p_b0 = priors.p_b0;
    let p_b1 = priors.p_b1;
    let p_b2 = priors.p_b2;
    let p_lin = p_b0 + p_b1 + p_b2;
    let sigma2 = state.sigma * state.sigma;

    let omega = state.omega_vec(&data.x_om);
    let rho   = state.rho_vec(&data.x_rho);
    let d: Vec<f64> = (0..n).map(|i| data.tau[i] - omega[i]).collect();
    let s: Vec<f64> = (0..n).map(|i| sigmoid(d[i] * rho[i])).collect();

    // Build effective design matrix X_full (n × p_lin)
    let mut x_full = DMatrix::<f64>::zeros(n, p_lin);
    for i in 0..n {
        for k in 0..p_b0 {
            x_full[(i, k)] = data.x_b0[(i, k)];
        }
        for k in 0..p_b1 {
            x_full[(i, p_b0 + k)] = d[i] * data.x_b1[(i, k)];
        }
        for k in 0..p_b2 {
            x_full[(i, p_b0 + p_b1 + k)] = d[i] * s[i] * data.x_b2[(i, k)];
        }
    }

    let mut y_tilde = data.y.clone();
    if data.n_groups_b0 > 0 {
        for i in 0..n {
            let g = data.group_b0[i];
            if g >= 0 {
                y_tilde[i] -= state.u_b0[g as usize];
            }
        }
    }

    let b0_r = priors.b0_range();
    let b1_r = priors.b1_range();
    let b2_r = priors.b2_range();

    // ---- Bounded path: b0 conjugate Gibbs, then b1/b2 coordinate-wise ------
    // Replaces the joint-rejection approach. Avoids wasting full Cholesky work
    // when a bounded parameter is out-of-bounds.
    if priors.lin_has_finite_bounds() {
        // Step 1: sample b0 from its conjugate full conditional given b1, b2.
        // y_b0[i] = y_tilde[i] - (b1 contributions) - (b2 contributions)
        let y_b0: Vec<f64> = (0..n).map(|i| {
            let mut r = y_tilde[i];
            for k in 0..p_b1 { r -= state.beta_b1[k] * d[i] * data.x_b1[(i, k)]; }
            for k in 0..p_b2 { r -= state.beta_b2[k] * d[i] * s[i] * data.x_b2[(i, k)]; }
            r
        }).collect();
        let x_b0m = DMatrix::<f64>::from_fn(n, p_b0, |i, k| data.x_b0[(i, k)]);
        let mut prec_b0 = DVector::<f64>::zeros(p_b0);
        let mut mu_b0   = DVector::<f64>::zeros(p_b0);
        for (k, i) in priors.b0_range().enumerate() {
            prec_b0[k] = 1.0 / (priors.sd[i] * priors.sd[i]);
            mu_b0[k]   = priors.mean[i];
        }
        let y_b0_dv = DVector::from_vec(y_b0);
        let xt_b0 = x_b0m.transpose();
        let xtx_b0 = &xt_b0 * &x_b0m;
        let mut pp_b0 = xtx_b0 / sigma2;
        for k in 0..p_b0 { pp_b0[(k, k)] += prec_b0[k]; }
        let xty_b0: DVector<f64> = &xt_b0 * &y_b0_dv / sigma2;
        let rhs_b0 = xty_b0 + prec_b0.zip_map(&mu_b0, |p, m| p * m);
        let chol_b0 = pp_b0.cholesky().expect("b0 precision matrix not positive definite");
        let mu_post_b0 = chol_b0.solve(&rhs_b0);
        let normal = Normal::new(0.0, 1.0f64).unwrap();
        let z_b0 = DVector::from_iterator(p_b0, (0..p_b0).map(|_| normal.sample(rng)));
        let l_b0 = chol_b0.l();
        let w_b0 = l_b0.transpose().solve_upper_triangular(&z_b0)
            .expect("b0 triangular solve failed");
        let theta_b0 = mu_post_b0 + w_b0;
        // Accept b0 if in bounds (full rejection for b0 bounds, which are rare).
        let b0_ok = priors.b0_range().enumerate().all(|(k, i)| {
            theta_b0[k] >= priors.lb[i] && theta_b0[k] <= priors.ub[i]
        });
        if b0_ok {
            for k in 0..p_b0 { state.beta_b0[k] = theta_b0[k]; }
        }

        // Step 2: coordinate-wise updates for b1 and b2.
        // Running residuals = y_tilde - b0 - all b1 - all b2
        let mut y_run: Vec<f64> = (0..n).map(|i| {
            let mut r = y_tilde[i];
            for k in 0..p_b0 { r -= state.beta_b0[k] * data.x_b0[(i, k)]; }
            for k in 0..p_b1 { r -= state.beta_b1[k] * d[i] * data.x_b1[(i, k)]; }
            for k in 0..p_b2 { r -= state.beta_b2[k] * d[i] * s[i] * data.x_b2[(i, k)]; }
            r
        }).collect();

        for k in 0..p_b1 {
            let idx = priors.b1_range().start + k;
            let c_k: Vec<f64> = (0..n).map(|i| d[i] * data.x_b1[(i, k)]).collect();
            let cur = state.beta_b1[k];
            for i in 0..n { y_run[i] += cur * c_k[i]; }
            let new_val = sample_bounded_coef(
                &c_k, &y_run,
                priors.mean[idx], priors.sd[idx],
                priors.lb[idx], priors.ub[idx],
                sigma2, rng,
            );
            state.beta_b1[k] = new_val;
            for i in 0..n { y_run[i] -= new_val * c_k[i]; }
        }
        for k in 0..p_b2 {
            let idx = priors.b2_range().start + k;
            let c_k: Vec<f64> = (0..n).map(|i| d[i] * s[i] * data.x_b2[(i, k)]).collect();
            let cur = state.beta_b2[k];
            for i in 0..n { y_run[i] += cur * c_k[i]; }
            let new_val = sample_bounded_coef(
                &c_k, &y_run,
                priors.mean[idx], priors.sd[idx],
                priors.lb[idx], priors.ub[idx],
                sigma2, rng,
            );
            state.beta_b2[k] = new_val;
            for i in 0..n { y_run[i] -= new_val * c_k[i]; }
        }
        return; // bounded path done
    }
    // ---- Unbounded path: full joint conjugate Gibbs (unchanged) -------------

    let mut prec_prior = DVector::<f64>::zeros(p_lin);
    let mut mu_prior = DVector::<f64>::zeros(p_lin);
    for (k, i) in b0_r.enumerate() {
        prec_prior[k] = 1.0 / (priors.sd[i] * priors.sd[i]);
        mu_prior[k] = priors.mean[i];
    }
    for (k, i) in b1_r.enumerate() {
        prec_prior[p_b0 + k] = 1.0 / (priors.sd[i] * priors.sd[i]);
        mu_prior[p_b0 + k] = priors.mean[i];
    }
    for (k, i) in b2_r.enumerate() {
        prec_prior[p_b0 + p_b1 + k] = 1.0 / (priors.sd[i] * priors.sd[i]);
        mu_prior[p_b0 + p_b1 + k] = priors.mean[i];
    }

    let xt = x_full.transpose();
    let xtx = &xt * &x_full;
    let mut prec_post = xtx / sigma2;
    for k in 0..p_lin {
        prec_post[(k, k)] += prec_prior[k];
    }

    let xty: DVector<f64> = &xt * &y_tilde / sigma2;
    let rhs: DVector<f64> = xty + prec_prior.zip_map(&mu_prior, |p, m| p * m);

    let chol = prec_post.clone().cholesky().expect("Precision matrix not positive definite");
    let mu_post = chol.solve(&rhs);

    let normal = Normal::new(0.0, 1.0f64).unwrap();
    let z = DVector::from_iterator(p_lin, (0..p_lin).map(|_| normal.sample(rng)));
    let l = chol.l();
    let x = l.transpose().solve_upper_triangular(&z)
        .expect("Triangular solve failed");
    let theta = mu_post + x;

    // Unbounded path: always accept the joint draw.
    for k in 0..p_b0 {
        state.beta_b0[k] = theta[k];
    }
    for k in 0..p_b1 {
        state.beta_b1[k] = theta[p_b0 + k];
    }
    for k in 0..p_b2 {
        state.beta_b2[k] = theta[p_b0 + p_b1 + k];
    }
}

// ---------------------------------------------------------------------------
// Block 1: Joint sample of (beta_b0, u_1..u_J, beta_b1, beta_b2) | rest
//
// When random intercepts are present, b0 and u are tightly coupled because the
// likelihood depends on (b0 + u_j).  Sampling them in separate Gibbs blocks
// creates a slow-mixing ridge.  This function samples them jointly via the
// exact Normal-Normal conjugate full conditional, using an augmented design
// matrix [X_b0 | Z | d*X_b1 | d*s*X_b2] where Z is the group indicator matrix.
// ---------------------------------------------------------------------------

fn sample_linear_coefs_joint(
    data: &ModelData,
    priors: &Priors,
    state: &mut State,
    rng: &mut StdRng,
) {
    let n = data.n;
    let p_b0 = priors.p_b0;
    let p_b1 = priors.p_b1;
    let p_b2 = priors.p_b2;
    let j_groups = data.n_groups_b0;
    let sigma2 = state.sigma * state.sigma;
    let sigma_u2 = state.sigma_u * state.sigma_u;

    let omega = state.omega_vec(&data.x_om);
    let rho   = state.rho_vec(&data.x_rho);
    let d: Vec<f64> = (0..n).map(|i| data.tau[i] - omega[i]).collect();
    let s: Vec<f64> = (0..n).map(|i| sigmoid(d[i] * rho[i])).collect();

    // ---- Bounded path: (b0, u_j) conjugate Gibbs, then b1/b2 coordinate-wise
    // Separating the bounded b1/b2 parameters from the large (p_b0 + J) joint
    // prevents the rejection of the entire J-dimensional Cholesky draw every
    // time a single bounded parameter falls outside its constraint.
    if priors.lin_has_finite_bounds() {
        // Step 1: sample (b0, u_j) jointly given current b1 and b2.
        // Residuals with b1 and b2 subtracted.
        let y_b0u: Vec<f64> = (0..n).map(|i| {
            let mut r = data.y[i];
            for k in 0..p_b1 { r -= state.beta_b1[k] * d[i] * data.x_b1[(i, k)]; }
            for k in 0..p_b2 { r -= state.beta_b2[k] * d[i] * s[i] * data.x_b2[(i, k)]; }
            r
        }).collect();
        let p_b0_u = p_b0 + j_groups;
        let mut x_b0u = DMatrix::<f64>::zeros(n, p_b0_u);
        for i in 0..n {
            for k in 0..p_b0 { x_b0u[(i, k)] = data.x_b0[(i, k)]; }
            let g = data.group_b0[i];
            if g >= 0 { x_b0u[(i, p_b0 + g as usize)] = 1.0; }
        }
        let mut prec_b0u = DVector::<f64>::zeros(p_b0_u);
        let mut mu_b0u   = DVector::<f64>::zeros(p_b0_u);
        for (k, i) in priors.b0_range().enumerate() {
            prec_b0u[k] = 1.0 / (priors.sd[i] * priors.sd[i]);
            mu_b0u[k]   = priors.mean[i];
        }
        let u_prec = 1.0 / sigma_u2;
        for j in 0..j_groups {
            prec_b0u[p_b0 + j] = u_prec;
            mu_b0u[p_b0 + j]   = 0.0;
        }
        let y_b0u_dv = DVector::from_vec(y_b0u);
        let xt_b0u = x_b0u.transpose();
        let xtx_b0u = &xt_b0u * &x_b0u;
        let mut pp_b0u = xtx_b0u / sigma2;
        for k in 0..p_b0_u { pp_b0u[(k, k)] += prec_b0u[k]; }
        let xty_b0u: DVector<f64> = &xt_b0u * &y_b0u_dv / sigma2;
        let rhs_b0u = xty_b0u + prec_b0u.zip_map(&mu_b0u, |p, m| p * m);
        let chol_b0u = pp_b0u.cholesky()
            .expect("(b0, u) precision matrix not positive definite");
        let mu_post_b0u = chol_b0u.solve(&rhs_b0u);
        let normal = Normal::new(0.0, 1.0f64).unwrap();
        let z_b0u = DVector::from_iterator(
            p_b0_u, (0..p_b0_u).map(|_| normal.sample(rng))
        );
        let l_b0u = chol_b0u.l();
        let w_b0u = l_b0u.transpose().solve_upper_triangular(&z_b0u)
            .expect("(b0, u) triangular solve failed");
        let theta_b0u = mu_post_b0u + w_b0u;
        // Accept b0 if in bounds; u_j have no bounds so always store.
        let b0_ok = priors.b0_range().enumerate().all(|(k, i)| {
            theta_b0u[k] >= priors.lb[i] && theta_b0u[k] <= priors.ub[i]
        });
        if b0_ok {
            for k in 0..p_b0 { state.beta_b0[k] = theta_b0u[k]; }
        }
        for j in 0..j_groups { state.u_b0[j] = theta_b0u[p_b0 + j]; }

        // Step 2: coordinate-wise updates for b1 and b2.
        // Running residuals = y - b0 - u_j - all b1 - all b2
        let mut y_run: Vec<f64> = (0..n).map(|i| {
            let mut r = data.y[i];
            for k in 0..p_b0 { r -= state.beta_b0[k] * data.x_b0[(i, k)]; }
            let g = data.group_b0[i];
            if g >= 0 { r -= state.u_b0[g as usize]; }
            for k in 0..p_b1 { r -= state.beta_b1[k] * d[i] * data.x_b1[(i, k)]; }
            for k in 0..p_b2 { r -= state.beta_b2[k] * d[i] * s[i] * data.x_b2[(i, k)]; }
            r
        }).collect();

        for k in 0..p_b1 {
            let idx = priors.b1_range().start + k;
            let c_k: Vec<f64> = (0..n).map(|i| d[i] * data.x_b1[(i, k)]).collect();
            let cur = state.beta_b1[k];
            for i in 0..n { y_run[i] += cur * c_k[i]; }
            let new_val = sample_bounded_coef(
                &c_k, &y_run,
                priors.mean[idx], priors.sd[idx],
                priors.lb[idx], priors.ub[idx],
                sigma2, rng,
            );
            state.beta_b1[k] = new_val;
            for i in 0..n { y_run[i] -= new_val * c_k[i]; }
        }
        for k in 0..p_b2 {
            let idx = priors.b2_range().start + k;
            let c_k: Vec<f64> = (0..n).map(|i| d[i] * s[i] * data.x_b2[(i, k)]).collect();
            let cur = state.beta_b2[k];
            for i in 0..n { y_run[i] += cur * c_k[i]; }
            let new_val = sample_bounded_coef(
                &c_k, &y_run,
                priors.mean[idx], priors.sd[idx],
                priors.lb[idx], priors.ub[idx],
                sigma2, rng,
            );
            state.beta_b2[k] = new_val;
            for i in 0..n { y_run[i] -= new_val * c_k[i]; }
        }
        return; // bounded path done
    }
    // ---- Unbounded path: two-block NC Gibbs (Option B) ----------------------
    //
    // Replaces the single (J + p_lin) x (J + p_lin) joint Cholesky introduced
    // in commit 5bb616c with two smaller, exact conjugate Gibbs steps:
    //
    //   Block A  [O(J)]:       NC step for u_j.
    //     Reparameterise u_j = sigma_u * z_j and sample each z_j independently
    //     from its scalar N(mu_j, v_j) full conditional.  The NC prior
    //     z_j ~ N(0,1) breaks the b0--u_j posterior ridge that the joint
    //     sampler of 5bb616c was designed to cure, because the prior no longer
    //     depends on sigma_u.
    //
    //   Block B  [O(p_lin^3)]: Small joint Cholesky for (b0, b1, b2).
    //     With u fixed, the full conditional for (b0, b1, b2) is a
    //     p_lin-dimensional normal (p_lin = p_b0 + p_b1 + p_b2, typically 3).
    //     sample_linear_coefs() handles this block; it already subtracts the
    //     current u_b0 contributions from y before forming the system.
    //
    // Cost comparison for J = 25, p_lin = 3, n = 250:
    //   Old joint:  O((J+p_lin)^3/3) ~ 5500 + O(n*(J+p_lin)^2) ~ 160 000
    //   NC + small: O(J)              ~   25 + O(n*p_lin^2)     ~   2 250
    //   Approx. 70x fewer flops per iteration on representative data.
    //
    // Mixing trade-off:
    //   Commit 5bb616c showed the joint sampler improved b0/sigma_u Rhat from
    //   ~1.18 to 1.00 and ESS ~37x on one dataset, at ~87x cost per iteration
    //   => net ~0.4x ESS/second.  The NC reparameterisation also breaks the
    //   b0--u ridge; if mixing is still poor in practice, consider Option A:
    //
    //   Option A (two-block partial split):
    //     Block A: sample (b0, u_1..u_J) jointly in a (p_b0 + J) matrix --
    //              captures the full b0-u posterior geometry at O((p_b0+J)^3).
    //     Block B: sample (b1, b2) jointly in a 2x2 (or similar) matrix.
    //     Cost: O((p_b0+J)^3/3) ~ 4600, roughly 20% cheaper than the old full
    //     joint but with the same mixing benefit for b0 and u.  Implement by
    //     extracting the bounded-path (b0, u_j) Cholesky block above into a
    //     shared helper and calling it unconditionally.
    sample_random_effects_nc(data, state, rng);
    sample_linear_coefs(data, priors, state, rng);
}

// ---------------------------------------------------------------------------
// Block 2a: HMC-within-Gibbs for beta_om  (p >= 2)
//
// Runs a short HMC trajectory (L leapfrog steps) with boundary reflection
// and diagonal mass matrix preconditioning.  Step size is adapted via
// Nesterov dual averaging during warmup.
// ---------------------------------------------------------------------------

fn hmc_step_om(
    data: &ModelData,
    priors: &Priors,
    state: &mut State,
    cache: &LinearCache,
    omega_cur: &mut DVector<f64>,
    rho_cur:   &DVector<f64>,
    adapt:     &mut HmcAdapt,
    rng: &mut StdRng,
) {
    let p = priors.p_om;
    let r_idx = priors.om_range();
    let prior_mean: Vec<f64> = priors.mean[r_idx.clone()].to_vec();
    let prior_sd:   Vec<f64> = priors.sd[r_idx.clone()].to_vec();
    let prior_lb:   Vec<f64> = priors.lb[r_idx.clone()].to_vec();
    let prior_ub:   Vec<f64> = priors.ub[r_idx.clone()].to_vec();

    // Potential energy U = -(log-likelihood + log-prior).
    // We work with the negative log-posterior so the leapfrog integrates
    // the Hamiltonian H = U(q) + K(p) = U(q) + 0.5 * p^T M^{-1} p.

    let q0 = state.beta_om.clone();
    let omega0 = omega_cur.clone();

    // Gradient of log-posterior = grad_ll + grad_prior.
    // grad_U = -grad_logpost.
    let grad_ll = cache.grad_beta_om(data, &omega0, rho_cur, state.sigma, p);
    let grad_pr = grad_truncated_normal_prior(
        q0.as_slice(), &prior_mean, &prior_sd, &prior_lb, &prior_ub,
    );
    let mut grad_logpost = &grad_ll + &grad_pr;

    // Sample momentum: p_k ~ N(0, M_kk) where M_kk = 1/inv_mass[k].
    let normal = Normal::new(0.0, 1.0f64).unwrap();
    let mut mom = DVector::<f64>::from_iterator(
        p, (0..p).map(|k| normal.sample(rng) / adapt.inv_mass[k].sqrt())
    );

    // Initial kinetic energy: K = 0.5 * Σ p_k² * inv_mass_k.
    let kinetic0: f64 = (0..p).map(|k| mom[k] * mom[k] * adapt.inv_mass[k]).sum::<f64>() * 0.5;
    let ll0 = cache.log_likelihood(data, &omega0, rho_cur, state.sigma);
    let lp0 = log_truncated_normal_prior(
        q0.as_slice(), &prior_mean, &prior_sd, &prior_lb, &prior_ub,
    );
    let h0 = -ll0 - lp0 + kinetic0;

    let mut q = q0.clone();
    let eps = adapt.epsilon;
    let n_leapfrog = adapt.sample_l(rng);

    // --- Leapfrog integration with boundary reflection ---
    for _step in 0..n_leapfrog {
        // Half-step momentum
        for k in 0..p {
            mom[k] += 0.5 * eps * grad_logpost[k];
        }

        // Full-step position (with reflection at bounds)
        for k in 0..p {
            q[k] += eps * mom[k] * adapt.inv_mass[k];
            // Reflect off bounds.  Loop handles the (rare) case where
            // the step is large enough to cross a boundary more than once.
            for _bounce in 0..MAX_REFLECTIONS {
                if q[k] < prior_lb[k] {
                    q[k] = 2.0 * prior_lb[k] - q[k];
                    mom[k] = -mom[k];
                } else if q[k] > prior_ub[k] {
                    q[k] = 2.0 * prior_ub[k] - q[k];
                    mom[k] = -mom[k];
                } else {
                    break;
                }
            }
        }

        // Recompute gradient at new position.
        let omega_prop = &data.x_om * &q;
        let gl = cache.grad_beta_om(data, &omega_prop, rho_cur, state.sigma, p);
        let gp = grad_truncated_normal_prior(
            q.as_slice(), &prior_mean, &prior_sd, &prior_lb, &prior_ub,
        );
        grad_logpost = &gl + &gp;

        // Half-step momentum
        for k in 0..p {
            mom[k] += 0.5 * eps * grad_logpost[k];
        }
    }

    // --- Accept/reject ---
    let omega_prop = &data.x_om * &q;
    let ll_prop = cache.log_likelihood(data, &omega_prop, rho_cur, state.sigma);
    let lp_prop = log_truncated_normal_prior(
        q.as_slice(), &prior_mean, &prior_sd, &prior_lb, &prior_ub,
    );
    let kinetic_prop: f64 = (0..p).map(|k| mom[k] * mom[k] * adapt.inv_mass[k]).sum::<f64>() * 0.5;
    let h_prop = -ll_prop - lp_prop + kinetic_prop;

    let delta_h = h_prop - h0;
    let log_alpha = -delta_h;
    // NaN-safe: treat NaN/Inf energy errors as rejection.
    let accept_prob = if log_alpha.is_finite() {
        1.0_f64.min(log_alpha.exp())
    } else {
        0.0
    };
    adapt.record_energy_error(delta_h);

    if rng.gen::<f64>() < accept_prob {
        state.beta_om = q;
        *omega_cur = omega_prop;
    }

    // Update dual averaging.
    adapt.update_epsilon(accept_prob);
}

// ---------------------------------------------------------------------------
// Block 2b: HMC-within-Gibbs for beta_rho  (p >= 2)
// ---------------------------------------------------------------------------

fn hmc_step_rho(
    data: &ModelData,
    priors: &Priors,
    state: &mut State,
    cache: &LinearCache,
    omega_cur: &DVector<f64>,
    rho_cur:   &mut DVector<f64>,
    adapt:     &mut HmcAdapt,
    rng: &mut StdRng,
) {
    let p = priors.p_rho;
    let r_idx = priors.rho_range();
    let prior_mean: Vec<f64> = priors.mean[r_idx.clone()].to_vec();
    let prior_sd:   Vec<f64> = priors.sd[r_idx.clone()].to_vec();
    let prior_lb:   Vec<f64> = priors.lb[r_idx.clone()].to_vec();
    let prior_ub:   Vec<f64> = priors.ub[r_idx.clone()].to_vec();

    let q0 = state.beta_rho.clone();
    let rho0 = rho_cur.clone();

    let grad_ll = cache.grad_beta_rho(data, omega_cur, &rho0, state.sigma, p);
    let grad_pr = grad_truncated_normal_prior(
        q0.as_slice(), &prior_mean, &prior_sd, &prior_lb, &prior_ub,
    );
    let mut grad_logpost = &grad_ll + &grad_pr;

    let normal = Normal::new(0.0, 1.0f64).unwrap();
    let mut mom = DVector::<f64>::from_iterator(
        p, (0..p).map(|k| normal.sample(rng) / adapt.inv_mass[k].sqrt())
    );

    let kinetic0: f64 = (0..p).map(|k| mom[k] * mom[k] * adapt.inv_mass[k]).sum::<f64>() * 0.5;
    let ll0 = cache.log_likelihood(data, omega_cur, &rho0, state.sigma);
    let lp0 = log_truncated_normal_prior(
        q0.as_slice(), &prior_mean, &prior_sd, &prior_lb, &prior_ub,
    );
    let h0 = -ll0 - lp0 + kinetic0;

    let mut q = q0.clone();
    let eps = adapt.epsilon;
    let n_leapfrog = adapt.sample_l(rng);

    for _step in 0..n_leapfrog {
        for k in 0..p {
            mom[k] += 0.5 * eps * grad_logpost[k];
        }

        for k in 0..p {
            q[k] += eps * mom[k] * adapt.inv_mass[k];
            for _bounce in 0..MAX_REFLECTIONS {
                if q[k] < prior_lb[k] {
                    q[k] = 2.0 * prior_lb[k] - q[k];
                    mom[k] = -mom[k];
                } else if q[k] > prior_ub[k] {
                    q[k] = 2.0 * prior_ub[k] - q[k];
                    mom[k] = -mom[k];
                } else {
                    break;
                }
            }
        }

        let rho_prop = &data.x_rho * &q;
        let gl = cache.grad_beta_rho(data, omega_cur, &rho_prop, state.sigma, p);
        let gp = grad_truncated_normal_prior(
            q.as_slice(), &prior_mean, &prior_sd, &prior_lb, &prior_ub,
        );
        grad_logpost = &gl + &gp;

        for k in 0..p {
            mom[k] += 0.5 * eps * grad_logpost[k];
        }
    }

    let rho_prop = &data.x_rho * &q;
    let ll_prop = cache.log_likelihood(data, omega_cur, &rho_prop, state.sigma);
    let lp_prop = log_truncated_normal_prior(
        q.as_slice(), &prior_mean, &prior_sd, &prior_lb, &prior_ub,
    );
    let kinetic_prop: f64 = (0..p).map(|k| mom[k] * mom[k] * adapt.inv_mass[k]).sum::<f64>() * 0.5;
    let h_prop = -ll_prop - lp_prop + kinetic_prop;

    let delta_h = h_prop - h0;
    let log_alpha = -delta_h;
    let accept_prob = if log_alpha.is_finite() {
        1.0_f64.min(log_alpha.exp())
    } else {
        0.0
    };
    adapt.record_energy_error(delta_h);

    if rng.gen::<f64>() < accept_prob {
        state.beta_rho = q;
        *rho_cur = rho_prop;
    }

    adapt.update_epsilon(accept_prob);
}

// ---------------------------------------------------------------------------
// Scalar adaptive random-walk MH for p == 1 parameters.
// Retained because a 1-D HMC has no advantage over a well-tuned scalar RWM
// and the overhead of leapfrog integration is unnecessary.
// ---------------------------------------------------------------------------

fn scalar_mh_step_om(
    data: &ModelData,
    priors: &Priors,
    state: &mut State,
    cache: &LinearCache,
    omega_cur: &mut DVector<f64>,
    rho_cur:   &DVector<f64>,
    adapt:     &mut ScalarAdapt,
    rng: &mut StdRng,
) {
    let r_idx = priors.om_range();
    let prior_mean: Vec<f64> = priors.mean[r_idx.clone()].to_vec();
    let prior_sd:   Vec<f64> = priors.sd[r_idx.clone()].to_vec();
    let prior_lb:   Vec<f64> = priors.lb[r_idx.clone()].to_vec();
    let prior_ub:   Vec<f64> = priors.ub[r_idx.clone()].to_vec();

    let normal = Normal::new(0.0, adapt.step).unwrap();
    let delta = normal.sample(rng);
    let proposed_val = state.beta_om[0] + delta;
    adapt.attempts += 1;

    if proposed_val < prior_lb[0] || proposed_val > prior_ub[0] { return; }

    let mut beta_prop = state.beta_om.clone();
    beta_prop[0] = proposed_val;
    let lp_prop = log_truncated_normal_prior(
        beta_prop.as_slice(), &prior_mean, &prior_sd, &prior_lb, &prior_ub,
    );
    if lp_prop == f64::NEG_INFINITY { return; }
    let lp_cur = log_truncated_normal_prior(
        state.beta_om.as_slice(), &prior_mean, &prior_sd, &prior_lb, &prior_ub,
    );

    let mut omega_prop = omega_cur.clone();
    for i in 0..data.n {
        omega_prop[i] += delta * data.x_om[(i, 0)];
    }

    let ll_cur  = cache.log_likelihood(data, omega_cur, rho_cur, state.sigma);
    let ll_prop = cache.log_likelihood(data, &omega_prop, rho_cur, state.sigma);
    let log_alpha = (ll_prop + lp_prop) - (ll_cur + lp_cur);
    if rng.gen::<f64>().ln() < log_alpha {
        state.beta_om[0] = proposed_val;
        *omega_cur = omega_prop;
        adapt.accepts += 1;
    }
}

fn scalar_mh_step_rho(
    data: &ModelData,
    priors: &Priors,
    state: &mut State,
    cache: &LinearCache,
    omega_cur: &DVector<f64>,
    rho_cur:   &mut DVector<f64>,
    adapt:     &mut ScalarAdapt,
    rng: &mut StdRng,
) {
    let r_idx = priors.rho_range();
    let prior_mean: Vec<f64> = priors.mean[r_idx.clone()].to_vec();
    let prior_sd:   Vec<f64> = priors.sd[r_idx.clone()].to_vec();
    let prior_lb:   Vec<f64> = priors.lb[r_idx.clone()].to_vec();
    let prior_ub:   Vec<f64> = priors.ub[r_idx.clone()].to_vec();

    let normal = Normal::new(0.0, adapt.step).unwrap();
    let delta = normal.sample(rng);
    let proposed_val = state.beta_rho[0] + delta;
    adapt.attempts += 1;

    if proposed_val < prior_lb[0] || proposed_val > prior_ub[0] { return; }

    let mut beta_prop = state.beta_rho.clone();
    beta_prop[0] = proposed_val;
    let lp_prop = log_truncated_normal_prior(
        beta_prop.as_slice(), &prior_mean, &prior_sd, &prior_lb, &prior_ub,
    );
    if lp_prop == f64::NEG_INFINITY { return; }
    let lp_cur = log_truncated_normal_prior(
        state.beta_rho.as_slice(), &prior_mean, &prior_sd, &prior_lb, &prior_ub,
    );

    let mut rho_prop = rho_cur.clone();
    for i in 0..data.n {
        rho_prop[i] += delta * data.x_rho[(i, 0)];
    }

    let ll_cur  = cache.log_likelihood(data, omega_cur, rho_cur, state.sigma);
    let ll_prop = cache.log_likelihood(data, omega_cur, &rho_prop, state.sigma);
    let log_alpha = (ll_prop + lp_prop) - (ll_cur + lp_cur);
    if rng.gen::<f64>().ln() < log_alpha {
        state.beta_rho[0] = proposed_val;
        *rho_cur = rho_prop;
        adapt.accepts += 1;
    }
}

// ---------------------------------------------------------------------------
// Block 3a: Sample sigma^2 | everything  (InvGamma conjugate)
// ---------------------------------------------------------------------------

fn sample_sigma(
    data: &ModelData,
    priors: &Priors,
    state: &mut State,
    rng: &mut StdRng,
) {
    let mu = state.means(data);
    let ssr: f64 = (0..data.n).map(|i| (data.y[i] - mu[i]).powi(2)).sum();

    let shape_post = priors.sigma_shape + 0.5 * data.n as f64;
    let rate_post = priors.sigma_scale + 0.5 * ssr;

    let g = Gamma::new(shape_post, 1.0 / rate_post).unwrap();
    let x = g.sample(rng);
    state.sigma = (1.0 / x).sqrt();
}

// ---------------------------------------------------------------------------
// Block 3b: Sample sigma_u^2 | u  (InvGamma conjugate Gibbs)
// ---------------------------------------------------------------------------

fn sample_sigma_u(priors: &Priors, state: &mut State, rng: &mut StdRng) {
    let m       = state.u_b0.len() as f64;
    let ss_u: f64 = state.u_b0.iter().map(|u| u * u).sum();

    let shape_post = priors.sigma_u_shape + 0.5 * m;
    let rate_post  = priors.sigma_u_scale  + 0.5 * ss_u;

    let g = Gamma::new(shape_post, 1.0 / rate_post).unwrap();
    state.sigma_u = (1.0 / g.sample(rng)).sqrt();
}

// ===========================================================================
// Spike-and-slab variable selection sampler
// ===========================================================================

use crate::model::SpikeSlabConfig;

// ---------------------------------------------------------------------------
// Sample gamma indicators (Kuo & Mallick 1998)
//
// For each eligible b2 coefficient k:
//   1. Compute log-likelihood with gamma_k = 1 (current state)
//   2. Temporarily zero beta_b2[k] (and mapped om/rho coefficients)
//   3. Compute log-likelihood with gamma_k = 0
//   4. Sample gamma_k from Bernoulli with posterior inclusion odds
//   5. If gamma_k = 0: keep zeroed; if gamma_k = 1: restore
// ---------------------------------------------------------------------------

fn sample_gamma(
    data: &ModelData,
    priors: &Priors,
    ss: &SpikeSlabConfig,
    state: &mut State,
    rng: &mut StdRng,
) {
    let p_b2 = priors.p_b2;

    for k in 0..p_b2 {
        if !ss.spike_mask[k] {
            continue; // not eligible for spike-and-slab
        }

        // Use per-coefficient pi when fixed, or shared state.pi when learning
        let pi_k = if ss.beta_a > 0.0 { state.pi } else { ss.pi_init[k] };
        let log_odds_prior = (pi_k / (1.0 - pi_k)).ln();

        // --- Compute log-likelihood with gamma_k = 1 ---
        // Save current values
        let saved_b2 = state.beta_b2[k];
        let saved_om = if ss.om_map[k] >= 0 {
            Some(state.beta_om[ss.om_map[k] as usize])
        } else { None };
        let saved_rho = if ss.rho_map[k] >= 0 {
            Some(state.beta_rho[ss.rho_map[k] as usize])
        } else { None };

        // Ensure included state: restore values if gamma was 0
        if !state.gamma[k] {
            state.beta_b2[k] = saved_b2; // already saved
            // om/rho were zeroed when gamma was 0, but we keep the
            // "latent" values in the state -- they are sampled from
            // the prior when gamma is off.  For the likelihood comparison
            // we need to evaluate with and without, so restore the latent values.
        }
        // Actually: when gamma_k was 0 in the Kuo-Mallick scheme, the latent
        // coefficients were drawn from their prior but the mean function
        // uses gamma_k * beta.  So for ll_include we temporarily set the
        // coefficients to their stored values.
        let mu_include = state.means(data);
        let ll_include = compute_ll_from_mu(data, &mu_include, state.sigma);

        // --- Compute log-likelihood with gamma_k = 0 ---
        state.beta_b2[k] = 0.0;
        if ss.om_map[k] >= 0 {
            state.beta_om[ss.om_map[k] as usize] = 0.0;
        }
        if ss.rho_map[k] >= 0 {
            state.beta_rho[ss.rho_map[k] as usize] = 0.0;
        }
        let mu_exclude = state.means(data);
        let ll_exclude = compute_ll_from_mu(data, &mu_exclude, state.sigma);

        // --- Posterior inclusion probability ---
        let log_odds_post = log_odds_prior + (ll_include - ll_exclude);
        let p_include = sigmoid(log_odds_post);

        // --- Sample gamma_k ---
        let u: f64 = rng.gen();
        if u < p_include {
            // gamma_k = 1: restore coefficients
            state.gamma[k] = true;
            state.beta_b2[k] = saved_b2;
            if let Some(v) = saved_om {
                state.beta_om[ss.om_map[k] as usize] = v;
            }
            if let Some(v) = saved_rho {
                state.beta_rho[ss.rho_map[k] as usize] = v;
            }
        } else {
            // gamma_k = 0: keep zeroed
            state.gamma[k] = false;
            // beta_b2[k] already 0, om/rho already 0
        }
    }
}

// ---------------------------------------------------------------------------
// Sample pi | gamma  (Beta-Bernoulli conjugate Gibbs step)
//
// pi | gamma ~ Beta(a + sum(gamma_eligible), b + K - sum(gamma_eligible))
// where K = number of spike-eligible coefficients.
// Only called when beta_a > 0 (i.e., learning pi).
// ---------------------------------------------------------------------------

fn sample_pi(
    ss: &SpikeSlabConfig,
    state: &mut State,
    rng: &mut StdRng,
) {
    let mut n_included = 0usize;
    let mut n_eligible = 0usize;
    for k in 0..state.gamma.len() {
        if ss.spike_mask[k] {
            n_eligible += 1;
            if state.gamma[k] {
                n_included += 1;
            }
        }
    }
    let a_post = ss.beta_a + n_included as f64;
    let b_post = ss.beta_b + (n_eligible - n_included) as f64;

    // Sample from Beta(a_post, b_post) using Gamma trick:
    // X ~ Gamma(a), Y ~ Gamma(b) => X/(X+Y) ~ Beta(a,b)
    let gx = Gamma::new(a_post, 1.0).unwrap();
    let gy = Gamma::new(b_post, 1.0).unwrap();
    let x = gx.sample(rng);
    let y = gy.sample(rng);
    state.pi = x / (x + y);
}

/// Compute log-likelihood from precomputed mean vector.
fn compute_ll_from_mu(data: &ModelData, mu: &DVector<f64>, sigma: f64) -> f64 {
    let sigma2 = sigma * sigma;
    let log_norm = 0.5 * (std::f64::consts::TAU * sigma2).ln();
    let mut ll = 0.0f64;
    for i in 0..data.n {
        let r = data.y[i] - mu[i];
        ll -= 0.5 * r * r / sigma2 + log_norm;
    }
    ll
}

// ---------------------------------------------------------------------------
// Modified linear coefficient sampling for spike-and-slab.
//
// "Included" b2 coefficients (gamma_k = 1) are sampled jointly with b0, b1
// via the standard conjugate Gibbs block.  "Excluded" b2 coefficients
// (gamma_k = 0) are drawn from their prior independently (keeping them
// as latent variables for the Kuo-Mallick scheme).
// ---------------------------------------------------------------------------

fn sample_linear_coefs_ss(
    data: &ModelData,
    priors: &Priors,
    _ss: &SpikeSlabConfig,
    state: &mut State,
    rng: &mut StdRng,
) {
    let n = data.n;
    let p_b0 = priors.p_b0;
    let p_b1 = priors.p_b1;
    let p_b2 = priors.p_b2;
    let sigma2 = state.sigma * state.sigma;

    let omega = state.omega_vec(&data.x_om);
    let rho   = state.rho_vec(&data.x_rho);
    let d: Vec<f64> = (0..n).map(|i| data.tau[i] - omega[i]).collect();
    let s: Vec<f64> = (0..n).map(|i| sigmoid(d[i] * rho[i])).collect();

    // Count included b2 coefficients
    let b2_included: Vec<usize> = (0..p_b2).filter(|&k| state.gamma[k]).collect();
    let n_b2_inc = b2_included.len();
    let p_lin = p_b0 + p_b1 + n_b2_inc;

    // Build effective design matrix X_full (n × p_lin)
    // Columns: [X_b0 | d*X_b1 | d*s*X_b2_included]
    let mut x_full = DMatrix::<f64>::zeros(n, p_lin);
    for i in 0..n {
        for k in 0..p_b0 {
            x_full[(i, k)] = data.x_b0[(i, k)];
        }
        for k in 0..p_b1 {
            x_full[(i, p_b0 + k)] = d[i] * data.x_b1[(i, k)];
        }
        for (j, &k) in b2_included.iter().enumerate() {
            x_full[(i, p_b0 + p_b1 + j)] = d[i] * s[i] * data.x_b2[(i, k)];
        }
    }

    // Subtract random intercepts from y
    let mut y_tilde = data.y.clone();
    if data.n_groups_b0 > 0 {
        for i in 0..n {
            let g = data.group_b0[i];
            if g >= 0 {
                y_tilde[i] -= state.u_b0[g as usize];
            }
        }
    }
    // Subtract contribution of excluded b2 coefficients (which are zero,
    // so nothing to subtract)

    // Prior precision and mean for included coefficients
    let b0_r = priors.b0_range();
    let b1_r = priors.b1_range();
    let b2_r = priors.b2_range();

    // ---- Bounded path: b0 conjugate Gibbs + b1/included-b2 coordinate-wise -
    if priors.lin_has_finite_bounds() {
        // Step 1: sample b0 from conjugate full conditional given b1, b2.
        let y_b0: Vec<f64> = (0..n).map(|i| {
            let mut r = y_tilde[i];
            for k in 0..p_b1 { r -= state.beta_b1[k] * d[i] * data.x_b1[(i, k)]; }
            for &orig_k in &b2_included {
                r -= state.beta_b2[orig_k] * d[i] * s[i] * data.x_b2[(i, orig_k)];
            }
            r
        }).collect();
        let x_b0m = DMatrix::<f64>::from_fn(n, p_b0, |i, k| data.x_b0[(i, k)]);
        let mut prec_b0 = DVector::<f64>::zeros(p_b0);
        let mut mu_b0   = DVector::<f64>::zeros(p_b0);
        for (k, i) in priors.b0_range().enumerate() {
            prec_b0[k] = 1.0 / (priors.sd[i] * priors.sd[i]);
            mu_b0[k]   = priors.mean[i];
        }
        let y_b0_dv = DVector::from_vec(y_b0);
        let xt_b0 = x_b0m.transpose();
        let xtx_b0 = &xt_b0 * &x_b0m;
        let mut pp_b0 = xtx_b0 / sigma2;
        for k in 0..p_b0 { pp_b0[(k, k)] += prec_b0[k]; }
        let xty_b0: DVector<f64> = &xt_b0 * &y_b0_dv / sigma2;
        let rhs_b0 = xty_b0 + prec_b0.zip_map(&mu_b0, |p, m| p * m);
        let chol_b0 = pp_b0.cholesky().expect("b0 precision matrix not positive definite");
        let mu_post_b0 = chol_b0.solve(&rhs_b0);
        let normal = Normal::new(0.0, 1.0f64).unwrap();
        let z_b0 = DVector::from_iterator(p_b0, (0..p_b0).map(|_| normal.sample(rng)));
        let l_b0 = chol_b0.l();
        let w_b0 = l_b0.transpose().solve_upper_triangular(&z_b0)
            .expect("b0 triangular solve failed");
        let theta_b0 = mu_post_b0 + w_b0;
        let b0_ok = priors.b0_range().enumerate().all(|(k, i)| {
            theta_b0[k] >= priors.lb[i] && theta_b0[k] <= priors.ub[i]
        });
        if b0_ok { for k in 0..p_b0 { state.beta_b0[k] = theta_b0[k]; } }

        // Step 2: coordinate-wise b1 and included b2.
        let mut y_run: Vec<f64> = (0..n).map(|i| {
            let mut r = y_tilde[i];
            for k in 0..p_b0 { r -= state.beta_b0[k] * data.x_b0[(i, k)]; }
            for k in 0..p_b1 { r -= state.beta_b1[k] * d[i] * data.x_b1[(i, k)]; }
            for &orig_k in &b2_included {
                r -= state.beta_b2[orig_k] * d[i] * s[i] * data.x_b2[(i, orig_k)];
            }
            r
        }).collect();
        for k in 0..p_b1 {
            let idx = priors.b1_range().start + k;
            let c_k: Vec<f64> = (0..n).map(|i| d[i] * data.x_b1[(i, k)]).collect();
            let cur = state.beta_b1[k];
            for i in 0..n { y_run[i] += cur * c_k[i]; }
            let new_val = sample_bounded_coef(
                &c_k, &y_run,
                priors.mean[idx], priors.sd[idx],
                priors.lb[idx], priors.ub[idx], sigma2, rng,
            );
            state.beta_b1[k] = new_val;
            for i in 0..n { y_run[i] -= new_val * c_k[i]; }
        }
        for &orig_k in &b2_included {
            let idx = priors.b2_range().start + orig_k;
            let c_k: Vec<f64> = (0..n).map(|i| d[i] * s[i] * data.x_b2[(i, orig_k)]).collect();
            let cur = state.beta_b2[orig_k];
            for i in 0..n { y_run[i] += cur * c_k[i]; }
            let new_val = sample_bounded_coef(
                &c_k, &y_run,
                priors.mean[idx], priors.sd[idx],
                priors.lb[idx], priors.ub[idx], sigma2, rng,
            );
            state.beta_b2[orig_k] = new_val;
            for i in 0..n { y_run[i] -= new_val * c_k[i]; }
        }
        // Step 3: excluded b2 from prior (Kuo-Mallick latent draw).
        for k in 0..p_b2 {
            if !state.gamma[k] {
                let idx = b2_r.start + k;
                let distr = Normal::new(priors.mean[idx], priors.sd[idx]).unwrap();
                let mut val = distr.sample(rng);
                if priors.lb[idx].is_finite() || priors.ub[idx].is_finite() {
                    for _ in 0..1000 {
                        if val >= priors.lb[idx] && val <= priors.ub[idx] { break; }
                        val = distr.sample(rng);
                    }
                }
                state.beta_b2[k] = val;
            }
        }
        return; // bounded path done
    }
    // ---- Unbounded path: full joint conjugate Gibbs (unchanged) -------------

    let mut prec_prior = DVector::<f64>::zeros(p_lin);
    let mut mu_prior = DVector::<f64>::zeros(p_lin);
    for (k, i) in b0_r.enumerate() {
        prec_prior[k] = 1.0 / (priors.sd[i] * priors.sd[i]);
        mu_prior[k] = priors.mean[i];
    }
    for (k, i) in b1_r.enumerate() {
        prec_prior[p_b0 + k] = 1.0 / (priors.sd[i] * priors.sd[i]);
        mu_prior[p_b0 + k] = priors.mean[i];
    }
    for (j, &orig_k) in b2_included.iter().enumerate() {
        let i = b2_r.start + orig_k;
        prec_prior[p_b0 + p_b1 + j] = 1.0 / (priors.sd[i] * priors.sd[i]);
        mu_prior[p_b0 + p_b1 + j] = priors.mean[i];
    }

    // Posterior precision and mean
    let xt = x_full.transpose();
    let xtx = &xt * &x_full;
    let mut prec_post = xtx / sigma2;
    for k in 0..p_lin {
        prec_post[(k, k)] += prec_prior[k];
    }

    let xty: DVector<f64> = &xt * &y_tilde / sigma2;
    let rhs: DVector<f64> = xty + prec_prior.zip_map(&mu_prior, |p, m| p * m);

    let chol = prec_post.clone().cholesky().expect("Precision matrix not positive definite");
    let mu_post = chol.solve(&rhs);

    let normal = Normal::new(0.0, 1.0f64).unwrap();
    let z = DVector::from_iterator(p_lin, (0..p_lin).map(|_| normal.sample(rng)));
    let l = chol.l();
    let x = l.transpose().solve_upper_triangular(&z)
        .expect("Triangular solve failed");
    let theta = mu_post + x;

    // Unbounded path: always accept the joint draw.
    for k in 0..p_b0 {
        state.beta_b0[k] = theta[k];
    }
    for k in 0..p_b1 {
        state.beta_b1[k] = theta[p_b0 + k];
    }
    for (j, &orig_k) in b2_included.iter().enumerate() {
        state.beta_b2[orig_k] = theta[p_b0 + p_b1 + j];
    }

    // Draw excluded b2 coefficients from their prior (Kuo-Mallick latent draw)
    for k in 0..p_b2 {
        if !state.gamma[k] {
            let idx = b2_r.start + k;
            let d = Normal::new(priors.mean[idx], priors.sd[idx]).unwrap();
            let mut val = d.sample(rng);
            // Respect bounds on the slab
            if priors.lb[idx].is_finite() || priors.ub[idx].is_finite() {
                // Simple rejection sampling from truncated prior
                for _ in 0..1000 {
                    if val >= priors.lb[idx] && val <= priors.ub[idx] {
                        break;
                    }
                    val = d.sample(rng);
                }
            }
            state.beta_b2[k] = val;
        }
    }
}

// Similar for joint sampling with random effects
fn sample_linear_coefs_joint_ss(
    data: &ModelData,
    priors: &Priors,
    _ss: &SpikeSlabConfig,
    state: &mut State,
    rng: &mut StdRng,
) {
    let n = data.n;
    let p_b0 = priors.p_b0;
    let p_b1 = priors.p_b1;
    let p_b2 = priors.p_b2;
    let j_groups = data.n_groups_b0;
    let sigma2 = state.sigma * state.sigma;
    let sigma_u2 = state.sigma_u * state.sigma_u;

    let omega = state.omega_vec(&data.x_om);
    let rho   = state.rho_vec(&data.x_rho);
    let d: Vec<f64> = (0..n).map(|i| data.tau[i] - omega[i]).collect();
    let s: Vec<f64> = (0..n).map(|i| sigmoid(d[i] * rho[i])).collect();

    let b2_included: Vec<usize> = (0..p_b2).filter(|&k| state.gamma[k]).collect();

    // ---- Bounded path: (b0, u_j) conjugate Gibbs + b1/included-b2 coord-wise
    if priors.lin_has_finite_bounds() {
        // Step 1: sample (b0, u_j) jointly given current b1 and included b2.
        let y_b0u: Vec<f64> = (0..n).map(|i| {
            let mut r = data.y[i];
            for k in 0..p_b1 { r -= state.beta_b1[k] * d[i] * data.x_b1[(i, k)]; }
            for &orig_k in &b2_included {
                r -= state.beta_b2[orig_k] * d[i] * s[i] * data.x_b2[(i, orig_k)];
            }
            r
        }).collect();
        let p_b0_u = p_b0 + j_groups;
        let mut x_b0u = DMatrix::<f64>::zeros(n, p_b0_u);
        for i in 0..n {
            for k in 0..p_b0 { x_b0u[(i, k)] = data.x_b0[(i, k)]; }
            let g = data.group_b0[i];
            if g >= 0 { x_b0u[(i, p_b0 + g as usize)] = 1.0; }
        }
        let mut prec_b0u = DVector::<f64>::zeros(p_b0_u);
        let mut mu_b0u   = DVector::<f64>::zeros(p_b0_u);
        for (k, i) in priors.b0_range().enumerate() {
            prec_b0u[k] = 1.0 / (priors.sd[i] * priors.sd[i]);
            mu_b0u[k]   = priors.mean[i];
        }
        let u_prec = 1.0 / sigma_u2;
        for j in 0..j_groups { prec_b0u[p_b0 + j] = u_prec; }
        let y_b0u_dv = DVector::from_vec(y_b0u);
        let xt_b0u = x_b0u.transpose();
        let xtx_b0u = &xt_b0u * &x_b0u;
        let mut pp_b0u = xtx_b0u / sigma2;
        for k in 0..p_b0_u { pp_b0u[(k, k)] += prec_b0u[k]; }
        let xty_b0u: DVector<f64> = &xt_b0u * &y_b0u_dv / sigma2;
        let rhs_b0u = xty_b0u + prec_b0u.zip_map(&mu_b0u, |p, m| p * m);
        let chol_b0u = pp_b0u.cholesky()
            .expect("(b0, u) precision matrix not positive definite");
        let mu_post_b0u = chol_b0u.solve(&rhs_b0u);
        let normal = Normal::new(0.0, 1.0f64).unwrap();
        let z_b0u = DVector::from_iterator(
            p_b0_u, (0..p_b0_u).map(|_| normal.sample(rng))
        );
        let l_b0u = chol_b0u.l();
        let w_b0u = l_b0u.transpose().solve_upper_triangular(&z_b0u)
            .expect("(b0, u) triangular solve failed");
        let theta_b0u = mu_post_b0u + w_b0u;
        let b0_ok = priors.b0_range().enumerate().all(|(k, i)| {
            theta_b0u[k] >= priors.lb[i] && theta_b0u[k] <= priors.ub[i]
        });
        if b0_ok { for k in 0..p_b0 { state.beta_b0[k] = theta_b0u[k]; } }
        for j in 0..j_groups { state.u_b0[j] = theta_b0u[p_b0 + j]; }

        // Step 2: coordinate-wise b1 and included b2.
        let mut y_run: Vec<f64> = (0..n).map(|i| {
            let mut r = data.y[i];
            for k in 0..p_b0 { r -= state.beta_b0[k] * data.x_b0[(i, k)]; }
            let g = data.group_b0[i];
            if g >= 0 { r -= state.u_b0[g as usize]; }
            for k in 0..p_b1 { r -= state.beta_b1[k] * d[i] * data.x_b1[(i, k)]; }
            for &orig_k in &b2_included {
                r -= state.beta_b2[orig_k] * d[i] * s[i] * data.x_b2[(i, orig_k)];
            }
            r
        }).collect();
        for k in 0..p_b1 {
            let idx = priors.b1_range().start + k;
            let c_k: Vec<f64> = (0..n).map(|i| d[i] * data.x_b1[(i, k)]).collect();
            let cur = state.beta_b1[k];
            for i in 0..n { y_run[i] += cur * c_k[i]; }
            let new_val = sample_bounded_coef(
                &c_k, &y_run,
                priors.mean[idx], priors.sd[idx],
                priors.lb[idx], priors.ub[idx], sigma2, rng,
            );
            state.beta_b1[k] = new_val;
            for i in 0..n { y_run[i] -= new_val * c_k[i]; }
        }
        for &orig_k in &b2_included {
            let idx = priors.b2_range().start + orig_k;
            let c_k: Vec<f64> = (0..n).map(|i| d[i] * s[i] * data.x_b2[(i, orig_k)]).collect();
            let cur = state.beta_b2[orig_k];
            for i in 0..n { y_run[i] += cur * c_k[i]; }
            let new_val = sample_bounded_coef(
                &c_k, &y_run,
                priors.mean[idx], priors.sd[idx],
                priors.lb[idx], priors.ub[idx], sigma2, rng,
            );
            state.beta_b2[orig_k] = new_val;
            for i in 0..n { y_run[i] -= new_val * c_k[i]; }
        }
        // Step 3: excluded b2 from prior.
        let b2_r_exc = priors.b2_range();
        for k in 0..p_b2 {
            if !state.gamma[k] {
                let idx = b2_r_exc.start + k;
                let distr = Normal::new(priors.mean[idx], priors.sd[idx]).unwrap();
                let mut val = distr.sample(rng);
                if priors.lb[idx].is_finite() || priors.ub[idx].is_finite() {
                    for _ in 0..1000 {
                        if val >= priors.lb[idx] && val <= priors.ub[idx] { break; }
                        val = distr.sample(rng);
                    }
                }
                state.beta_b2[k] = val;
            }
        }
        return; // bounded path done
    }
    // ---- Unbounded path: two-block NC Gibbs (Option B) ----------------------
    //
    // Same rationale as sample_linear_coefs_joint(); see that function for the
    // full cost/mixing analysis and the Option A description.
    //
    // sample_linear_coefs_ss() handles the spike-and-slab specifics:
    //   - included b2 (gamma_k = 1): sampled jointly with b0 and b1
    //   - excluded b2 (gamma_k = 0): drawn from their prior (Kuo-Mallick)
    sample_random_effects_nc(data, state, rng);
    sample_linear_coefs_ss(data, priors, _ss, state, rng);
}

// ---------------------------------------------------------------------------
// Enforce gamma on omega/rho: zero out excluded coefficients.
// Called after sampling omega/rho so that coefficients whose b2 partner
// has gamma=0 remain at zero.
// ---------------------------------------------------------------------------

fn enforce_gamma_on_om_rho(state: &mut State, ss: &SpikeSlabConfig) {
    for k in 0..state.gamma.len() {
        if !state.gamma[k] {
            if ss.om_map[k] >= 0 {
                state.beta_om[ss.om_map[k] as usize] = 0.0;
            }
            if ss.rho_map[k] >= 0 {
                state.beta_rho[ss.rho_map[k] as usize] = 0.0;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Main chain loop for spike-and-slab model.
//
// Same structure as run_chain but adds:
// - gamma sampling block (Kuo & Mallick indicators)
// - modified linear coefficient sampling that respects gamma
// - gamma enforcement on omega/rho after their update
// - gamma draws in the output
// ---------------------------------------------------------------------------

#[allow(unused_assignments)]
pub fn run_chain_ss(
    data: &ModelData,
    priors: &Priors,
    ss: &SpikeSlabConfig,
    n_iter: usize,
    n_warmup: usize,
    step_om_init: f64,
    step_rho_init: f64,
    target_accept: f64,
    seed: u64,
    verbose: bool,
    chain_id: usize,
    n_chains: usize,
    progress_fn: &dyn Fn(usize, usize, usize, usize, bool),
) -> (DMatrix<f64>, usize) {
    let mut rng = StdRng::seed_from_u64(seed);

    let mut state = init_state(data, priors, &mut rng);
    // Initialise gamma: start all eligible coefficients as included
    for k in 0..priors.p_b2 {
        state.gamma[k] = true;
    }
    // Initialise pi from the first eligible coefficient's pi (or 0.5)
    state.pi = ss.pi_init.iter().zip(ss.spike_mask.iter())
        .find(|(_, &m)| m).map(|(&p, _)| p).unwrap_or(0.5);

    let learn_pi = ss.beta_a > 0.0;
    let n_post = n_iter - n_warmup;
    let n_params = state.n_params_ss(learn_pi);

    let mut draws = DMatrix::<f64>::zeros(n_post, n_params);

    // --- Adaptation state (same as base model) ---
    let use_hmc_om  = priors.p_om  >= 2;
    let use_hmc_rho = priors.p_rho >= 2;

    let mut hmc_om  = if use_hmc_om  {
        Some(HmcAdapt::new(priors.p_om,  step_om_init,  target_accept, 5, 15))
    } else { None };
    let mut hmc_rho = if use_hmc_rho {
        Some(HmcAdapt::new(priors.p_rho, step_rho_init, target_accept, 5, 15))
    } else { None };

    let mut scalar_om  = if priors.p_om  == 1 { Some(ScalarAdapt::new(step_om_init))  } else { None };
    let mut scalar_rho = if priors.p_rho == 1 { Some(ScalarAdapt::new(step_rho_init)) } else { None };

    let tune_window = 100usize;
    let mass_refresh_iter_om  = (n_warmup as f64 * 0.60) as usize;
    let mass_refresh_iter_rho = (n_warmup as f64 * 0.60) as usize;

    let mut omega_cur: DVector<f64> = DVector::<f64>::zeros(0);
    let mut rho_cur:   DVector<f64> = DVector::<f64>::zeros(0);
    let mut cache: LinearCache = LinearCache {
        b0_fixed:   DVector::<f64>::zeros(0),
        b1_vals:    DVector::<f64>::zeros(0),
        b2_vals:    DVector::<f64>::zeros(0),
        re_contrib: DVector::<f64>::zeros(0),
    };

    let report_every = (n_iter / 10).max(1);

    // --- Find reasonable initial epsilon ---
    {
        if data.n_groups_b0 > 0 { sample_random_effects_nc(data, &mut state, &mut rng); }
        sample_linear_coefs(data, priors, &mut state, &mut rng);
        let init_cache = LinearCache::build(&state, data);
        let init_omega = &data.x_om  * &state.beta_om;
        let init_rho   = &data.x_rho * &state.beta_rho;

        if let Some(ref mut h) = hmc_om {
            let p = priors.p_om;
            let r_idx = priors.om_range();
            let pr_mean: Vec<f64> = priors.mean[r_idx.clone()].to_vec();
            let pr_sd:   Vec<f64> = priors.sd[r_idx.clone()].to_vec();
            let pr_lb:   Vec<f64> = priors.lb[r_idx.clone()].to_vec();
            let pr_ub:   Vec<f64> = priors.ub[r_idx.clone()].to_vec();
            let q0 = state.beta_om.clone();
            let sigma = state.sigma;
            let inv_mass = h.inv_mass.clone();
            h.find_reasonable_epsilon(|eps| {
                let normal_dist = Normal::new(0.0, 1.0f64).unwrap();
                let mut mom = DVector::<f64>::from_iterator(
                    p, (0..p).map(|k| normal_dist.sample(&mut rng) / inv_mass[k].sqrt()));
                let kinetic0: f64 = (0..p).map(|k| mom[k]*mom[k]*inv_mass[k]).sum::<f64>() * 0.5;
                let ll0 = init_cache.log_likelihood(data, &init_omega, &init_rho, sigma);
                let lp0 = log_truncated_normal_prior(q0.as_slice(), &pr_mean, &pr_sd, &pr_lb, &pr_ub);
                let h0_val = -ll0 - lp0 + kinetic0;
                let grad_ll = init_cache.grad_beta_om(data, &init_omega, &init_rho, sigma, p);
                let grad_pr = grad_truncated_normal_prior(q0.as_slice(), &pr_mean, &pr_sd, &pr_lb, &pr_ub);
                let grad_lp: DVector<f64> = &grad_ll + &grad_pr;
                let mut q = q0.clone();
                for k in 0..p { mom[k] += 0.5 * eps * grad_lp[k]; }
                for k in 0..p {
                    q[k] += eps * mom[k] * inv_mass[k];
                    for _b in 0..MAX_REFLECTIONS {
                        if q[k] < pr_lb[k] { q[k] = 2.0*pr_lb[k]-q[k]; mom[k]=-mom[k]; }
                        else if q[k] > pr_ub[k] { q[k] = 2.0*pr_ub[k]-q[k]; mom[k]=-mom[k]; }
                        else { break; }
                    }
                }
                let omega_p = &data.x_om * &q;
                let gl2 = init_cache.grad_beta_om(data, &omega_p, &init_rho, sigma, p);
                let gp2 = grad_truncated_normal_prior(q.as_slice(), &pr_mean, &pr_sd, &pr_lb, &pr_ub);
                let grad2: DVector<f64> = &gl2 + &gp2;
                for k in 0..p { mom[k] += 0.5 * eps * grad2[k]; }
                let ll1 = init_cache.log_likelihood(data, &omega_p, &init_rho, sigma);
                let lp1 = log_truncated_normal_prior(q.as_slice(), &pr_mean, &pr_sd, &pr_lb, &pr_ub);
                let kin1: f64 = (0..p).map(|k| mom[k]*mom[k]*inv_mass[k]).sum::<f64>() * 0.5;
                let h1_val = -ll1 - lp1 + kin1;
                h0_val - h1_val
            });
        }

        if let Some(ref mut h) = hmc_rho {
            let p = priors.p_rho;
            let r_idx = priors.rho_range();
            let pr_mean: Vec<f64> = priors.mean[r_idx.clone()].to_vec();
            let pr_sd:   Vec<f64> = priors.sd[r_idx.clone()].to_vec();
            let pr_lb:   Vec<f64> = priors.lb[r_idx.clone()].to_vec();
            let pr_ub:   Vec<f64> = priors.ub[r_idx.clone()].to_vec();
            let q0 = state.beta_rho.clone();
            let sigma = state.sigma;
            let inv_mass_rho = h.inv_mass.clone();
            h.find_reasonable_epsilon(|eps| {
                let normal_dist = Normal::new(0.0, 1.0f64).unwrap();
                let mut mom = DVector::<f64>::from_iterator(
                    p, (0..p).map(|k| normal_dist.sample(&mut rng) / inv_mass_rho[k].sqrt()));
                let kinetic0: f64 = (0..p).map(|k| mom[k]*mom[k]*inv_mass_rho[k]).sum::<f64>() * 0.5;
                let ll0 = init_cache.log_likelihood(data, &init_omega, &init_rho, sigma);
                let lp0 = log_truncated_normal_prior(q0.as_slice(), &pr_mean, &pr_sd, &pr_lb, &pr_ub);
                let h0_val = -ll0 - lp0 + kinetic0;
                let grad_ll = init_cache.grad_beta_rho(data, &init_omega, &init_rho, sigma, p);
                let grad_pr = grad_truncated_normal_prior(q0.as_slice(), &pr_mean, &pr_sd, &pr_lb, &pr_ub);
                let grad_lp: DVector<f64> = &grad_ll + &grad_pr;
                let mut q = q0.clone();
                for k in 0..p { mom[k] += 0.5 * eps * grad_lp[k]; }
                for k in 0..p {
                    q[k] += eps * mom[k] * inv_mass_rho[k];
                    for _b in 0..MAX_REFLECTIONS {
                        if q[k] < pr_lb[k] { q[k] = 2.0*pr_lb[k]-q[k]; mom[k]=-mom[k]; }
                        else if q[k] > pr_ub[k] { q[k] = 2.0*pr_ub[k]-q[k]; mom[k]=-mom[k]; }
                        else { break; }
                    }
                }
                let rho_p = &data.x_rho * &q;
                let gl2 = init_cache.grad_beta_rho(data, &init_omega, &rho_p, sigma, p);
                let gp2 = grad_truncated_normal_prior(q.as_slice(), &pr_mean, &pr_sd, &pr_lb, &pr_ub);
                let grad2: DVector<f64> = &gl2 + &gp2;
                for k in 0..p { mom[k] += 0.5 * eps * grad2[k]; }
                let ll1 = init_cache.log_likelihood(data, &init_omega, &rho_p, sigma);
                let lp1 = log_truncated_normal_prior(q.as_slice(), &pr_mean, &pr_sd, &pr_lb, &pr_ub);
                let kin1: f64 = (0..p).map(|k| mom[k]*mom[k]*inv_mass_rho[k]).sum::<f64>() * 0.5;
                let h1_val = -ll1 - lp1 + kin1;
                h0_val - h1_val
            });
        }
    }

    // === Main sampling loop ===
    for iter in 0..n_iter {
        if verbose && iter % report_every == 0 {
            progress_fn(chain_id, n_chains, iter, n_iter, iter < n_warmup);
        }

        // Block 1: Linear coefficients (respecting gamma)
        if data.n_groups_b0 > 0 {
            sample_linear_coefs_joint_ss(data, priors, ss, &mut state, &mut rng);
        } else {
            sample_linear_coefs_ss(data, priors, ss, &mut state, &mut rng);
        }

        // Enforce gamma zeroing on b2 coefficients for the cache
        for k in 0..priors.p_b2 {
            if !state.gamma[k] {
                state.beta_b2[k] = 0.0;
            }
        }
        enforce_gamma_on_om_rho(&mut state, ss);

        // Rebuild cache and linear predictors
        cache     = LinearCache::build(&state, data);
        omega_cur = &data.x_om  * &state.beta_om;
        rho_cur   = &data.x_rho * &state.beta_rho;

        // Block 2a: Update beta_om
        if use_hmc_om {
            hmc_step_om(
                data, priors, &mut state,
                &cache, &mut omega_cur, &rho_cur,
                hmc_om.as_mut().unwrap(),
                &mut rng,
            );
        } else if priors.p_om == 1 {
            scalar_mh_step_om(
                data, priors, &mut state,
                &cache, &mut omega_cur, &rho_cur,
                scalar_om.as_mut().unwrap(),
                &mut rng,
            );
        }

        // Block 2b: Update beta_rho
        if use_hmc_rho {
            hmc_step_rho(
                data, priors, &mut state,
                &cache, &omega_cur, &mut rho_cur,
                hmc_rho.as_mut().unwrap(),
                &mut rng,
            );
        } else if priors.p_rho == 1 {
            scalar_mh_step_rho(
                data, priors, &mut state,
                &cache, &omega_cur, &mut rho_cur,
                scalar_rho.as_mut().unwrap(),
                &mut rng,
            );
        }

        // Enforce gamma on omega/rho after their update
        enforce_gamma_on_om_rho(&mut state, ss);

        // Block 3: Sample gamma indicators
        // (Uses the Kuo-Mallick approach after all continuous parameters updated)
        sample_gamma(data, priors, ss, &mut state, &mut rng);

        // Block 3b: Sample pi | gamma (Beta-Bernoulli conjugate)
        if learn_pi {
            sample_pi(ss, &mut state, &mut rng);
        }

        // Block 4a: Sample sigma
        sample_sigma(data, priors, &mut state, &mut rng);

        // Block 4b: Sample sigma_u
        if data.n_groups_b0 > 0 {
            sample_sigma_u(priors, &mut state, &mut rng);
        }

        // --- Adaptation during warmup ---
        if iter < n_warmup {
            if (iter + 1) % tune_window == 0 {
                if let Some(ref mut s) = scalar_om  { s.tune(0.234); }
                if let Some(ref mut s) = scalar_rho { s.tune(0.234); }
            }

            if let Some(ref mut h) = hmc_om  { h.observe(&state.beta_om);  }
            if let Some(ref mut h) = hmc_rho { h.observe(&state.beta_rho); }

            if iter == mass_refresh_iter_om {
                if let Some(ref mut h) = hmc_om {
                    h.refresh_mass_matrix();
                    h.da_count = 0;
                    h.h_bar = 0.0;
                    h.mu = (10.0 * h.epsilon).ln();
                    h.log_eps_bar = h.epsilon.ln();
                }
            }
            if iter == mass_refresh_iter_rho {
                if let Some(ref mut h) = hmc_rho {
                    h.refresh_mass_matrix();
                    h.da_count = 0;
                    h.h_bar = 0.0;
                    h.mu = (10.0 * h.epsilon).ln();
                    h.log_eps_bar = h.epsilon.ln();
                }
            }
        }

        // Freeze adaptation
        if iter + 1 == n_warmup {
            if let Some(ref mut h) = hmc_om  { h.freeze(); }
            if let Some(ref mut h) = hmc_rho { h.freeze(); }
        }

        // Store post-warmup draws (including gamma)
        if iter >= n_warmup {
            let row = iter - n_warmup;
            let draw = state.to_vec_ss(learn_pi);
            for (col, &val) in draw.iter().enumerate() {
                draws[(row, col)] = val;
            }
        }
    }

    if verbose {
        progress_fn(chain_id, n_chains, n_iter, n_iter, false);
    }

    let n_div = hmc_om.as_ref().map_or(0, |h| h.n_divergent)
        + hmc_rho.as_ref().map_or(0, |h| h.n_divergent);
    (draws, n_div)
}
