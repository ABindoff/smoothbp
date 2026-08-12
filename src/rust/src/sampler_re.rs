use nalgebra::{DMatrix, DVector};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand::Rng;
use rand_distr::{Normal, Gamma, Distribution};

use crate::model::{ModelData, Priors, State, SpikeSlabConfig, log_truncated_normal_prior,
                   log_normal_interval_mass, rtruncnorm, sigmoid, b0_translation_step};

// ---------------------------------------------------------------------------
// LinearCache: precomputed parts of the mean function for segment HMC steps.
// mu_i = base_i + delta_i * d_i * s_i
// where base_i includes everything else.
// ---------------------------------------------------------------------------

struct LinearCache {
    b0_fixed: DVector<f64>,
    re_contrib: DVector<f64>,
    b1_vals: DVector<f64>,
    delta_vals: Vec<DVector<f64>>,
}

impl LinearCache {
    fn build(state: &State, data: &ModelData) -> Self {
        let b0_fixed = &data.x_b0 * &state.beta_b0;
        let mut re_contrib = DVector::<f64>::zeros(data.n);
        if data.n_groups_b0 > 0 {
            for i in 0..data.n {
                let g = data.group_b0[i];
                if g >= 0 { re_contrib[i] = state.u_b0[g as usize]; }
            }
        }

        let mut b1_eff = state.beta_b1.clone();
        for j in 0..b1_eff.len() {
            if !state.gamma_b1[j] { b1_eff[j] = 0.0; }
        }
        let b1_vals = &data.x_b1 * &b1_eff;

        let mut delta_vals = Vec::with_capacity(data.n_breakpoints);
        for k in 0..data.n_breakpoints {
            let mut bd_eff = state.beta_deltas[k].clone();
            for j in 0..bd_eff.len() {
                if !state.gamma_deltas[k][j] { bd_eff[j] = 0.0; }
            }
            delta_vals.push(&data.x_deltas[k] * &bd_eff);
        }

        LinearCache { b0_fixed, re_contrib, b1_vals, delta_vals }
    }

    /// Compute mu_i excluding segment k's dynamic part and the b1*d part if requested.
    fn mu_without_segment(&self, data: &ModelData, state: &State, k: usize) -> DVector<f64> {
        let n = data.n;
        let mut mu = self.b0_fixed.clone() + &self.re_contrib;

        // Add all segments except k
        for i in 0..data.n_breakpoints {
            if i == k { continue; }
            let om = state.omega_vec(i, &data.x_om[i]);
            let rho = state.rho_vec(i, &data.x_rho[i]);
            for j in 0..n {
                let di = data.tau[j] - om[j];
                let si = sigmoid(di * rho[j]);
                mu[j] += self.delta_vals[i][j] * di * si;
            }
        }
        
        // Add b1 part (Special handling if k == 0 for the centering)
        if data.n_breakpoints > 0 {
            if k != 0 {
                let om1 = state.omega_vec(0, &data.x_om[0]);
                for j in 0..n {
                    mu[j] += self.b1_vals[j] * (data.tau[j] - om1[j]);
                }
            }
        } else {
            for j in 0..n {
                mu[j] += self.b1_vals[j] * data.tau[j];
            }
        }
        mu
    }
}

// ---------------------------------------------------------------------------
// HmcAdapt: dual-averaging and mass matrix estimation
// ---------------------------------------------------------------------------

const EPSILON_FLOOR: f64 = 1e-6;
const DIVERGENCE_THRESHOLD: f64 = 1000.0;
const MAX_REFLECTIONS: usize = 20;
const NUTS_MAX_DEPTH: usize = 10;

struct HmcAdapt {
    p: usize,
    epsilon: f64,
    target_accept: f64,
    mu: f64,
    log_eps_bar: f64,
    h_bar: f64,
    gamma: f64,
    t0: f64,
    kappa: f64,
    da_count: usize,
    inv_mass: Vec<f64>,
    welford_n: usize,
    welford_mean: DVector<f64>,
    welford_m2: DVector<f64>,
    adapting: bool,
    n_divergent: usize,
}

impl HmcAdapt {
    fn new(p: usize, init_epsilon: f64, target_accept: f64) -> Self {
        HmcAdapt {
            p, epsilon: init_epsilon, target_accept,
            mu: (10.0 * init_epsilon).ln(), log_eps_bar: 0.0, h_bar: 0.0,
            gamma: 0.05, t0: 10.0, kappa: 0.75, da_count: 0,
            inv_mass: vec![1.0; p], welford_n: 0,
            welford_mean: DVector::<f64>::zeros(p), welford_m2: DVector::<f64>::zeros(p),
            adapting: true, n_divergent: 0,
        }
    }

    fn update_epsilon(&mut self, accept_prob: f64) {
        if !self.adapting { return; }
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

    fn record_energy_error(&mut self, delta_h: f64) {
        if !self.adapting && (delta_h.abs() > DIVERGENCE_THRESHOLD || delta_h.is_nan()) {
            self.n_divergent += 1;
        }
    }

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

    fn refresh_mass_matrix(&mut self) {
        if self.welford_n < 20 { return; }
        let n = self.welford_n as f64;
        for k in 0..self.p {
            let var_k = self.welford_m2[k] / (n - 1.0);
            self.inv_mass[k] = var_k.max(1e-8);
        }
    }

    fn freeze(&mut self) {
        self.epsilon = self.log_eps_bar.exp().max(EPSILON_FLOOR);
        self.adapting = false;
    }

}

// ---------------------------------------------------------------------------
// Gibbs Samplers
// ---------------------------------------------------------------------------

fn sample_sigma(data: &ModelData, priors: &Priors, state: &mut State, rng: &mut StdRng) {
    let mu = state.means(data);
    let r = &data.y - &mu;
    let ss = r.dot(&r);
    let n = data.n as f64;
    let shape = priors.sigma_shape + n * 0.5;
    let scale = priors.sigma_scale + ss * 0.5;
    let gamma_dist = Gamma::new(shape, 1.0 / scale).unwrap();
    state.sigma = 1.0 / gamma_dist.sample(rng).sqrt();
}

fn sample_sigma_u(priors: &Priors, state: &mut State, rng: &mut StdRng) {
    let ss = state.u_b0.dot(&state.u_b0);
    let n = state.u_b0.len() as f64;
    let shape = priors.sigma_u_shape + n * 0.5;
    let scale = priors.sigma_u_scale + ss * 0.5;
    let gamma_dist = Gamma::new(shape, 1.0 / scale).unwrap();
    state.sigma_u = 1.0 / gamma_dist.sample(rng).sqrt();
}

fn sample_sigma_re_om(data: &ModelData, priors: &Priors, state: &mut State, rng: &mut StdRng) {
    for k in 0..data.n_breakpoints {
        let mut ss = 0.0;
        let mut count = 0.0;
        for j in 0..state.beta_om[k].len() {
            if data.re_mask_om[k][j] {
                let val = state.beta_om[k][j];
                ss += val * val;
                count += 1.0;
            }
        }
        if count > 0.0 {
            let shape = priors.sigma_re_om_shape + count * 0.5;
            let scale = priors.sigma_re_om_scale + ss * 0.5;
            let gamma_dist = Gamma::new(shape, 1.0 / scale).unwrap();
            state.sigma_re_om[k] = 1.0 / gamma_dist.sample(rng).sqrt();
        }
    }
}

fn sample_sigma_re_b1(data: &ModelData, priors: &Priors, state: &mut State, rng: &mut StdRng) {
    let mut ss = 0.0;
    let mut count = 0.0;
    for j in 0..state.beta_b1.len() {
        if data.re_mask_b1[j] {
            let val = state.beta_b1[j];
            ss += val * val;
            count += 1.0;
        }
    }
    if count > 0.0 {
        let shape = priors.sigma_re_b1_shape + count * 0.5;
        let scale = priors.sigma_re_b1_scale + ss * 0.5;
        let gamma_dist = Gamma::new(shape, 1.0 / scale).unwrap();
        state.sigma_re_b1 = 1.0 / gamma_dist.sample(rng).sqrt();
    }
}

fn sample_sigma_re_deltas(data: &ModelData, priors: &Priors, state: &mut State, rng: &mut StdRng) {
    for k in 0..data.n_breakpoints {
        let mut ss = 0.0;
        let mut count = 0.0;
        for j in 0..state.beta_deltas[k].len() {
            if data.re_mask_deltas[k][j] {
                let val = state.beta_deltas[k][j];
                ss += val * val;
                count += 1.0;
            }
        }
        if count > 0.0 {
            let shape = priors.sigma_re_deltas_shape + count * 0.5;
            let scale = priors.sigma_re_deltas_scale + ss * 0.5;
            let gamma_dist = Gamma::new(shape, 1.0 / scale).unwrap();
            state.sigma_re_deltas[k] = 1.0 / gamma_dist.sample(rng).sqrt();
        }
    }
}

fn sample_pi(ss: &SpikeSlabConfig, state: &mut State, rng: &mut StdRng) {
    let mut n1 = 0.0;
    let mut n0 = 0.0;
    for &g in &state.gamma_b1 { if g { n1 += 1.0; } else { n0 += 1.0; } }
    for g_vec in &state.gamma_deltas {
        for &g in g_vec { if g { n1 += 1.0; } else { n0 += 1.0; } }
    }
    let a = ss.beta_a + n1;
    let b = ss.beta_b + n0;
    let gamma_a = Gamma::new(a, 1.0).unwrap();
    let gamma_b = Gamma::new(b, 1.0).unwrap();
    let x = gamma_a.sample(rng);
    let y = gamma_b.sample(rng);
    state.pi = x / (x + y);
}

/// Collapsed (Rao-Blackwellised) spike-and-slab indicator update.
///
/// See the twin in `sampler.rs` for the derivation.  The only difference here is
/// the slab: a random-effect column's prior is N(0, sigma_re^2) with sigma_re
/// learned, not the fixed-effect prior, matching how `sample_linear_coefs` and
/// `joint_subject_nuts` treat those columns.
fn sample_gamma(data: &ModelData, priors: &Priors, ss: &SpikeSlabConfig, state: &mut State, rng: &mut StdRng) {
    let sigma2 = state.sigma * state.sigma;
    let pi = state.pi;
    let logit_pi = (pi / (1.0 - pi)).ln();

    let collapsed = |y: &DVector<f64>,
                     mu_without: &DVector<f64>,
                     v: &DVector<f64>,
                     tau2: f64, m0: f64, lb: f64, ub: f64,
                     rng: &mut StdRng| -> (bool, f64) {
        let r = y - mu_without;
        let a = v.dot(v);
        let b = v.dot(&r);
        let p = a / sigma2 + 1.0 / tau2;
        let q = b / sigma2 + m0 / tau2;
        let m = q / p;
        let post_sd = (1.0 / p).sqrt();
        let slab_sd = tau2.sqrt();

        let log_bf = -0.5 * (tau2 * p).ln() - 0.5 * m0 * m0 / tau2 + 0.5 * q * q / p
            + log_normal_interval_mass(m, post_sd, lb, ub)
            - log_normal_interval_mass(m0, slab_sd, lb, ub);

        let logit = logit_pi + log_bf;
        let prob = if logit.is_nan() { 0.5 } else { sigmoid(logit) };
        let g = rng.gen_bool(prob.clamp(0.0, 1.0));
        let beta = if g {
            rtruncnorm(m, post_sd, lb, ub, rng)
        } else {
            rtruncnorm(m0, slab_sd, lb, ub, rng)
        };
        (g, beta)
    };

    let mut current_mu = state.means(data);

    // b1 columns eligible for spike-and-slab
    for j in 0..state.gamma_b1.len() {
        if !ss.b1_spike_mask[j] { continue; }
        let is_re = data.re_mask_b1.get(j).copied().unwrap_or(false);

        let x_col = &data.x_b1.column(j);
        let mut v = x_col.clone_owned();
        if data.n_breakpoints > 0 {
            let om1 = state.omega_vec(0, &data.x_om[0]);
            for i in 0..data.n { v[i] *= data.tau[i] - om1[i]; }
        } else {
            for i in 0..data.n { v[i] *= data.tau[i]; }
        }

        let (tau2, m0) = if is_re {
            (state.sigma_re_b1 * state.sigma_re_b1, 0.0)
        } else {
            (priors.b1_sd[j] * priors.b1_sd[j], priors.b1_mean[j])
        };

        let mu_without = if state.gamma_b1[j] {
            &current_mu - &v * state.beta_b1[j]
        } else {
            current_mu.clone()
        };
        let (g, beta) = collapsed(&data.y, &mu_without, &v, tau2, m0,
                                  priors.b1_lb[j], priors.b1_ub[j], rng);
        state.gamma_b1[j] = g;
        state.beta_b1[j] = beta;
        current_mu = if g { &mu_without + &v * beta } else { mu_without };
    }

    // delta columns eligible for spike-and-slab
    for k in 0..data.n_breakpoints {
        let om = state.omega_vec(k, &data.x_om[k]);
        let rho = state.rho_vec(k, &data.x_rho[k]);
        for j in 0..state.gamma_deltas[k].len() {
            if !ss.delta_spike_mask[k][j] { continue; }
            let is_re = data.re_mask_deltas.get(k)
                .and_then(|m| m.get(j)).copied().unwrap_or(false);

            let x_col = &data.x_deltas[k].column(j);
            let mut v = x_col.clone_owned();
            for i in 0..data.n {
                let di = data.tau[i] - om[i];
                let si = sigmoid(di * rho[i]);
                v[i] *= di * si;
            }

            let (tau2, m0) = if is_re {
                (state.sigma_re_deltas[k] * state.sigma_re_deltas[k], 0.0)
            } else {
                (priors.delta_sd[k][j] * priors.delta_sd[k][j], priors.delta_mean[k][j])
            };

            let mu_without = if state.gamma_deltas[k][j] {
                &current_mu - &v * state.beta_deltas[k][j]
            } else {
                current_mu.clone()
            };
            let (g, beta) = collapsed(&data.y, &mu_without, &v, tau2, m0,
                                      priors.delta_lb[k][j], priors.delta_ub[k][j], rng);
            state.gamma_deltas[k][j] = g;
            state.beta_deltas[k][j] = beta;
            current_mu = if g { &mu_without + &v * beta } else { mu_without };
        }
    }
}

// ---------------------------------------------------------------------------
// Linear Coefficient Block (beta_b0, beta_b1, beta_deltas)
// Joint sampler for regression coefficients.
// ---------------------------------------------------------------------------

fn sample_linear_coefs(data: &ModelData, priors: &Priors, state: &mut State, rng: &mut StdRng) {
    let n = data.n;
    let sigma2 = state.sigma * state.sigma;
    
    // Construct joint design matrix X for ALL linear terms
    let p_b0 = data.x_b0.ncols();
    let p_b1 = data.x_b1.ncols();
    let mut p_total = p_b0 + p_b1;
    for k in 0..data.n_breakpoints { p_total += data.x_deltas[k].ncols(); }
    
    let mut x_full = DMatrix::<f64>::zeros(n, p_total);
    let mut prec_prior = DVector::<f64>::zeros(p_total);
    let mut mu_prior = DVector::<f64>::zeros(p_total);

    // b0
    x_full.view_mut((0, 0), (n, p_b0)).copy_from(&data.x_b0);
    for j in 0..p_b0 {
        prec_prior[j] = 1.0 / (priors.b0_sd[j] * priors.b0_sd[j]);
        mu_prior[j] = priors.b0_mean[j];
    }

    // b1
    let mut b1_design = data.x_b1.clone();
    if data.n_breakpoints > 0 {
        let om1 = state.omega_vec(0, &data.x_om[0]);
        for i in 0..n {
            let mut row = b1_design.row_mut(i);
            for j in 0..p_b1 { row[j] *= data.tau[i] - om1[i]; }
        }
    } else {
        for i in 0..n {
            let mut row = b1_design.row_mut(i);
            for j in 0..p_b1 { row[j] *= data.tau[i]; }
        }
    }
    // Apply gamma_b1 (Kuo-Mallick) AND zero out RE columns (handled by joint_subject_nuts)
    for j in 0..p_b1 {
        if !state.gamma_b1[j] || data.re_mask_b1[j] {
            let mut col = b1_design.column_mut(j);
            col.fill(0.0);
        }
    }
    x_full.view_mut((0, p_b0), (n, p_b1)).copy_from(&b1_design);
    for j in 0..p_b1 {
        // RE columns: use adaptive prior (sigma_re_b1); fixed columns: static prior
        let sd = if data.re_mask_b1[j] { state.sigma_re_b1 } else { priors.b1_sd[j] };
        prec_prior[p_b0 + j] = 1.0 / (sd * sd);
        mu_prior[p_b0 + j] = priors.b1_mean[j];
    }

    // deltas
    let mut offset = p_b0 + p_b1;
    for k in 0..data.n_breakpoints {
        let pk = data.x_deltas[k].ncols();
        let mut d_design = data.x_deltas[k].clone();
        let om = state.omega_vec(k, &data.x_om[k]);
        let rho = state.rho_vec(k, &data.x_rho[k]);
        for i in 0..n {
            let di = data.tau[i] - om[i];
            let si = sigmoid(di * rho[i]);
            let mut row = d_design.row_mut(i);
            for j in 0..pk { row[j] *= di * si; }
        }
        // Apply gamma_deltas AND zero RE columns (handled by joint_subject_nuts)
        for j in 0..pk {
            if !state.gamma_deltas[k][j] || data.re_mask_deltas[k][j] {
                let mut col = d_design.column_mut(j);
                col.fill(0.0);
            }
        }
        x_full.view_mut((0, offset), (n, pk)).copy_from(&d_design);
        for j in 0..pk {
            let sd = if data.re_mask_deltas[k][j] { state.sigma_re_deltas[k] }
                     else { priors.delta_sd[k][j] };
            prec_prior[offset + j] = 1.0 / (sd * sd);
            mu_prior[offset + j] = priors.delta_mean[k][j];
        }
        offset += pk;
    }

    // Sufficient statistics: subtract all random-effect contributions so the
    // joint precision sampler only updates fixed-effect coefficients.
    let mut y_tilde = data.y.clone();

    // b0 random intercepts
    if data.n_groups_b0 > 0 {
        for i in 0..n {
            let g = data.group_b0[i];
            if g >= 0 { y_tilde[i] -= state.u_b0[g as usize]; }
        }
    }

    // b1 RE columns — subtract current RE contribution to mu
    {
        let om1 = if data.n_breakpoints > 0 { state.omega_vec(0, &data.x_om[0]) }
                  else { DVector::zeros(n) };
        for j in 0..p_b1 {
            if data.re_mask_b1[j] && state.gamma_b1[j] {
                for i in 0..n {
                    let scale = if data.n_breakpoints > 0 { data.tau[i] - om1[i] }
                                else { data.tau[i] };
                    y_tilde[i] -= state.beta_b1[j] * data.x_b1[(i, j)] * scale;
                }
            }
        }
    }

    // delta RE columns — subtract current RE contribution to mu
    for k in 0..data.n_breakpoints {
        let pk = data.x_deltas[k].ncols();
        let om  = state.omega_vec(k, &data.x_om[k]);
        let rho = state.rho_vec(k, &data.x_rho[k]);
        for j in 0..pk {
            if data.re_mask_deltas[k][j] && state.gamma_deltas[k][j] {
                for i in 0..n {
                    let di = data.tau[i] - om[i];
                    let si = sigmoid(di * rho[i]);
                    y_tilde[i] -= state.beta_deltas[k][j] * data.x_deltas[k][(i, j)] * di * si;
                }
            }
        }
    }

    let xt = x_full.transpose();
    let mut precision = &xt * &x_full / sigma2;
    for j in 0..p_total { precision[(j, j)] += prec_prior[j]; }

    let cholesky = precision.cholesky().expect("Linear precision matrix not positive definite");
    let xty = &xt * &y_tilde / sigma2;
    let rhs = xty + prec_prior.component_mul(&mu_prior);
    let mean = cholesky.solve(&rhs);

    let mut z = DVector::<f64>::zeros(p_total);
    let normal = Normal::new(0.0, 1.0).unwrap();
    for j in 0..p_total { z[j] = normal.sample(rng); }
    
    // To sample from N(mean, P^-1) where P = L*L^T:
    // x = mean + (L^T)^-1 * z  =>  L^T * (x - mean) = z
    let y = cholesky.l().transpose().solve_upper_triangular(&z).expect("Failed to solve upper triangular system");
    let theta = mean + y;

    // Export back to state: copy only fixed-effect columns for b1 and deltas,
    // preserving the random effects updated by joint_subject_nuts.
    state.beta_b0.copy_from(&theta.rows(0, p_b0));
    for j in 0..p_b1 {
        if !data.re_mask_b1[j] {
            state.beta_b1[j] = theta[p_b0 + j];
        }
    }
    offset = p_b0 + p_b1;
    for k in 0..data.n_breakpoints {
        let pk = data.x_deltas[k].ncols();
        for j in 0..pk {
            if !data.re_mask_deltas[k][j] {
                state.beta_deltas[k][j] = theta[offset + j];
            }
        }
        offset += pk;
    }
}

fn sample_random_effects(data: &ModelData, _priors: &Priors, state: &mut State, rng: &mut StdRng) {
    let sigma2 = state.sigma * state.sigma;
    let sigma_u2 = state.sigma_u * state.sigma_u;
    let n_groups = data.n_groups_b0;

    // Reconstruct mu without random effects
    let mut state_no_re = state.clone();
    state_no_re.u_b0.fill(0.0);
    let mu_fixed = state_no_re.means(data);
    let resid = &data.y - &mu_fixed;

    let mut sum_r = vec![0.0f64; n_groups];
    let mut count = vec![0.0f64; n_groups];
    for i in 0..data.n {
        let g = data.group_b0[i];
        if g >= 0 {
            sum_r[g as usize] += resid[i];
            count[g as usize] += 1.0;
        }
    }

    let normal = Normal::new(0.0, 1.0).unwrap();
    for j in 0..n_groups {
        let prec = count[j] / sigma2 + 1.0 / sigma_u2;
        let post_sd = (1.0 / prec).sqrt();
        let post_mean = (sum_r[j] / sigma2) / prec;
        state.u_b0[j] = post_mean + post_sd * normal.sample(rng);
    }
}

// ---------------------------------------------------------------------------
// Collapsed (Rao-Blackwellised) change-point update
//
// See the twin in `sampler.rs` for why the collapse runs in this direction:
// omega cannot be integrated out of the delta update (it enters through
// d = t - omega and s(d; rho)), but the conditionally Gaussian linear block can
// be integrated out of the omega update, which breaks the same dependence.
//
// This version collapses only the FIXED linear columns, which are the ones
// `sample_linear_coefs` owns.  The per-subject random-effect coefficients stay
// conditioned at their current values and are folded into the response.  They
// are themselves functions of omega, so unlike the non-RE case the response is
// omega-dependent and contributes its own gradient term.
// ---------------------------------------------------------------------------

/// As `linear_block_unbounded` in `sampler.rs`: the closed form needs an
/// untruncated Gaussian prior on every collapsed coefficient.
fn linear_block_unbounded(data: &ModelData, priors: &Priors) -> bool {
    let free = |lo: f64, hi: f64| lo == f64::NEG_INFINITY && hi == f64::INFINITY;
    if !priors.b0_lb.iter().zip(&priors.b0_ub).all(|(&l, &u)| free(l, u)) { return false; }
    if !priors.b1_lb.iter().zip(&priors.b1_ub).all(|(&l, &u)| free(l, u)) { return false; }
    for k in 0..data.n_breakpoints {
        if !priors.delta_lb[k].iter().zip(&priors.delta_ub[k]).all(|(&l, &u)| free(l, u)) {
            return false;
        }
    }
    true
}

/// Negative log marginal likelihood with the fixed linear block integrated out,
/// plus its derivative with respect to each observation's omega_k.
///
/// With y~ the response after removing the random-effect contributions:
///
///   P = X'X/s2 + V^-1     Q = X'y~/s2 + V^-1 m     M = P^-1 Q
///   U = 0.5 y~'y~/s2 + 0.5 log|P| - 0.5 Q'M        (+ terms free of omega)
///
/// and, with xi_i the row observation i contributes to dX/d(omega_i) and
/// r = y~ - X M,
///
///   dU/d(omega_i) = [ X_i' P^-1 xi_i - (xi_i' M) r_i + r_i * dy~_i/d(omega_i) ] / s2
///
/// The last term is absent in the non-RE sampler, where y~ does not move with
/// omega.
#[allow(clippy::too_many_arguments)]
fn collapsed_omega_energy_re(
    data: &ModelData,
    priors: &Priors,
    state: &State,
    k: usize,
    om_k: &DVector<f64>,
    y_base: &DVector<f64>,
    sigma2: f64,
) -> (f64, Vec<f64>) {
    let n = data.n;
    let nb = data.n_breakpoints;
    let p_b0 = data.x_b0.ncols();
    let p_b1 = data.x_b1.ncols();

    // Per-breakpoint omega, rho and the two transition factors.
    let om_all: Vec<DVector<f64>> = (0..nb).map(|kk| {
        if kk == k { om_k.clone() } else { state.omega_vec(kk, &data.x_om[kk]) }
    }).collect();
    let rho_all: Vec<DVector<f64>> = (0..nb).map(|kk| state.rho_vec(kk, &data.x_rho[kk])).collect();

    let mut ds = vec![vec![0.0f64; n]; nb];    // d * s
    let mut dds = vec![vec![0.0f64; n]; nb];   // d(d*s)/d(omega)
    for kk in 0..nb {
        for i in 0..n {
            let di = data.tau[i] - om_all[kk][i];
            let si = sigmoid(di * rho_all[kk][i]);
            ds[kk][i] = di * si;
            dds[kk][i] = -(si + di * rho_all[kk][i] * si * (1.0 - si));
        }
    }

    // Random-effect contributions, held fixed as coefficients but still moving
    // with omega through the design.
    let mut b1_re = vec![0.0f64; n];
    for j in 0..p_b1 {
        if data.re_mask_b1.get(j).copied().unwrap_or(false) && state.gamma_b1[j] {
            for i in 0..n { b1_re[i] += state.beta_b1[j] * data.x_b1[(i, j)]; }
        }
    }
    let mut d_re = vec![vec![0.0f64; n]; nb];
    for kk in 0..nb {
        for j in 0..data.x_deltas[kk].ncols() {
            if data.re_mask_deltas.get(kk).and_then(|m| m.get(j)).copied().unwrap_or(false)
                && state.gamma_deltas[kk][j] {
                for i in 0..n { d_re[kk][i] += state.beta_deltas[kk][j] * data.x_deltas[kk][(i, j)]; }
            }
        }
    }

    let mut y_tilde = y_base.clone();
    let mut dy_tilde = vec![0.0f64; n];
    for i in 0..n {
        let b1_scale = if nb > 0 { data.tau[i] - om_all[0][i] } else { data.tau[i] };
        y_tilde[i] -= b1_re[i] * b1_scale;
        for kk in 0..nb { y_tilde[i] -= d_re[kk][i] * ds[kk][i]; }
        // d(y~)/d(omega_k) = -d(mu_re)/d(omega_k)
        if nb > 0 && k == 0 { dy_tilde[i] += b1_re[i]; }
        if nb > 0 { dy_tilde[i] -= d_re[k][i] * dds[k][i]; }
    }

    // Fixed linear columns only: the RE columns are already in y~.
    let mut p_total = p_b0;
    for j in 0..p_b1 {
        if !data.re_mask_b1.get(j).copied().unwrap_or(false) { p_total += 1; }
    }
    for kk in 0..nb {
        for j in 0..data.x_deltas[kk].ncols() {
            if !data.re_mask_deltas.get(kk).and_then(|m| m.get(j)).copied().unwrap_or(false) {
                p_total += 1;
            }
        }
    }

    let mut x = DMatrix::<f64>::zeros(n, p_total);
    let mut xi = DMatrix::<f64>::zeros(n, p_total);
    let mut prec_prior = DVector::<f64>::zeros(p_total);
    let mut mu_prior = DVector::<f64>::zeros(p_total);

    let mut c = 0usize;
    for j in 0..p_b0 {
        for i in 0..n { x[(i, c)] = data.x_b0[(i, j)]; }
        prec_prior[c] = 1.0 / (priors.b0_sd[j] * priors.b0_sd[j]);
        mu_prior[c] = priors.b0_mean[j];
        c += 1;
    }
    for j in 0..p_b1 {
        if data.re_mask_b1.get(j).copied().unwrap_or(false) { continue; }
        if state.gamma_b1[j] {
            for i in 0..n {
                let scale = if nb > 0 { data.tau[i] - om_all[0][i] } else { data.tau[i] };
                x[(i, c)] = data.x_b1[(i, j)] * scale;
                if nb > 0 && k == 0 { xi[(i, c)] = -data.x_b1[(i, j)]; }
            }
        }
        prec_prior[c] = 1.0 / (priors.b1_sd[j] * priors.b1_sd[j]);
        mu_prior[c] = priors.b1_mean[j];
        c += 1;
    }
    for kk in 0..nb {
        for j in 0..data.x_deltas[kk].ncols() {
            if data.re_mask_deltas.get(kk).and_then(|m| m.get(j)).copied().unwrap_or(false) { continue; }
            if state.gamma_deltas[kk][j] {
                for i in 0..n {
                    x[(i, c)] = data.x_deltas[kk][(i, j)] * ds[kk][i];
                    if kk == k { xi[(i, c)] = data.x_deltas[kk][(i, j)] * dds[kk][i]; }
                }
            }
            prec_prior[c] = 1.0 / (priors.delta_sd[kk][j] * priors.delta_sd[kk][j]);
            mu_prior[c] = priors.delta_mean[kk][j];
            c += 1;
        }
    }

    let xt = x.transpose();
    let mut precision = &xt * &x / sigma2;
    for j in 0..p_total { precision[(j, j)] += prec_prior[j]; }

    let chol = match precision.clone().cholesky() {
        Some(ch) => ch,
        None => return (f64::INFINITY, vec![0.0; n]),
    };

    let q_vec = &xt * &y_tilde / sigma2 + prec_prior.component_mul(&mu_prior);
    let m_vec = chol.solve(&q_vec);
    let p_inv = chol.inverse();
    let log_det_p: f64 = 2.0 * chol.l().diagonal().iter().map(|v| v.ln()).sum::<f64>();

    let u = 0.5 * y_tilde.dot(&y_tilde) / sigma2 + 0.5 * log_det_p - 0.5 * q_vec.dot(&m_vec);

    let r = &y_tilde - &x * &m_vec;
    let mut grad_om = vec![0.0f64; n];
    let mut xi_i = DVector::<f64>::zeros(p_total);
    for i in 0..n {
        for j in 0..p_total { xi_i[j] = xi[(i, j)]; }
        let pinv_xi = &p_inv * &xi_i;
        let mut quad = 0.0;
        let mut lin = 0.0;
        for j in 0..p_total {
            quad += x[(i, j)] * pinv_xi[j];
            lin += xi_i[j] * m_vec[j];
        }
        grad_om[i] = (quad - lin * r[i] + r[i] * dy_tilde[i]) / sigma2;
    }

    (u, grad_om)
}

// ---------------------------------------------------------------------------
// HMC Steps for Omega and Rho
// ---------------------------------------------------------------------------

fn hmc_step_om(
    data: &ModelData,
    priors: &Priors,
    state: &mut State,
    k: usize,
    cache: &LinearCache,
    adapt: &mut HmcAdapt,
    rng: &mut StdRng,
    nc: bool,
    // When true, only update fixed (non-RE) columns.  RE columns are treated
    // as frozen offsets so that joint_subject_nuts has exclusive ownership of
    // the RE directions, eliminating the (omega_bar + c, u_j - c) null
    // direction that causes mode-switching when both steps update the same
    // columns.
    skip_re: bool,
) {
    let p_full = data.x_om[k].ncols();

    // Which columns are active in this call?
    let active_idx: Vec<usize> = (0..p_full)
        .filter(|&j| !skip_re || !data.re_mask_om[k][j])
        .collect();
    let p = active_idx.len();
    if p == 0 { return; }

    // Return early if all active columns are fixed (sd == 0, not an RE column).
    let all_fixed = active_idx.iter().all(|&j| {
        !data.re_mask_om[k][j] && priors.om_sd[k][j] == 0.0
    });
    if all_fixed { return; }

    let sigma    = state.sigma;
    let sigma_re = state.sigma_re_om[k];
    let mu_base  = cache.mu_without_segment(data, state, k);
    let is_om1   = k == 0 && data.n_breakpoints > 0;

    // NC is only meaningful when RE columns are included in the active set.
    let do_nc = !skip_re && nc && sigma_re > 1e-10;

    // Pre-compute the frozen RE offset: contribution of RE columns at their
    // current values.  Only non-zero when skip_re=true.
    let re_offset: Vec<f64> = if skip_re {
        (0..data.n).map(|i| {
            (0..p_full)
                .filter(|&j| data.re_mask_om[k][j])
                .map(|j| data.x_om[k][(i, j)] * state.beta_om[k][j])
                .sum::<f64>()
        }).collect()
    } else {
        vec![0.0f64; data.n]
    };

    // Build q0 from active columns, applying NC transform where applicable.
    let q0 = DVector::from_vec(
        active_idx.iter().map(|&j| {
            if do_nc && data.re_mask_om[k][j] { state.beta_om[k][j] / sigma_re }
            else                               { state.beta_om[k][j] }
        }).collect::<Vec<_>>()
    );

    // Bounds in HMC space.
    let lb: Vec<f64> = active_idx.iter().map(|&j| {
        if do_nc && data.re_mask_om[k][j] { priors.om_lb[k][j] / sigma_re }
        else { priors.om_lb[k][j] }
    }).collect();
    let ub: Vec<f64> = active_idx.iter().map(|&j| {
        if do_nc && data.re_mask_om[k][j] { priors.om_ub[k][j] / sigma_re }
        else { priors.om_ub[k][j] }
    }).collect();

    // Collapse the fixed linear block out of this update when every collapsed
    // coefficient is unbounded. Only meaningful for the fixed-column pass:
    // with skip_re = false the RE omega columns are in play and the per-subject
    // step owns that geometry.
    let collapsed = skip_re && linear_block_unbounded(data, priors);
    let sigma2 = sigma * sigma;
    let mut y_base = data.y.clone();
    if data.n_groups_b0 > 0 {
        for i in 0..data.n {
            let g = data.group_b0[i];
            if g >= 0 { y_base[i] -= state.u_b0[g as usize]; }
        }
    }

    let energy_fn = |q: &DVector<f64>| -> (f64, DVector<f64>) {
        // om_k[i] = frozen RE offset + active columns × q (with NC undo)
        let om_k: Vec<f64> = (0..data.n).map(|i| {
            let active_contrib: f64 = active_idx.iter().enumerate().map(|(qi, &j)| {
                let beta_j = if do_nc && data.re_mask_om[k][j] { q[qi] * sigma_re }
                             else { q[qi] };
                data.x_om[k][(i, j)] * beta_j
            }).sum();
            re_offset[i] + active_contrib
        }).collect();

        if collapsed {
            // Bounds first: the collapsed energy has no notion of them.
            for (qi, _) in active_idx.iter().enumerate() {
                if q[qi] < lb[qi] || q[qi] > ub[qi] {
                    return (f64::INFINITY, DVector::<f64>::zeros(p));
                }
            }
            let om_vec = DVector::from_vec(om_k.clone());
            let (u_ll, dom) = collapsed_omega_energy_re(
                data, priors, state, k, &om_vec, &y_base, sigma2);
            if !u_ll.is_finite() {
                return (f64::INFINITY, DVector::<f64>::zeros(p));
            }
            let mut lp = 0.0;
            let log_sqrt2pi = 0.5 * std::f64::consts::TAU.ln();
            for (qi, &j) in active_idx.iter().enumerate() {
                let sd = priors.om_sd[k][j];
                let z = (q[qi] - priors.om_mean[k][j]) / sd;
                lp -= 0.5 * z * z + sd.ln() + log_sqrt2pi;
            }
            let mut grad = DVector::<f64>::zeros(p);
            for i in 0..data.n {
                for (qi, &j) in active_idx.iter().enumerate() {
                    grad[qi] += dom[i] * data.x_om[k][(i, j)];
                }
            }
            for (qi, &j) in active_idx.iter().enumerate() {
                let sd = priors.om_sd[k][j];
                grad[qi] += (q[qi] - priors.om_mean[k][j]) / (sd * sd);
            }
            return (u_ll - lp, grad);
        }


        let rho_k   = state.rho_vec(k, &data.x_rho[k]);
        let delta_k = &cache.delta_vals[k];

        let mut mu = mu_base.clone();
        if is_om1 {
            for i in 0..data.n { mu[i] += cache.b1_vals[i] * (data.tau[i] - om_k[i]); }
        }
        for i in 0..data.n {
            let di = data.tau[i] - om_k[i];
            let si = sigmoid(di * rho_k[i]);
            mu[i] += delta_k[i] * di * si;
        }

        let r  = &data.y - &mu;
        let ll = -0.5 * r.dot(&r) / (sigma * sigma);

        // Log-prior over active columns; bounds check.
        let mut lp = 0.0;
        let log_sqrt2pi = 0.5 * std::f64::consts::TAU.ln();
        for (qi, &j) in active_idx.iter().enumerate() {
            let v = q[qi];
            if v < lb[qi] || v > ub[qi] {
                return (f64::INFINITY, DVector::<f64>::zeros(p));
            }
            if do_nc && data.re_mask_om[k][j] {
                lp -= 0.5 * v * v;
            } else {
                let sd = if data.re_mask_om[k][j] { sigma_re } else { priors.om_sd[k][j] };
                let z  = (v - priors.om_mean[k][j]) / sd;
                lp -= 0.5 * z * z + sd.ln() + log_sqrt2pi;
            }
        }

        // Likelihood gradient w.r.t. active columns only.
        let inv_s2 = 1.0 / (sigma * sigma);
        let mut grad_active = vec![0.0f64; p];
        for i in 0..data.n {
            let di  = data.tau[i] - om_k[i];
            let si  = sigmoid(di * rho_k[i]);
            let ri  = rho_k[i];
            let bi  = delta_k[i];
            let mut dmu_dom = -(bi * si + di * ri * si * (1.0 - si) * bi);
            if is_om1 { dmu_dom -= cache.b1_vals[i]; }
            let factor = r[i] * inv_s2 * dmu_dom;
            for (qi, &j) in active_idx.iter().enumerate() {
                grad_active[qi] -= factor * data.x_om[k][(i, j)];
            }
        }

        // Convert to q-space and add prior gradient.
        let mut grad = DVector::<f64>::zeros(p);
        for (qi, &j) in active_idx.iter().enumerate() {
            if do_nc && data.re_mask_om[k][j] {
                grad[qi] = sigma_re * grad_active[qi] + q[qi];
            } else {
                let sd = if data.re_mask_om[k][j] { sigma_re } else { priors.om_sd[k][j] };
                grad[qi] = grad_active[qi] + (q[qi] - priors.om_mean[k][j]) / (sd * sd);
            }
        }

        (-ll - lp, grad)
    };

    let (q_new, accept) = nuts_sample(&q0, energy_fn, adapt, rng, &lb, &ub);

    // Write back: active columns only.
    for (i, &j) in active_idx.iter().enumerate() {
        state.beta_om[k][j] = if do_nc && data.re_mask_om[k][j] { q_new[i] * sigma_re }
                               else                            { q_new[i] };
    }
    adapt.update_epsilon(accept);
}

fn hmc_step_rho(
    data: &ModelData,
    priors: &Priors,
    state: &mut State,
    k: usize,
    cache: &LinearCache,
    adapt: &mut HmcAdapt,
    rng: &mut StdRng,
) {
    let p = adapt.p;

    // If all prior SDs are 0, this parameter block is fixed.
    let mut all_fixed = true;
    for j in 0..p {
        if priors.rho_sd[k][j] > 0.0 { all_fixed = false; break; }
    }
    if all_fixed { return; }

    let sigma = state.sigma;
    let mu_base = cache.mu_without_segment(data, state, k);
    
    let energy_fn = |q: &DVector<f64>| -> (f64, DVector<f64>) {
        let rho_k = &data.x_rho[k] * q;
        let om_k = state.omega_vec(k, &data.x_om[k]);
        let delta_k = &cache.delta_vals[k];
        
        let mut mu = mu_base.clone();
        if k == 0 && data.n_breakpoints > 0 {
             let om1 = state.omega_vec(0, &data.x_om[0]);
             for i in 0..data.n { mu[i] += cache.b1_vals[i] * (data.tau[i] - om1[i]); }
        }

        for i in 0..data.n {
            let di = data.tau[i] - om_k[i];
            let si = sigmoid(di * rho_k[i]);
            mu[i] += delta_k[i] * di * si;
        }
        
        let r = &data.y - &mu;
        let ll = -0.5 * r.dot(&r) / (sigma * sigma);
        let lp = log_truncated_normal_prior(q.as_slice(), &priors.rho_mean[k], &priors.rho_sd[k], &priors.rho_lb[k], &priors.rho_ub[k]);
        
        let inv_s2 = 1.0 / (sigma * sigma);
        let mut grad = DVector::<f64>::zeros(p);
        for i in 0..data.n {
            let di = data.tau[i] - om_k[i];
            let si = sigmoid(di * rho_k[i]);
            let bi = cache.delta_vals[k][i];
            let dmu_drho = di * di * si * (1.0 - si) * bi;
            let factor = r[i] * inv_s2 * dmu_drho;
            for j in 0..p { grad[j] -= factor * data.x_rho[k][(i, j)]; }
        }
        for j in 0..p {
            grad[j] += (q[j] - priors.rho_mean[k][j]) / (priors.rho_sd[k][j] * priors.rho_sd[k][j]);
        }
        
        (-ll - lp, grad)
    };

    let (q_new, accept) = nuts_sample(&state.beta_rho[k], energy_fn, adapt, rng, &priors.rho_lb[k], &priors.rho_ub[k]);
    state.beta_rho[k] = q_new;
    adapt.update_epsilon(accept);
}

// ---------------------------------------------------------------------------
// NUTS (No-U-Turn Sampler) — replaces fixed-L HMC for omega and rho steps
// ---------------------------------------------------------------------------

/// State of a NUTS subtree (Hoffman & Gelman 2011, Algorithm 2).
struct NutsTree {
    q_minus: DVector<f64>,
    p_minus: DVector<f64>,
    grad_minus: DVector<f64>,
    q_plus: DVector<f64>,
    p_plus: DVector<f64>,
    grad_plus: DVector<f64>,
    q_proposal: DVector<f64>,
    n_valid: i64,        // leaf nodes inside the slice
    keep_going: bool,    // no U-turn yet and no divergence
    sum_accept: f64,     // Σ min(1, exp(H0 - H_leaf)) for dual-averaging
    n_evals: usize,      // gradient evaluations in this subtree
    has_divergence: bool,
}

/// Single leapfrog step, using the gradient already computed at q.
/// Returns (q_new, p_new, u_new, grad_new).
fn leapfrog_one<F>(
    q: &DVector<f64>,
    p: &DVector<f64>,
    grad_q: &DVector<f64>,
    v_eps: f64,          // signed step: direction * epsilon
    energy_fn: &mut F,
    inv_mass: &[f64],
    lb: &[f64],
    ub: &[f64],
) -> (DVector<f64>, DVector<f64>, f64, DVector<f64>)
where F: FnMut(&DVector<f64>) -> (f64, DVector<f64>)
{
    let dim = p.len();
    let mut p_half = p.clone();
    for i in 0..dim { p_half[i] -= 0.5 * v_eps * grad_q[i]; }

    let mut q_new = q.clone();
    for i in 0..dim {
        q_new[i] += v_eps * p_half[i] * inv_mass[i];
        for _ in 0..MAX_REFLECTIONS {
            if      q_new[i] < lb[i] { q_new[i] = 2.0*lb[i] - q_new[i]; p_half[i] = -p_half[i]; }
            else if q_new[i] > ub[i] { q_new[i] = 2.0*ub[i] - q_new[i]; p_half[i] = -p_half[i]; }
            else { break; }
        }
    }

    let (u_new, grad_new) = energy_fn(&q_new);
    let mut p_new = p_half;
    for i in 0..dim { p_new[i] -= 0.5 * v_eps * grad_new[i]; }

    (q_new, p_new, u_new, grad_new)
}

/// Recursive tree builder for NUTS.
#[allow(clippy::too_many_arguments)]
fn build_tree<F>(
    q: &DVector<f64>,
    p: &DVector<f64>,
    grad_q: &DVector<f64>,
    log_u: f64,   // log of the slice variable
    v: f64,       // direction: +1 or -1
    depth: usize,
    eps: f64,
    energy_fn: &mut F,
    inv_mass: &[f64],
    lb: &[f64],
    ub: &[f64],
    h0: f64,      // initial Hamiltonian (for divergence check and alpha)
    rng: &mut StdRng,
) -> NutsTree
where F: FnMut(&DVector<f64>) -> (f64, DVector<f64>)
{
    if depth == 0 {
        // Base case: one leapfrog step
        let (q_new, p_new, u_new, grad_new) = leapfrog_one(
            q, p, grad_q, v * eps, energy_fn, inv_mass, lb, ub,
        );
        let k_new: f64 = p_new.iter().zip(inv_mass).map(|(pi, mi)| pi*pi*mi).sum::<f64>() * 0.5;
        let h_new = u_new + k_new;
        let dh = h_new - h0;

        let in_slice     = h_new.is_finite() && h_new <= -log_u;
        let not_diverged = h_new.is_finite() && dh.abs() < DIVERGENCE_THRESHOLD;
        // Acceptance probability for this leaf (used in dual-averaging)
        let alpha = if dh.is_finite() { (-dh).exp().min(1.0) } else { 0.0 };

        NutsTree {
            q_minus: q_new.clone(), p_minus: p_new.clone(), grad_minus: grad_new.clone(),
            q_plus:  q_new.clone(), p_plus:  p_new.clone(), grad_plus:  grad_new.clone(),
            q_proposal: q_new,
            n_valid: if in_slice { 1 } else { 0 },
            keep_going: not_diverged,
            sum_accept: alpha,
            n_evals: 1,
            has_divergence: !not_diverged,
        }
    } else {
        // Recursive case: build first half-tree
        let mut tree = build_tree(
            q, p, grad_q, log_u, v, depth - 1, eps,
            energy_fn, inv_mass, lb, ub, h0, rng,
        );

        if tree.keep_going {
            // Build second half-tree from the frontier endpoint
            let (q2, p2, grad2) = if v < 0.0 {
                (tree.q_minus.clone(), tree.p_minus.clone(), tree.grad_minus.clone())
            } else {
                (tree.q_plus.clone(), tree.p_plus.clone(), tree.grad_plus.clone())
            };

            let tree2 = build_tree(
                &q2, &p2, &grad2, log_u, v, depth - 1, eps,
                energy_fn, inv_mass, lb, ub, h0, rng,
            );

            // Extend the appropriate endpoint
            if v < 0.0 {
                tree.q_minus    = tree2.q_minus;
                tree.p_minus    = tree2.p_minus;
                tree.grad_minus = tree2.grad_minus;
            } else {
                tree.q_plus    = tree2.q_plus;
                tree.p_plus    = tree2.p_plus;
                tree.grad_plus = tree2.grad_plus;
            }

            // Biased progressive sampling of the proposal
            if tree2.n_valid > 0 {
                let n_total = tree.n_valid + tree2.n_valid;
                let p_swap = (tree2.n_valid as f64) / (n_total as f64).max(f64::EPSILON);
                if rng.gen::<f64>() < p_swap {
                    tree.q_proposal = tree2.q_proposal;
                }
            }

            tree.n_valid      += tree2.n_valid;
            tree.sum_accept   += tree2.sum_accept;
            tree.n_evals      += tree2.n_evals;
            tree.has_divergence = tree.has_divergence || tree2.has_divergence;

            // No-U-turn check across the full span
            let delta = &tree.q_plus - &tree.q_minus;
            let no_uturn = delta.dot(&tree.p_minus) >= 0.0
                        && delta.dot(&tree.p_plus)  >= 0.0;
            tree.keep_going = tree2.keep_going && no_uturn;
        }

        tree
    }
}

/// NUTS sampler — drop-in replacement for hmc_sample with the same signature.
fn nuts_sample<F>(
    q0: &DVector<f64>,
    mut energy_fn: F,
    adapt: &mut HmcAdapt,
    rng: &mut StdRng,
    lb: &[f64],
    ub: &[f64],
) -> (DVector<f64>, f64)
where F: FnMut(&DVector<f64>) -> (f64, DVector<f64>)
{
    let dim     = adapt.p;
    let eps     = adapt.epsilon;
    // Clone inv_mass to avoid holding a borrow on adapt during energy_fn calls
    let inv_mass = adapt.inv_mass.clone();

    // Sample initial momentum p0 ~ N(0, M)
    let normal = Normal::new(0.0, 1.0).unwrap();
    let p0 = DVector::<f64>::from_iterator(
        dim,
        (0..dim).map(|i| normal.sample(rng) / inv_mass[i].sqrt()),
    );

    // Hamiltonian at start
    let (u0, grad0) = energy_fn(q0);
    let k0: f64 = (0..dim).map(|i| p0[i]*p0[i]*inv_mass[i]).sum::<f64>() * 0.5;
    let h0 = u0 + k0;

    // Slice variable: u ~ Uniform(0, exp(-H0)), work in log-space
    // log_u = log(Uniform(0,1)) - H0; accept leaf if H_leaf <= -log_u
    let log_u: f64 = rng.gen::<f64>().ln() - h0;

    // Initialise spanning endpoints and proposal
    let mut q_minus    = q0.clone();
    let mut p_minus    = p0.clone();
    let mut grad_minus = grad0.clone();
    let mut q_plus     = q0.clone();
    let mut p_plus     = p0.clone();
    let mut grad_plus  = grad0.clone();
    let mut q_prop     = q0.clone();
    let mut n_valid:   i64 = 1;  // q0 is always in its own slice
    let mut keep_going     = true;
    let mut sum_accept     = 0.0_f64;
    let mut n_evals: usize = 0;
    let mut any_divergence = false;

    for j in 0..NUTS_MAX_DEPTH {
        if !keep_going { break; }

        let v: f64 = if rng.gen_bool(0.5) { 1.0 } else { -1.0 };

        let subtree = if v < 0.0 {
            build_tree(
                &q_minus, &p_minus, &grad_minus,
                log_u, v, j, eps,
                &mut energy_fn, &inv_mass, lb, ub, h0, rng,
            )
        } else {
            build_tree(
                &q_plus, &p_plus, &grad_plus,
                log_u, v, j, eps,
                &mut energy_fn, &inv_mass, lb, ub, h0, rng,
            )
        };

        // Extend spanning endpoints
        if v < 0.0 {
            q_minus    = subtree.q_minus;
            p_minus    = subtree.p_minus;
            grad_minus = subtree.grad_minus;
        } else {
            q_plus    = subtree.q_plus;
            p_plus    = subtree.p_plus;
            grad_plus = subtree.grad_plus;
        }

        // Biased progressive proposal update (Hoffman & Gelman 2014, Algorithm 3).
        //
        // The `subtree.keep_going` guard is required, not an optimisation. When
        // the new subtree terminates on a U-turn or a divergence its doubling is
        // rejected, and the states inside it are ones the reverse trajectory
        // could not have reached; proposing from it anyway breaks reversibility.
        // U-turn termination is the ordinary way the doubling loop ends, not a
        // rare event, so omitting the guard biases a large share of transitions.
        if subtree.keep_going && subtree.n_valid > 0 {
            let p_swap = ((subtree.n_valid as f64) / (n_valid as f64)).min(1.0);
            if rng.gen::<f64>() < p_swap { q_prop = subtree.q_proposal; }
        }

        n_valid      += subtree.n_valid;
        sum_accept   += subtree.sum_accept;
        n_evals      += subtree.n_evals;
        any_divergence = any_divergence || subtree.has_divergence;

        // Global no-U-turn check
        let delta   = &q_plus - &q_minus;
        let no_uturn = delta.dot(&p_minus) >= 0.0 && delta.dot(&p_plus) >= 0.0;
        keep_going  = subtree.keep_going && no_uturn;
    }

    // Average acceptance probability for dual-averaging
    let avg_alpha = if n_evals > 0 {
        (sum_accept / n_evals as f64).clamp(0.0, 1.0)
    } else {
        0.0
    };

    // Record divergences (uses same infrastructure as fixed-L HMC)
    if any_divergence {
        adapt.record_energy_error(DIVERGENCE_THRESHOLD + 1.0);
    }

    (q_prop, avg_alpha)
}

// ---------------------------------------------------------------------------
// Joint per-subject NUTS with within-subject Fisher metric (geometry-aware)
// ---------------------------------------------------------------------------

/// For each subject s, jointly updates all per-subject RE:
///   q_s = [u_b1_s?, u_delta_1_s?, ..., u_delta_K_s?, u_omega_1_s?, ..., u_omega_K_s?]
///
/// The within-subject Fisher metric G_s (analytic, ≤ (1+2K)×(1+2K)) is applied
/// via SoftAbs to give a PD mass matrix for NUTS.  This captures the off-diagonal
/// coupling between slopes and changepoints that makes naive per-parameter HMC slow.
#[allow(clippy::too_many_arguments)]
fn joint_subject_nuts(
    data: &ModelData,
    priors: &Priors,
    state: &mut State,
    adapt_per_subject: &mut Vec<HmcAdapt>,
    rng: &mut StdRng,
    nc_b1: bool,
    nc_deltas: &[bool],
    nc_omega: &[Vec<bool>],
) {
    let n = data.n;
    let nb = data.n_breakpoints;
    let sigma = state.sigma;
    let inv_s2 = 1.0 / (sigma * sigma);

    // Precompute current omega, rho, delta per observation (fixing all non-RE)
    // We'll recompute inside the closure for each subject.

    // For each subject s, identify which columns in each RE block belong to s.
    // With identity coding there is exactly one RE column per subject in each RE block.
    // Column index for subject s in beta_b1 RE: among all j where re_mask_b1[j]==true, subject s
    // maps to the s-th such j (0-indexed among RE subjects).
    // We precompute a lookup: re_col_b1[s], re_col_deltas[k][s], re_col_omega[k][s]

    let has_re_b1 = data.re_mask_b1.iter().any(|&m| m);
    let has_re_deltas: Vec<bool> = (0..nb).map(|k| data.re_mask_deltas[k].iter().any(|&m| m)).collect();
    let has_re_omega: Vec<bool>  = (0..nb).map(|k| data.re_mask_om[k].iter().any(|&m| m)).collect();

    let n_sub = data.n_subjects;
    if n_sub == 0 { return; }

    // Build lookup: for each RE block, what column in the state vector corresponds to subject s?
    let re_cols_b1: Vec<usize> = data.re_mask_b1.iter().enumerate()
        .filter_map(|(j, &m)| if m { Some(j) } else { None }).collect();
    let re_cols_deltas: Vec<Vec<usize>> = (0..nb).map(|k| {
        data.re_mask_deltas[k].iter().enumerate()
            .filter_map(|(j, &m)| if m { Some(j) } else { None }).collect()
    }).collect();
    let re_cols_omega: Vec<Vec<usize>> = (0..nb).map(|k| {
        data.re_mask_om[k].iter().enumerate()
            .filter_map(|(j, &m)| if m { Some(j) } else { None }).collect()
    }).collect();

    // Observation-to-subject mapping (via group_re)
    // Precompute per-subject observation indices
    let mut obs_for_subject: Vec<Vec<usize>> = vec![Vec::new(); n_sub];
    for i in 0..n {
        let s = data.group_re[i];
        if s >= 0 { obs_for_subject[s as usize].push(i); }
    }

    // Precompute global mu without any RE contribution (we'll add per-subject RE inside)
    // Use a modified state with all per-subject RE zeroed
    let mut state_no_re = state.clone();
    for j in 0..state_no_re.beta_b1.len() {
        if data.re_mask_b1[j] { state_no_re.beta_b1[j] = 0.0; }
    }
    for k in 0..nb {
        for j in 0..state_no_re.beta_deltas[k].len() {
            if data.re_mask_deltas[k][j] { state_no_re.beta_deltas[k][j] = 0.0; }
        }
        for j in 0..state_no_re.beta_om[k].len() {
            if data.re_mask_om[k][j] { state_no_re.beta_om[k][j] = 0.0; }
        }
    }
    let mu_no_re = state_no_re.means(data);   // N-vector: contribution of all non-RE terms

    // Build per-subject bounds: b1 bounds come from priors.b1_lb/ub for the RE columns,
    // delta bounds from priors.delta_lb/ub, omega bounds from priors.om_lb/ub.
    // Per-subject dim: 1 (b1) + nb (delta) + nb (omega) at most.

    for s in 0..n_sub {
        let obs = &obs_for_subject[s];
        if obs.is_empty() { continue; }

        // Assemble q_s and bounds
        let mut q_s_vals: Vec<f64> = Vec::new();
        let mut lb_s: Vec<f64> = Vec::new();
        let mut ub_s: Vec<f64> = Vec::new();
        let mut sigma_re_s: Vec<f64> = Vec::new();

        if has_re_b1 && s < re_cols_b1.len() {
            let j = re_cols_b1[s];
            q_s_vals.push(state.beta_b1[j]);
            lb_s.push(priors.b1_lb[j]);
            ub_s.push(priors.b1_ub[j]);
            sigma_re_s.push(state.sigma_re_b1);
        }
        for k in 0..nb {
            if has_re_deltas[k] && s < re_cols_deltas[k].len() {
                let j = re_cols_deltas[k][s];
                q_s_vals.push(state.beta_deltas[k][j]);
                lb_s.push(priors.delta_lb[k][j]);
                ub_s.push(priors.delta_ub[k][j]);
                sigma_re_s.push(state.sigma_re_deltas[k]);
            }
        }
        for k in 0..nb {
            if has_re_omega[k] && s < re_cols_omega[k].len() {
                let j = re_cols_omega[k][s];
                q_s_vals.push(state.beta_om[k][j]);
                // RE columns are deviations from the population intercept: they
                // are unconstrained (prior is N(0, sigma_re^2)).  The user's
                // omega prior bounds (lb/ub) apply only to the fixed intercept.
                lb_s.push(f64::NEG_INFINITY);
                ub_s.push(f64::INFINITY);
                sigma_re_s.push(state.sigma_re_om[k]);
            }
        }

        let d_s = q_s_vals.len();
        if d_s == 0 { continue; }

        // NC transform: z[i] = q[i] / sigma_re[i] for each component
        // Determine which components use NC
        let mut nc_flags: Vec<bool> = Vec::with_capacity(d_s);
        {
            let mut idx = 0usize;
            if has_re_b1 && s < re_cols_b1.len() {
                nc_flags.push(nc_b1);
                idx += 1;
            }
            for k in 0..nb {
                if has_re_deltas[k] && s < re_cols_deltas[k].len() {
                    nc_flags.push(if k < nc_deltas.len() { nc_deltas[k] } else { false });
                    idx += 1;
                }
            }
            for k in 0..nb {
                if has_re_omega[k] && s < re_cols_omega[k].len() {
                    // Per-subject NC flag: nc_omega[k][s] if available, else centred.
                    let flag = k < nc_omega.len() && s < nc_omega[k].len() && nc_omega[k][s];
                    nc_flags.push(flag);
                    idx += 1;
                }
            }
            let _ = idx;
        }

        // Current per-subject contributions to mu (we need these to form the residual inside closure)
        // The closure will recompute mu from q_s each call, using mu_no_re as the baseline.

        // Clone what the closure needs
        let obs_s = obs.clone();
        let tau_s: Vec<f64> = obs_s.iter().map(|&i| data.tau[i]).collect();
        let y_s: Vec<f64>   = obs_s.iter().map(|&i| data.y[i]).collect();
        let n_s = obs_s.len();


        // Design column slices for this subject in each RE block
        let x_b1_s: Vec<f64> = if has_re_b1 && s < re_cols_b1.len() {
            let j = re_cols_b1[s];
            obs_s.iter().map(|&i| data.x_b1[(i, j)]).collect()
        } else { vec![] };

        let x_delta_s: Vec<Vec<f64>> = (0..nb).map(|k| {
            if has_re_deltas[k] && s < re_cols_deltas[k].len() {
                let j = re_cols_deltas[k][s];
                obs_s.iter().map(|&i| data.x_deltas[k][(i, j)]).collect()
            } else { vec![] }
        }).collect();

        let x_om_s: Vec<Vec<f64>> = (0..nb).map(|k| {
            if has_re_omega[k] && s < re_cols_omega[k].len() {
                let j = re_cols_omega[k][s];
                obs_s.iter().map(|&i| data.x_om[k][(i, j)]).collect()
            } else { vec![] }
        }).collect();

        // All rho values for this subject (fixed, not per-subject RE)
        let rho_s: Vec<Vec<f64>> = (0..nb).map(|k| {
            let rho_k = state.rho_vec(k, &data.x_rho[k]);
            obs_s.iter().map(|&i| rho_k[i]).collect()
        }).collect();

        // Global omega for non-RE columns (to compute d_ki baseline)
        let om_fixed_s: Vec<Vec<f64>> = (0..nb).map(|k| {
            // omega contribution from FIXED columns only (RE column zeroed)
            let mut beta_fixed = state.beta_om[k].clone();
            for j in 0..beta_fixed.len() {
                if data.re_mask_om[k][j] { beta_fixed[j] = 0.0; }
            }
            let om_fixed = &data.x_om[k] * &beta_fixed;
            obs_s.iter().map(|&i| om_fixed[i]).collect()
        }).collect();

        // b1 effective values (fixed columns only)
        let b1_fixed_s: Vec<f64> = {
            let mut b1_eff = state.beta_b1.clone();
            for j in 0..b1_eff.len() {
                if !state.gamma_b1[j] || data.re_mask_b1[j] { b1_eff[j] = 0.0; }
            }
            let b1_vec = &data.x_b1 * &b1_eff;
            obs_s.iter().map(|&i| b1_vec[i]).collect()
        };

        // delta effective (fixed columns only)
        let delta_fixed_s: Vec<Vec<f64>> = (0..nb).map(|k| {
            let mut bd_eff = state.beta_deltas[k].clone();
            for j in 0..bd_eff.len() {
                if !state.gamma_deltas[k][j] || data.re_mask_deltas[k][j] { bd_eff[j] = 0.0; }
            }
            let d_vec = &data.x_deltas[k] * &bd_eff;
            obs_s.iter().map(|&i| d_vec[i]).collect()
        }).collect();

        // b0_only_s: mu_no_re[i] minus the fixed b1 and delta contributions computed at
        // omega_fixed (RE = 0). The energy closure recomputes these at the proposed omega_s,
        // so the baseline must be pure b0 to avoid double-counting at the wrong omega.
        let b0_only_s: Vec<f64> = obs_s.iter().enumerate().map(|(li, &i)| {
            let mut b0 = mu_no_re[i];
            if nb > 0 {
                b0 -= b1_fixed_s[li] * (data.tau[i] - om_fixed_s[0][li]);
            }
            for k in 0..nb {
                let d_old = data.tau[i] - om_fixed_s[k][li];
                let s_old = sigmoid(d_old * rho_s[k][li]);
                b0 -= delta_fixed_s[k][li] * d_old * s_old;
            }
            b0
        }).collect();

        let sigma_re_s_clone = sigma_re_s.clone();
        let has_re_b1_s  = has_re_b1 && s < re_cols_b1.len();
        let has_re_del_s: Vec<bool> = (0..nb).map(|k| has_re_deltas[k] && s < re_cols_deltas[k].len()).collect();
        let has_re_om_s:  Vec<bool> = (0..nb).map(|k| has_re_omega[k]  && s < re_cols_omega[k].len()).collect();

        let g_b1 = if has_re_b1_s { state.gamma_b1[re_cols_b1[s]] } else { false };
        let g_del: Vec<bool> = (0..nb).map(|k| {
            if has_re_del_s[k] { state.gamma_deltas[k][re_cols_deltas[k][s]] } else { false }
        }).collect();

        // ── Energy function for subject s ──────────────────────────────────
        // q = [u_b1_s?, u_delta_1_s?, ..., u_omega_1_s?, ...]  in NC or centred coords
        let energy_fn = |q: &DVector<f64>| -> (f64, DVector<f64>) {
            let mut idx = 0usize;

            // Convert q to beta values (undo NC if needed)
            let u_b1_s = if has_re_b1_s {
                let v = if nc_flags[idx] { q[idx] * sigma_re_s_clone[idx] } else { q[idx] };
                idx += 1; v
            } else { 0.0 };

            let u_del_s: Vec<f64> = (0..nb).map(|k| {
                if has_re_del_s[k] {
                    let v = if nc_flags[idx] { q[idx] * sigma_re_s_clone[idx] } else { q[idx] };
                    idx += 1; v
                } else { 0.0 }
            }).collect();

            let u_om_s: Vec<f64> = (0..nb).map(|k| {
                if has_re_om_s[k] {
                    let v = if nc_flags[idx] { q[idx] * sigma_re_s_clone[idx] } else { q[idx] };
                    idx += 1; v
                } else { 0.0 }
            }).collect();

            // Compute mu for each obs in subject s
            let mut ll = 0.0f64;
            let mut grad_beta = vec![0.0f64; d_s];

            for li in 0..n_s {
                let tau_i = tau_s[li];
                let y_i   = y_s[li];

                // omega per breakpoint for this obs (fixed base + subject RE)
                let omega_i: Vec<f64> = (0..nb).map(|k| om_fixed_s[k][li] + u_om_s[k] * x_om_s[k].get(li).copied().unwrap_or(0.0)).collect();

                // d_ki, s_ki for each breakpoint
                let d_ki: Vec<f64> = (0..nb).map(|k| tau_i - omega_i[k]).collect();
                let s_ki: Vec<f64> = (0..nb).map(|k| sigmoid(d_ki[k] * rho_s[k][li])).collect();

                // b1 contribution
                let om1_i = if nb > 0 { omega_i[0] } else { 0.0 };
                let b1_scale = if nb > 0 { tau_i - om1_i } else { tau_i };
                let b1_contrib = b1_fixed_s[li] * b1_scale
                    + if has_re_b1_s && g_b1 { u_b1_s * x_b1_s[li] * b1_scale } else { 0.0 };

                // delta contributions
                let mut delta_contrib = 0.0f64;
                for k in 0..nb {
                    delta_contrib += delta_fixed_s[k][li] * d_ki[k] * s_ki[k];
                    if has_re_del_s[k] && g_del[k] {
                        delta_contrib += u_del_s[k] * x_delta_s[k][li] * d_ki[k] * s_ki[k];
                    }
                }

                let mu_i = b0_only_s[li] + b1_contrib + delta_contrib;
                let r_i  = y_i - mu_i;
                ll += -0.5 * r_i * r_i * inv_s2;

                // Gradients in beta space
                let mut qi = 0usize;

                // dmu/d(u_b1_s)
                if has_re_b1_s {
                    let dmu = if g_b1 { x_b1_s[li] * b1_scale } else { 0.0 };
                    grad_beta[qi] -= r_i * inv_s2 * dmu;
                    qi += 1;
                }

                // dmu/d(u_delta_k_s)
                for k in 0..nb {
                    if has_re_del_s[k] {
                        let dmu = if g_del[k] { x_delta_s[k][li] * d_ki[k] * s_ki[k] } else { 0.0 };
                        grad_beta[qi] -= r_i * inv_s2 * dmu;
                        qi += 1;
                    }
                }

                // dmu/d(u_omega_k_s)
                for k in 0..nb {
                    if has_re_om_s[k] {
                        let bi = delta_fixed_s[k][li]
                            + if has_re_del_s[k] && g_del[k] { u_del_s[k] * x_delta_s[k][li] } else { 0.0 };
                        let ri_k = rho_s[k][li];
                        let mut dmu_dom = -(bi * s_ki[k] + d_ki[k] * ri_k * s_ki[k] * (1.0 - s_ki[k]) * bi);
                        if k == 0 {
                            let b1_i = b1_fixed_s[li] + if has_re_b1_s && g_b1 { u_b1_s * x_b1_s[li] } else { 0.0 };
                            dmu_dom -= b1_i;
                        }
                        grad_beta[qi] -= r_i * inv_s2 * dmu_dom * x_om_s[k][li];
                        qi += 1;
                    }
                }
                let _ = qi;
            }

            // Prior terms and chain-rule for NC
            let mut lp = 0.0f64;
            let mut grad = vec![0.0f64; d_s];
            for i in 0..d_s {
                let sr = sigma_re_s_clone[i];
                if nc_flags[i] {
                    // z[i] ~ N(0,1); beta = z * sr
                    // grad_z = sr * grad_beta + z
                    lp -= 0.5 * q[i] * q[i];
                    grad[i] = sr * grad_beta[i] + q[i];
                } else {
                    // beta ~ N(0, sr^2)
                    lp -= 0.5 * (q[i] / sr).powi(2);
                    grad[i] = grad_beta[i] + q[i] / (sr * sr);
                }
            }

            let u = -ll - lp;
            (u, DVector::from_vec(grad))
        };

        let q0 = DVector::from_vec({
            let mut v = q_s_vals.clone();
            // Apply NC transform to initial q
            for i in 0..d_s {
                if nc_flags[i] && sigma_re_s[i] > 1e-10 { v[i] /= sigma_re_s[i]; }
            }
            v
        });

        // Bounds in q-space (NC-adjusted where applicable)
        let lb_q: Vec<f64> = lb_s.iter().enumerate().map(|(i, &lb)| {
            if nc_flags[i] && sigma_re_s[i] > 1e-10 { lb / sigma_re_s[i] } else { lb }
        }).collect();
        let ub_q: Vec<f64> = ub_s.iter().enumerate().map(|(i, &ub)| {
            if nc_flags[i] && sigma_re_s[i] > 1e-10 { ub / sigma_re_s[i] } else { ub }
        }).collect();

        // NUTS step
        let (q_new, accept) = nuts_sample(
            &q0, energy_fn, &mut adapt_per_subject[s], rng, &lb_q, &ub_q,
        );

        // Write back: undo NC transform → beta values
        {
            let mut qi = 0usize;
            if has_re_b1_s {
                let j = re_cols_b1[s];
                state.beta_b1[j] = if nc_flags[qi] { q_new[qi] * sigma_re_s[qi] } else { q_new[qi] };
                qi += 1;
            }
            for k in 0..nb {
                if has_re_del_s[k] {
                    let j = re_cols_deltas[k][s];
                    state.beta_deltas[k][j] = if nc_flags[qi] { q_new[qi] * sigma_re_s[qi] } else { q_new[qi] };
                    qi += 1;
                }
            }
            for k in 0..nb {
                if has_re_om_s[k] {
                    let j = re_cols_omega[k][s];
                    state.beta_om[k][j] = if nc_flags[qi] { q_new[qi] * sigma_re_s[qi] } else { q_new[qi] };
                    qi += 1;
                }
            }
        }

        adapt_per_subject[s].update_epsilon(accept);
        if adapt_per_subject[s].adapting {
            adapt_per_subject[s].observe(&q_new);
            if adapt_per_subject[s].da_count % 500 == 0 {
                adapt_per_subject[s].refresh_mass_matrix();
            }
        }
    }
}

fn sample_truncated_normal(mean: f64, sd: f64, lb: f64, ub: f64, rng: &mut StdRng) -> f64 {
    let normal = rand_distr::Normal::new(mean, sd).unwrap();
    for _ in 0..1000 {
        let val = normal.sample(rng);
        if val >= lb && val <= ub {
            return val;
        }
    }
    normal.sample(rng).clamp(lb, ub)
}

fn omega_translation_step(
    data: &ModelData,
    priors: &Priors,
    state: &mut State,
    k: usize,
    rng: &mut StdRng,
) {
    let has_re = data.re_mask_om[k].iter().any(|&m| m);
    if !has_re { return; }

    let n_sub = data.n_subjects;
    if n_sub == 0 { return; }

    let re_cols: Vec<usize> = data.re_mask_om[k].iter().enumerate()
        .filter_map(|(j, &m)| if m { Some(j) } else { None }).collect();

    if re_cols.len() != n_sub { return; }

    let omega_bar = state.beta_om[k][0];
    let mean_bar = priors.om_mean[k][0];
    let sd_bar = priors.om_sd[k][0];

    let sigma_re = state.sigma_re_om[k];
    if sigma_re <= 1e-10 { return; }

    // state.beta_om always holds CENTRED values: omega_vec() multiplies them into
    // the likelihood directly, and every sampling step that works in non-centred
    // coordinates converts back on write-back (hmc_step_om, joint_subject_nuts).
    // So no sigma_re rescaling belongs here, whatever nc_om says.
    let mut sum_u = 0.0;
    for s in 0..n_sub {
        sum_u += state.beta_om[k][re_cols[s]];
    }

    let prec = 1.0 / (sd_bar * sd_bar) + (n_sub as f64) / (sigma_re * sigma_re);
    let var_c = 1.0 / prec;
    let sd_c = var_c.sqrt();
    let mean_c = var_c * ( (mean_bar - omega_bar) / (sd_bar * sd_bar) + sum_u / (sigma_re * sigma_re) );

    let lb_c = priors.om_lb[k][0] - omega_bar;
    let ub_c = priors.om_ub[k][0] - omega_bar;

    let c = sample_truncated_normal(mean_c, sd_c, lb_c, ub_c, rng);

    // (omega_bar + c, u_s - c) leaves every omega_i = omega_bar + u_s unchanged,
    // so the likelihood is invariant and c is drawn from its prior-only
    // conditional above.  Shifting the RE columns by anything other than c would
    // break that invariance while still being accepted with probability one.
    state.beta_om[k][0] += c;
    for s in 0..n_sub {
        state.beta_om[k][re_cols[s]] -= c;
    }
}

// ---------------------------------------------------------------------------
// Main Chain Loops
// ---------------------------------------------------------------------------

pub fn run_chain_re(
    data: &ModelData, priors: &Priors, n_iter: usize, n_warmup: usize,
    step_om_init: f64, step_rho_init: f64, target_accept: f64,
    seed: u64, verbose: bool, chain_id: usize, n_chains: usize,
    nc_om: &[Vec<bool>],
    nc_b1: bool,
    nc_deltas: &[bool],
    progress_fn: &dyn Fn(usize, usize, usize, usize, bool),
) -> (DMatrix<f64>, [usize; 4]) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut state = init_state_re(data, priors, &mut rng);
    let n_post = n_iter - n_warmup;
    let n_params = state.n_params(false, false, true);
    let mut draws = DMatrix::<f64>::zeros(n_post, n_params);

    // Per-subject NUTS adapters (one per subject for the joint RE update)
    let d_s = (if data.re_mask_b1.iter().any(|&m| m) { 1 } else { 0 })
            + data.re_mask_deltas.iter().filter(|m| m.iter().any(|&v| v)).count()
            + data.re_mask_om.iter().filter(|m| m.iter().any(|&v| v)).count();
    let mut adapt_subj: Vec<HmcAdapt> = (0..data.n_subjects)
        .map(|_| HmcAdapt::new(d_s.max(1), step_om_init, target_accept))
        .collect();

    // Per-breakpoint adapters for the fixed-only omega update.
    // Dimension = number of fixed (non-RE) columns in x_om
    let mut adapt_om: Vec<HmcAdapt> = (0..data.n_breakpoints)
        .map(|k| {
            let p_fixed = data.x_om[k].ncols() - data.re_mask_om[k].iter().filter(|&&m| m).count();
            HmcAdapt::new(p_fixed, step_om_init, target_accept)
        })
        .collect();

    let mut adapt_rho: Vec<HmcAdapt> = (0..data.n_breakpoints)
        .map(|k| HmcAdapt::new(data.x_rho[k].ncols(), step_rho_init, target_accept))
        .collect();

    let report_every = (n_iter / 10).max(1);

    for iter in 0..n_iter {
        if verbose && iter % report_every == 0 {
            progress_fn(chain_id, n_chains, iter, n_iter, iter < n_warmup);
        }

        sample_linear_coefs(data, priors, &mut state, &mut rng);
        if data.n_groups_b0 > 0 { sample_random_effects(data, priors, &mut state, &mut rng); }
        // Step along the (b0, u) ridge. Only b0 + u_j is identified within a
        // group, so the two conjugate blocks above cannot traverse it on their
        // own; this move is likelihood-invariant and accepted outright.
        if data.n_groups_b0 > 0 { b0_translation_step(data, priors, &mut state, &mut rng); }

        // Joint per-subject NUTS for all RE (b1, delta, omega)
        joint_subject_nuts(data, priors, &mut state, &mut adapt_subj, &mut rng,
                           nc_b1, nc_deltas, nc_om);

        for k in 0..data.n_breakpoints {
            let cache_k = LinearCache::build(&state, data);
            let nc_flag = k < nc_om.len() && nc_om[k].iter().any(|&x| x);
            hmc_step_om(data, priors, &mut state, k, &cache_k, &mut adapt_om[k], &mut rng,
                        nc_flag, true);
            omega_translation_step(data, priors, &mut state, k, &mut rng);
        }

        // Rho update (conditions on current omega)
        // Redraw the linear block at the new omega before rho conditions on it.
        // The collapsed omega update deliberately leaves beta untouched, so
        // without this rho would be fitted against coefficients belonging to the
        // previous change-point. That mismatch is what drives the chain onto the
        // degenerate rho -> 0 ridge, where the transition weight flattens to a
        // constant and delta trades off against the baseline slope.
        sample_linear_coefs(data, priors, &mut state, &mut rng);

        let cache = LinearCache::build(&state, data);
        for k in 0..data.n_breakpoints {
            hmc_step_rho(data, priors, &mut state, k, &cache, &mut adapt_rho[k], &mut rng);
        }

        sample_linear_coefs(data, priors, &mut state, &mut rng);
        sample_sigma(data, priors, &mut state, &mut rng);
        if data.n_groups_b0 > 0 { sample_sigma_u(priors, &mut state, &mut rng); }
        sample_sigma_re_om(data, priors, &mut state, &mut rng);
        sample_sigma_re_b1(data, priors, &mut state, &mut rng);
        sample_sigma_re_deltas(data, priors, &mut state, &mut rng);

        if iter < n_warmup {
            for k in 0..data.n_breakpoints {
                let fixed_om = DVector::from_iterator(
                    adapt_om[k].p,
                    state.beta_om[k].iter().enumerate()
                        .filter(|&(j, _)| !data.re_mask_om[k][j])
                        .map(|(_, &v)| v)
                );
                adapt_om[k].observe(&fixed_om);
                if (iter + 1) % 500 == 0 { adapt_om[k].refresh_mass_matrix(); }
                adapt_rho[k].observe(&state.beta_rho[k]);
                if (iter + 1) % 500 == 0 { adapt_rho[k].refresh_mass_matrix(); }
            }
        } else if iter == n_warmup {
            for k in 0..data.n_breakpoints {
                adapt_om[k].freeze();
                adapt_rho[k].freeze();
            }
            for s in 0..data.n_subjects { adapt_subj[s].freeze(); }
        }

        if iter >= n_warmup {
            let row = iter - n_warmup;
            let draw = state.to_vec(false, false, true);
            for (col, &val) in draw.iter().enumerate() { draws[(row, col)] = val; }
        }
    }
    let div_subj = adapt_subj.iter().map(|h| h.n_divergent).sum::<usize>();
    let div_om = adapt_om.iter().map(|h| h.n_divergent).sum::<usize>();
    let div_rho = adapt_rho.iter().map(|h| h.n_divergent).sum::<usize>();
    println!("Chain {} finished. Divergences - Subj: {}, Om: {}, Rho: {}", chain_id, div_subj, div_om, div_rho);
    let n_div = div_subj + div_om + div_rho;
    // [total, subj, om, rho]
    (draws, [n_div, div_subj, div_om, div_rho])
}
pub fn run_chain_re_ss(
    data: &ModelData, priors: &Priors, ss: &SpikeSlabConfig, n_iter: usize, n_warmup: usize,
    step_om_init: f64, step_rho_init: f64, target_accept: f64,
    seed: u64, verbose: bool, chain_id: usize, n_chains: usize,
    nc_om: &[Vec<bool>],
    nc_b1: bool,
    nc_deltas: &[bool],
    progress_fn: &dyn Fn(usize, usize, usize, usize, bool),
) -> (DMatrix<f64>, [usize; 4]) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut state = init_state_re(data, priors, &mut rng);
    state.pi = ss.pi_init;
    let n_post = n_iter - n_warmup;
    let learn_pi = ss.beta_a > 0.0;
    let n_params = state.n_params(true, learn_pi, true);
    let mut draws = DMatrix::<f64>::zeros(n_post, n_params);

    let d_s = (if data.re_mask_b1.iter().any(|&m| m) { 1 } else { 0 })
            + data.re_mask_deltas.iter().filter(|m| m.iter().any(|&v| v)).count()
            + data.re_mask_om.iter().filter(|m| m.iter().any(|&v| v)).count();
    let mut adapt_subj: Vec<HmcAdapt> = (0..data.n_subjects)
        .map(|_| HmcAdapt::new(d_s.max(1), step_om_init, target_accept))
        .collect();
    let mut adapt_om: Vec<HmcAdapt> = (0..data.n_breakpoints)
        .map(|k| {
            let p_fixed = data.x_om[k].ncols() - data.re_mask_om[k].iter().filter(|&&m| m).count();
            HmcAdapt::new(p_fixed, step_om_init, target_accept)
        })
        .collect();
    let mut adapt_rho: Vec<HmcAdapt> = (0..data.n_breakpoints)
        .map(|k| HmcAdapt::new(data.x_rho[k].ncols(), step_rho_init, target_accept))
        .collect();

    let report_every = (n_iter / 10).max(1);

    for iter in 0..n_iter {
        if verbose && iter % report_every == 0 {
            progress_fn(chain_id, n_chains, iter, n_iter, iter < n_warmup);
        }

        sample_linear_coefs(data, priors, &mut state, &mut rng);
        if data.n_groups_b0 > 0 { sample_random_effects(data, priors, &mut state, &mut rng); }
        // Step along the (b0, u) ridge. Only b0 + u_j is identified within a
        // group, so the two conjugate blocks above cannot traverse it on their
        // own; this move is likelihood-invariant and accepted outright.
        if data.n_groups_b0 > 0 { b0_translation_step(data, priors, &mut state, &mut rng); }

        joint_subject_nuts(data, priors, &mut state, &mut adapt_subj, &mut rng,
                           nc_b1, nc_deltas, nc_om);

        for k in 0..data.n_breakpoints {
            let cache_k = LinearCache::build(&state, data);
            let nc_flag = k < nc_om.len() && nc_om[k].iter().any(|&x| x);
            hmc_step_om(data, priors, &mut state, k, &cache_k, &mut adapt_om[k], &mut rng,
                        nc_flag, true);
            omega_translation_step(data, priors, &mut state, k, &mut rng);
        }

        // Redraw the linear block at the new omega before rho conditions on it.
        // The collapsed omega update deliberately leaves beta untouched, so
        // without this rho would be fitted against coefficients belonging to the
        // previous change-point. That mismatch is what drives the chain onto the
        // degenerate rho -> 0 ridge, where the transition weight flattens to a
        // constant and delta trades off against the baseline slope.
        sample_linear_coefs(data, priors, &mut state, &mut rng);

        let cache = LinearCache::build(&state, data);
        for k in 0..data.n_breakpoints {
            hmc_step_rho(data, priors, &mut state, k, &cache, &mut adapt_rho[k], &mut rng);
        }

        sample_gamma(data, priors, ss, &mut state, &mut rng);
        sample_linear_coefs(data, priors, &mut state, &mut rng);
        if learn_pi { sample_pi(ss, &mut state, &mut rng); }
        sample_sigma(data, priors, &mut state, &mut rng);
        if data.n_groups_b0 > 0 { sample_sigma_u(priors, &mut state, &mut rng); }
        sample_sigma_re_om(data, priors, &mut state, &mut rng);
        sample_sigma_re_b1(data, priors, &mut state, &mut rng);
        sample_sigma_re_deltas(data, priors, &mut state, &mut rng);

        if iter < n_warmup {
            for k in 0..data.n_breakpoints {
                let fixed_om = DVector::from_iterator(
                    adapt_om[k].p,
                    state.beta_om[k].iter().enumerate()
                        .filter(|&(j, _)| !data.re_mask_om[k][j])
                        .map(|(_, &v)| v)
                );
                adapt_om[k].observe(&fixed_om);
                if (iter + 1) % 500 == 0 { adapt_om[k].refresh_mass_matrix(); }
                adapt_rho[k].observe(&state.beta_rho[k]);
                if (iter + 1) % 500 == 0 { adapt_rho[k].refresh_mass_matrix(); }
            }
        } else if iter == n_warmup {
            for k in 0..data.n_breakpoints {
                adapt_om[k].freeze();
                adapt_rho[k].freeze();
            }
            for s in 0..data.n_subjects { adapt_subj[s].freeze(); }
        }

        if iter >= n_warmup {
            let row = iter - n_warmup;
            let draw = state.to_vec(true, learn_pi, true);
            for (col, &val) in draw.iter().enumerate() { draws[(row, col)] = val; }
        }
    }
    let div_subj = adapt_subj.iter().map(|h| h.n_divergent).sum::<usize>();
    let div_om   = adapt_om.iter().map(|h| h.n_divergent).sum::<usize>();
    let div_rho  = adapt_rho.iter().map(|h| h.n_divergent).sum::<usize>();
    let n_div = div_subj + div_om + div_rho;
    // [total, subj, om, rho]
    (draws, [n_div, div_subj, div_om, div_rho])
}


pub fn init_state_re(data: &ModelData, priors: &Priors, rng: &mut StdRng) -> State {
    let jitter = Normal::new(0.0, 0.01).unwrap();
    let beta_b0 = DVector::from_iterator(data.x_b0.ncols(), (0..data.x_b0.ncols()).map(|i| {
        let m = priors.b0_mean[i];
        if priors.b0_sd[i] > 0.0 { m + jitter.sample(rng) } else { m }
    }));
    let u_b0 = DVector::zeros(data.n_groups_b0);
    let beta_b1 = DVector::from_iterator(data.x_b1.ncols(), (0..data.x_b1.ncols()).map(|i| {
        let m = priors.b1_mean[i];
        if priors.b1_sd[i] > 0.0 { m + jitter.sample(rng) } else { m }
    }));
    let mut beta_deltas = Vec::new();
    let mut beta_om = Vec::new();
    let mut beta_rho = Vec::new();
    let mut gamma_deltas = Vec::new();

    for k in 0..data.n_breakpoints {
        beta_deltas.push(DVector::from_iterator(data.x_deltas[k].ncols(), (0..data.x_deltas[k].ncols()).map(|i| {
            let m = priors.delta_mean[k][i];
            if priors.delta_sd[k][i] > 0.0 { m + jitter.sample(rng) } else { m }
        })));
        beta_om.push(DVector::from_iterator(data.x_om[k].ncols(), (0..data.x_om[k].ncols()).map(|j| {
            if data.re_mask_om[k][j] {
                // RE columns are deviations from the intercept: prior is N(0, sigma_re^2).
                // Initialise near 0, not at the user's omega prior mean.
                jitter.sample(rng)
            } else {
                let m = priors.om_mean[k][j];
                if priors.om_sd[k][j] > 0.0 { m + jitter.sample(rng) } else { m }
            }
        })));
        beta_rho.push(DVector::from_iterator(data.x_rho[k].ncols(), (0..data.x_rho[k].ncols()).map(|i| {
            let m = priors.rho_mean[k][i];
            if priors.rho_sd[k][i] > 0.0 { m + jitter.sample(rng) } else { m }
        })));
        gamma_deltas.push(vec![true; data.x_deltas[k].ncols()]);
    }

    State {
        beta_b0, u_b0, beta_b1, beta_deltas, beta_om, beta_rho,
        sigma: 1.0, sigma_u: 1.0,
        gamma_b1: vec![true; data.x_b1.ncols()],
        gamma_deltas, pi: 0.5,
        sigma_re_om:     vec![1.0; data.n_breakpoints],
        sigma_re_b1:     1.0,
        sigma_re_deltas: vec![1.0; data.n_breakpoints],
    }
}

// ---------------------------------------------------------------------------
// Unit tests for the collapsed change-point energy (random-effect version)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod collapsed_re_tests {
    use super::*;

    /// Two subjects, six observations each, with random effects on omega, b1 and
    /// delta so that every term of dy~/d(omega) is exercised: the b1 random
    /// slope (active only for breakpoint 0) and the delta random effect (which
    /// enters through the saturating transition factor).
    fn toy_re() -> (ModelData, Priors, State) {
        let n = 12usize;
        let subj: Vec<usize> = (0..n).map(|i| i / 6).collect();
        let tau: Vec<f64> = (0..n).map(|i| -4.0 + 14.0 * ((i % 6) as f64) / 5.0).collect();
        let y: Vec<f64> = tau.iter().enumerate().map(|(i, &t)| {
            let d = t - 2.0;
            3.0 + 0.2 * d + 1.3 * d * sigmoid(3.0 * d) + 0.1 * ((i * 41 % 13) as f64 - 6.0)
        }).collect();

        // Column layout for x_b1, x_deltas, x_om: [intercept, subject 0, subject 1]
        let mut design = DMatrix::<f64>::zeros(n, 3);
        for i in 0..n {
            design[(i, 0)] = 1.0;
            design[(i, 1 + subj[i])] = 1.0;
        }

        let data = ModelData {
            y: DVector::from_vec(y),
            tau: DVector::from_vec(tau),
            x_b0: DMatrix::from_element(n, 1, 1.0),
            x_b1: design.clone(),
            x_deltas: vec![design.clone()],
            x_om: vec![design.clone()],
            x_rho: vec![DMatrix::from_element(n, 1, 1.0)],
            group_b0: vec![],
            n_groups_b0: 0,
            n,
            n_breakpoints: 1,
            re_mask_om: vec![vec![false, true, true]],
            re_mask_b1: vec![false, true, true],
            re_mask_deltas: vec![vec![false, true, true]],
            group_re: subj.iter().map(|&s| s as i32).collect(),
            n_subjects: 2,
        };

        let inf3 = || vec![f64::INFINITY; 3];
        let ninf3 = || vec![f64::NEG_INFINITY; 3];
        let priors = Priors {
            b0_mean: vec![3.0], b0_sd: vec![2.0],
            b0_lb: vec![f64::NEG_INFINITY], b0_ub: vec![f64::INFINITY],
            b1_mean: vec![0.0; 3], b1_sd: vec![0.5; 3],
            b1_lb: ninf3(), b1_ub: inf3(),
            delta_mean: vec![vec![0.0; 3]], delta_sd: vec![vec![1.5; 3]],
            delta_lb: vec![ninf3()], delta_ub: vec![inf3()],
            om_mean: vec![vec![2.0, 0.0, 0.0]], om_sd: vec![vec![1.0, 1.0, 1.0]],
            om_lb: vec![vec![-9.0, f64::NEG_INFINITY, f64::NEG_INFINITY]],
            om_ub: vec![vec![12.0, f64::INFINITY, f64::INFINITY]],
            rho_mean: vec![vec![3.0]], rho_sd: vec![vec![0.0]],
            rho_lb: vec![vec![0.0]], rho_ub: vec![vec![f64::INFINITY]],
            sigma_shape: 2.0, sigma_scale: 1.0,
            sigma_u_shape: 1.0, sigma_u_scale: 1.0,
            sigma_re_om_shape: 1.0, sigma_re_om_scale: 1.0,
            sigma_re_b1_shape: 1.0, sigma_re_b1_scale: 1.0,
            sigma_re_deltas_shape: 1.0, sigma_re_deltas_scale: 1.0,
            p_b0: 1, p_b1: 3, p_deltas: vec![3], p_om: vec![3], p_rho: vec![1],
        };

        let state = State {
            beta_b0: DVector::from_vec(vec![3.0]),
            u_b0: DVector::zeros(0),
            // non-zero random effects, so their gradient terms are not silently zero
            beta_b1: DVector::from_vec(vec![0.2, 0.15, -0.1]),
            beta_deltas: vec![DVector::from_vec(vec![1.3, 0.25, -0.3])],
            beta_om: vec![DVector::from_vec(vec![2.0, 0.4, -0.5])],
            beta_rho: vec![DVector::from_vec(vec![3.0])],
            sigma: 0.4, sigma_u: 1.0,
            gamma_b1: vec![true; 3],
            gamma_deltas: vec![vec![true; 3]],
            pi: 0.5,
            sigma_re_om: vec![0.8], sigma_re_b1: 0.5, sigma_re_deltas: vec![0.7],
        };
        (data, priors, state)
    }

    #[test]
    fn collapsed_re_gradient_matches_finite_differences() {
        let (data, priors, state) = toy_re();
        let n = data.n;
        let sigma2 = state.sigma * state.sigma;
        let y_base = data.y.clone();

        for &shift in &[-1.0_f64, 0.0, 1.5] {
            let om_k = DVector::from_iterator(
                n, (0..n).map(|i| 2.0 + shift + 0.3 * ((i % 3) as f64)));
            let (_, grad) = collapsed_omega_energy_re(
                &data, &priors, &state, 0, &om_k, &y_base, sigma2);

            let h = 1e-6;
            for i in 0..n {
                let mut up = om_k.clone(); up[i] += h;
                let mut dn = om_k.clone(); dn[i] -= h;
                let (u_up, _) = collapsed_omega_energy_re(&data, &priors, &state, 0, &up, &y_base, sigma2);
                let (u_dn, _) = collapsed_omega_energy_re(&data, &priors, &state, 0, &dn, &y_base, sigma2);
                let fd = (u_up - u_dn) / (2.0 * h);
                let scale = fd.abs().max(grad[i].abs()).max(1.0);
                assert!((fd - grad[i]).abs() / scale < 1e-5,
                        "shift={shift} obs {i}: analytic {} vs finite difference {}", grad[i], fd);
            }
        }
    }

    /// With every random effect switched off, the RE energy must reduce to the
    /// plain marginal likelihood: same value up to an omega-free constant, and
    /// the same gradient. This pins the two implementations against each other.
    #[test]
    fn collapsed_re_reduces_to_non_re_when_random_effects_vanish() {
        let (data, priors, mut state) = toy_re();
        let n = data.n;
        state.beta_b1 = DVector::from_vec(vec![0.2, 0.0, 0.0]);
        state.beta_deltas = vec![DVector::from_vec(vec![1.3, 0.0, 0.0])];
        let sigma2 = state.sigma * state.sigma;
        let y_base = data.y.clone();

        // dy~/d(omega) is now identically zero, so the gradient must match the
        // finite differences of an energy whose response no longer moves.
        let om_k = DVector::from_element(n, 2.4);
        let (_, grad) = collapsed_omega_energy_re(&data, &priors, &state, 0, &om_k, &y_base, sigma2);
        let h = 1e-6;
        for i in 0..n {
            let mut up = om_k.clone(); up[i] += h;
            let mut dn = om_k.clone(); dn[i] -= h;
            let (u_up, _) = collapsed_omega_energy_re(&data, &priors, &state, 0, &up, &y_base, sigma2);
            let (u_dn, _) = collapsed_omega_energy_re(&data, &priors, &state, 0, &dn, &y_base, sigma2);
            let fd = (u_up - u_dn) / (2.0 * h);
            let scale = fd.abs().max(grad[i].abs()).max(1.0);
            assert!((fd - grad[i]).abs() / scale < 1e-5, "obs {i}");
        }
    }
}

#[cfg(test)]
mod nuts_tests {
    use super::*;

    /// NUTS must reproduce a target whose moments are known exactly.
    ///
    /// This is the regression guard for the missing `s' = 1` guard in the
    /// doubling loop: accepting proposals from a subtree that ended on a U-turn
    /// pulls draws outward, inflating the sampled variance above one. The
    /// energy here is a standard normal, so any such deviation shows up
    /// directly rather than being absorbed by a nuisance parameter.
    fn run_standard_normal(dim: usize, n_draws: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut adapt = HmcAdapt::new(dim, 0.4, 0.8);
        adapt.adapting = false;                       // fixed step, no dual averaging
        let lb = vec![f64::NEG_INFINITY; dim];
        let ub = vec![f64::INFINITY; dim];

        let mut q = DVector::<f64>::zeros(dim);
        let mut sum = vec![0.0f64; dim];
        let mut sumsq = vec![0.0f64; dim];
        for _ in 0..n_draws {
            let energy = |x: &DVector<f64>| -> (f64, DVector<f64>) {
                (0.5 * x.dot(x), x.clone())
            };
            let (q_new, _) = nuts_sample(&q, energy, &mut adapt, &mut rng, &lb, &ub);
            q = q_new;
            for i in 0..dim { sum[i] += q[i]; sumsq[i] += q[i] * q[i]; }
        }
        let n = n_draws as f64;
        let means: Vec<f64> = (0..dim).map(|i| sum[i] / n).collect();
        let vars: Vec<f64> = (0..dim).map(|i| sumsq[i] / n - means[i] * means[i]).collect();
        (means, vars)
    }

    #[test]
    fn nuts_recovers_standard_normal_moments() {
        let dim = 2;
        let (means, vars) = run_standard_normal(dim, 40_000, 20260811);
        for i in 0..dim {
            assert!(means[i].abs() < 0.05, "dim {i}: mean {} should be 0", means[i]);
            assert!((vars[i] - 1.0).abs() < 0.05,
                    "dim {i}: variance {} should be 1", vars[i]);
        }
    }
}
