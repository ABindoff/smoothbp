use nalgebra::{DMatrix, DVector};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand::Rng;
use rand_distr::{Normal, Gamma, Distribution};

use crate::model::{ModelData, Priors, State, SpikeSlabConfig, log_truncated_normal_prior, sigmoid};

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

struct HmcAdapt {
    p: usize,
    l_min: usize,
    l_max: usize,
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
    fn new(p: usize, init_epsilon: f64, target_accept: f64, l_min: usize, l_max: usize) -> Self {
        HmcAdapt {
            p, l_min, l_max, epsilon: init_epsilon, target_accept,
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

    fn sample_l(&self, rng: &mut StdRng) -> usize {
        if self.l_min == self.l_max { self.l_min } else { rng.gen_range(self.l_min..=self.l_max) }
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

fn sample_gamma(data: &ModelData, _priors: &Priors, ss: &SpikeSlabConfig, state: &mut State, cache: &LinearCache, rng: &mut StdRng) {
    let mu_full = state.means(data);
    let sigma2 = state.sigma * state.sigma;
    let pi = state.pi;

    // Helper to update one gamma
    let mut update_gamma = |mu_without: &DVector<f64>, x_col: &DVector<f64>, beta: f64, g: &mut bool| {
        let mu1 = mu_without + x_col * beta;
        let r0 = &data.y - mu_without;
        let r1 = &data.y - &mu1;
        let log_p1 = -0.5 * r1.dot(&r1) / sigma2 + pi.ln();
        let log_p0 = -0.5 * r0.dot(&r0) / sigma2 + (1.0 - pi).ln();
        let max_lp = f64::max(log_p1, log_p0);
        let p1 = (log_p1 - max_lp).exp();
        let p0 = (log_p0 - max_lp).exp();
        *g = rng.gen_bool(p1 / (p1 + p0));
    };

    let mut current_mu = mu_full;

    // Gamma b1
    for j in 0..state.gamma_b1.len() {
        if !ss.b1_spike_mask[j] { continue; }
        
        let x_col = &data.x_b1.column(j);
        let mut x_eff = x_col.clone_owned();
        if data.n_breakpoints > 0 {
            let om1 = state.omega_vec(0, &data.x_om[0]);
            for i in 0..data.n { x_eff[i] *= data.tau[i] - om1[i]; }
        } else {
            for i in 0..data.n { x_eff[i] *= data.tau[i]; }
        }
        
        let beta_j = state.beta_b1[j];
        if state.gamma_b1[j] {
            let mu_without = &current_mu - &x_eff * beta_j;
            update_gamma(&mu_without, &x_eff, beta_j, &mut state.gamma_b1[j]);
            if !state.gamma_b1[j] { current_mu = mu_without; }
        } else {
            update_gamma(&current_mu, &x_eff, beta_j, &mut state.gamma_b1[j]);
            if state.gamma_b1[j] { current_mu = &current_mu + &x_eff * beta_j; }
        }
    }

    // Gamma deltas
    for k in 0..data.n_breakpoints {
        let om = state.omega_vec(k, &data.x_om[k]);
        let rho = state.rho_vec(k, &data.x_rho[k]);
        for j in 0..state.gamma_deltas[k].len() {
            if !ss.delta_spike_mask[k][j] { continue; }
            
            let x_col = &data.x_deltas[k].column(j);
            let mut x_eff = x_col.clone_owned();
            for i in 0..data.n {
                let di = data.tau[i] - om[i];
                let si = sigmoid(di * rho[i]);
                x_eff[i] *= di * si;
            }
            let _ = &cache; // suppress unused warning
            
            let beta_kj = state.beta_deltas[k][j];
            if state.gamma_deltas[k][j] {
                let mu_without = &current_mu - &x_eff * beta_kj;
                update_gamma(&mu_without, &x_eff, beta_kj, &mut state.gamma_deltas[k][j]);
                if !state.gamma_deltas[k][j] { current_mu = mu_without; }
            } else {
                update_gamma(&current_mu, &x_eff, beta_kj, &mut state.gamma_deltas[k][j]);
                if state.gamma_deltas[k][j] { current_mu = &current_mu + &x_eff * beta_kj; }
            }
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
    // Apply gamma_b1 ( Kuo-Mallick )
    for j in 0..p_b1 {
        if !state.gamma_b1[j] {
            let mut col = b1_design.column_mut(j);
            col.fill(0.0);
        }
    }
    x_full.view_mut((0, p_b0), (n, p_b1)).copy_from(&b1_design);
    for j in 0..p_b1 {
        prec_prior[p_b0 + j] = 1.0 / (priors.b1_sd[j] * priors.b1_sd[j]);
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
        // Apply gamma_deltas
        for j in 0..pk {
            if !state.gamma_deltas[k][j] {
                let mut col = d_design.column_mut(j);
                col.fill(0.0);
            }
        }
        x_full.view_mut((0, offset), (n, pk)).copy_from(&d_design);
        for j in 0..pk {
            prec_prior[offset + j] = 1.0 / (priors.delta_sd[k][j] * priors.delta_sd[k][j]);
            mu_prior[offset + j] = priors.delta_mean[k][j];
        }
        offset += pk;
    }

    // Sufficient statistics
    let mut y_tilde = &data.y - DVector::zeros(n);
    if data.n_groups_b0 > 0 {
        for i in 0..n {
            let g = data.group_b0[i];
            if g >= 0 { y_tilde[i] -= state.u_b0[g as usize]; }
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

    // Export back to state
    state.beta_b0.copy_from(&theta.rows(0, p_b0));
    state.beta_b1.copy_from(&theta.rows(p_b0, p_b1));
    offset = p_b0 + p_b1;
    for k in 0..data.n_breakpoints {
        let pk = data.x_deltas[k].ncols();
        state.beta_deltas[k].copy_from(&theta.rows(offset, pk));
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
    nc: bool,   // use non-centred reparameterisation for RE columns at this breakpoint
) {
    let p = adapt.p;

    // If all prior SDs are 0, this parameter block is fixed.
    let mut all_fixed = true;
    for j in 0..p {
        if !data.re_mask_om[k][j] && priors.om_sd[k][j] > 0.0 { all_fixed = false; break; }
        if data.re_mask_om[k][j] { all_fixed = false; break; }
    }
    if all_fixed { return; }

    let sigma    = state.sigma;
    let sigma_re = state.sigma_re_om[k];
    let mu_base  = cache.mu_without_segment(data, state, k);
    let is_om1   = k == 0 && data.n_breakpoints > 0;

    // Non-centred only when nc=true and sigma_re is non-degenerate.
    // In NC mode, HMC operates on z[j] = beta_om[j] / sigma_re for RE columns.
    let do_nc = nc && sigma_re > 1e-10;

    // Initial q in HMC space
    let q0: DVector<f64> = if do_nc {
        let mut z = state.beta_om[k].clone();
        for j in 0..p { if data.re_mask_om[k][j] { z[j] /= sigma_re; } }
        z
    } else {
        state.beta_om[k].clone()
    };

    // Bounds in HMC space (z-space for NC RE columns)
    let lb: Vec<f64> = (0..p).map(|j| {
        if do_nc && data.re_mask_om[k][j] { priors.om_lb[k][j] / sigma_re }
        else { priors.om_lb[k][j] }
    }).collect();
    let ub: Vec<f64> = (0..p).map(|j| {
        if do_nc && data.re_mask_om[k][j] { priors.om_ub[k][j] / sigma_re }
        else { priors.om_ub[k][j] }
    }).collect();

    let energy_fn = |q: &DVector<f64>| -> (f64, DVector<f64>) {
        // Convert q (HMC space) → beta_om for likelihood.
        // For NC RE columns: beta_om[j] = z[j] * sigma_re.
        let beta_om: DVector<f64> = if do_nc {
            let mut b = q.clone();
            for j in 0..p { if data.re_mask_om[k][j] { b[j] = q[j] * sigma_re; } }
            b
        } else {
            q.clone()
        };

        let om_k   = &data.x_om[k] * &beta_om;
        let rho_k  = state.rho_vec(k, &data.x_rho[k]);
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

        let r   = &data.y - &mu;
        let ll  = -0.5 * r.dot(&r) / (sigma * sigma);

        // Log-prior in q-space and bounds check
        let mut lp = 0.0;
        let log_sqrt2pi = 0.5 * std::f64::consts::TAU.ln();
        for j in 0..p {
            let v = q[j];
            if v < lb[j] || v > ub[j] { return (f64::INFINITY, DVector::<f64>::zeros(p)); }
            if do_nc && data.re_mask_om[k][j] {
                // z ~ N(0,1): unnormalised log-prior = -0.5 z^2
                lp -= 0.5 * v * v;
            } else {
                let sd = if data.re_mask_om[k][j] { sigma_re } else { priors.om_sd[k][j] };
                let z = (v - priors.om_mean[k][j]) / sd;
                lp -= 0.5 * z * z + sd.ln() + log_sqrt2pi;
            }
        }

        // Likelihood gradient w.r.t. beta_om
        let inv_s2 = 1.0 / (sigma * sigma);
        let mut grad_beta = DVector::<f64>::zeros(p);
        for i in 0..data.n {
            let di  = data.tau[i] - om_k[i];
            let si  = sigmoid(di * rho_k[i]);
            let ri  = rho_k[i];
            let bi  = delta_k[i];
            let mut dmu_dom = -(bi * si + di * ri * si * (1.0 - si) * bi);
            if is_om1 { dmu_dom -= cache.b1_vals[i]; }
            let factor = r[i] * inv_s2 * dmu_dom;
            for j in 0..p { grad_beta[j] -= factor * data.x_om[k][(i, j)]; }
        }

        // Transform gradient to q-space and add prior gradient
        let mut grad = DVector::<f64>::zeros(p);
        for j in 0..p {
            if do_nc && data.re_mask_om[k][j] {
                // Chain rule: dU_lik/dz = sigma_re * dU_lik/d(beta_om)
                grad[j] = sigma_re * grad_beta[j];
                // N(0,1) prior: d(-log N(0,1))/dz = z
                grad[j] += q[j];
            } else {
                grad[j] = grad_beta[j];
                let sd = if data.re_mask_om[k][j] { sigma_re } else { priors.om_sd[k][j] };
                grad[j] += (q[j] - priors.om_mean[k][j]) / (sd * sd);
            }
        }

        (-ll - lp, grad)
    };

    let (q_new, accept) = hmc_sample(&q0, energy_fn, adapt, rng, &lb, &ub);

    // Convert q_new back to beta_om space
    if do_nc {
        for j in 0..p {
            state.beta_om[k][j] = if data.re_mask_om[k][j] { q_new[j] * sigma_re }
                                   else { q_new[j] };
        }
    } else {
        state.beta_om[k] = q_new;
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

    let (q_new, accept) = hmc_sample(&state.beta_rho[k], energy_fn, adapt, rng, &priors.rho_lb[k], &priors.rho_ub[k]);
    state.beta_rho[k] = q_new;
    adapt.update_epsilon(accept);
}

fn hmc_sample<F>(
    q0: &DVector<f64>,
    mut energy_fn: F,
    adapt: &mut HmcAdapt,
    rng: &mut StdRng,
    lb: &[f64],
    ub: &[f64],
) -> (DVector<f64>, f64)
where F: FnMut(&DVector<f64>) -> (f64, DVector<f64>)
{
    let p = adapt.p;
    let normal = Normal::new(0.0, 1.0).unwrap();
    let p0 = DVector::<f64>::from_iterator(p, (0..p).map(|i| normal.sample(rng) / adapt.inv_mass[i].sqrt()));
    let kinetic0: f64 = (0..p).map(|i| p0[i]*p0[i]*adapt.inv_mass[i]).sum::<f64>() * 0.5;
    let (u0, mut grad) = energy_fn(q0);
    let h0 = u0 + kinetic0;

    let mut q = q0.clone();
    let mut mom = p0.clone();
    let l = adapt.sample_l(rng);
    let eps = adapt.epsilon;

    for _ in 0..l {
        for i in 0..p { mom[i] -= 0.5 * eps * grad[i]; }
        for i in 0..p {
            q[i] += eps * mom[i] * adapt.inv_mass[i];
            // Reflections
            for _ in 0..MAX_REFLECTIONS {
                if q[i] < lb[i] { q[i] = 2.0 * lb[i] - q[i]; mom[i] = -mom[i]; }
                else if q[i] > ub[i] { q[i] = 2.0 * ub[i] - q[i]; mom[i] = -mom[i]; }
                else { break; }
            }
        }
        let (_, g_new) = energy_fn(&q);
        grad = g_new;
        for i in 0..p { mom[i] -= 0.5 * eps * grad[i]; }
    }

    let (u1, _) = energy_fn(&q);
    let kinetic1: f64 = (0..p).map(|i| mom[i]*mom[i]*adapt.inv_mass[i]).sum::<f64>() * 0.5;
    let h1 = u1 + kinetic1;

    let dh = h1 - h0;
    adapt.record_energy_error(dh);
    let accept_prob = (-dh).exp().min(1.0);
    if rng.gen_bool(accept_prob) { (q, accept_prob) } else { (q0.clone(), accept_prob) }
}

// ---------------------------------------------------------------------------
// Main Chain Loops
// ---------------------------------------------------------------------------

pub fn run_chain_re(
    data: &ModelData, priors: &Priors, n_iter: usize, n_warmup: usize,
    step_om_init: f64, step_rho_init: f64, target_accept: f64,
    seed: u64, verbose: bool, chain_id: usize, n_chains: usize,
    nc_om: &[bool],   // per-breakpoint: true = NC reparameterisation for omega RE
    progress_fn: &dyn Fn(usize, usize, usize, usize, bool),
) -> (DMatrix<f64>, usize) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut state = init_state_re(data, priors, &mut rng);
    let n_post = n_iter - n_warmup;
    let n_params = state.n_params(false, false, true);
    let mut draws = DMatrix::<f64>::zeros(n_post, n_params);

    let mut adapt_om: Vec<HmcAdapt> = (0..data.n_breakpoints).map(|k| HmcAdapt::new(data.x_om[k].ncols(), step_om_init, target_accept, 5, 15)).collect();
    let mut adapt_rho: Vec<HmcAdapt> = (0..data.n_breakpoints).map(|k| HmcAdapt::new(data.x_rho[k].ncols(), step_rho_init, target_accept, 5, 15)).collect();

    let report_every = (n_iter / 10).max(1);

    for iter in 0..n_iter {
        if verbose && iter % report_every == 0 { progress_fn(chain_id, n_chains, iter, n_iter, iter < n_warmup); }

        sample_linear_coefs(data, priors, &mut state, &mut rng);
        if data.n_groups_b0 > 0 { sample_random_effects(data, priors, &mut state, &mut rng); }

        let cache = LinearCache::build(&state, data);
        for k in 0..data.n_breakpoints {
            let nc_k = nc_om.get(k).copied().unwrap_or(false);
            hmc_step_om(data, priors, &mut state, k, &cache, &mut adapt_om[k], &mut rng, nc_k);
            hmc_step_rho(data, priors, &mut state, k, &cache, &mut adapt_rho[k], &mut rng);
        }

        sample_linear_coefs(data, priors, &mut state, &mut rng);
        sample_sigma(data, priors, &mut state, &mut rng);
        if data.n_groups_b0 > 0 { sample_sigma_u(priors, &mut state, &mut rng); }

        // Hierarchical update for omega variances (uses beta_om, correct for both centred and NC)
        sample_sigma_re_om(data, priors, &mut state, &mut rng);

        if iter < n_warmup {
            for k in 0..data.n_breakpoints {
                // Observe in q-space: z-space for NC RE columns so mass matrix adapts correctly
                let nc_k = nc_om.get(k).copied().unwrap_or(false);
                let sigma_re = state.sigma_re_om[k];
                let q_obs = if nc_k && sigma_re > 1e-10 {
                    let mut z = state.beta_om[k].clone();
                    for j in 0..data.x_om[k].ncols() {
                        if data.re_mask_om[k][j] { z[j] /= sigma_re; }
                    }
                    z
                } else {
                    state.beta_om[k].clone()
                };
                adapt_om[k].observe(&q_obs);
                adapt_rho[k].observe(&state.beta_rho[k]);
                if (iter + 1) % 500 == 0 {
                    adapt_om[k].refresh_mass_matrix();
                    adapt_rho[k].refresh_mass_matrix();
                }
            }
        } else if iter == n_warmup {
            for k in 0..data.n_breakpoints {
                adapt_om[k].freeze();
                adapt_rho[k].freeze();
            }
        }

        if iter >= n_warmup {
            let row = iter - n_warmup;
            let draw = state.to_vec(false, false, true);
            for (col, &val) in draw.iter().enumerate() { draws[(row, col)] = val; }
        }
    }
    let n_div = adapt_om.iter().map(|h| h.n_divergent).sum::<usize>() + adapt_rho.iter().map(|h| h.n_divergent).sum::<usize>();
    (draws, n_div)
}
pub fn run_chain_re_ss(
    data: &ModelData, priors: &Priors, ss: &SpikeSlabConfig, n_iter: usize, n_warmup: usize,
    step_om_init: f64, step_rho_init: f64, target_accept: f64,
    seed: u64, verbose: bool, chain_id: usize, n_chains: usize,
    nc_om: &[bool],
    progress_fn: &dyn Fn(usize, usize, usize, usize, bool),
) -> (DMatrix<f64>, usize) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut state = init_state_re(data, priors, &mut rng);
    state.pi = ss.pi_init;
    let n_post = n_iter - n_warmup;
    let learn_pi = ss.beta_a > 0.0;
    let n_params = state.n_params(true, learn_pi, true);
    let mut draws = DMatrix::<f64>::zeros(n_post, n_params);

    let mut adapt_om: Vec<HmcAdapt> = (0..data.n_breakpoints).map(|k| HmcAdapt::new(data.x_om[k].ncols(), step_om_init, target_accept, 5, 15)).collect();
    let mut adapt_rho: Vec<HmcAdapt> = (0..data.n_breakpoints).map(|k| HmcAdapt::new(data.x_rho[k].ncols(), step_rho_init, target_accept, 5, 15)).collect();

    let report_every = (n_iter / 10).max(1);

    for iter in 0..n_iter {
        if verbose && iter % report_every == 0 { progress_fn(chain_id, n_chains, iter, n_iter, iter < n_warmup); }

        sample_linear_coefs(data, priors, &mut state, &mut rng);
        if data.n_groups_b0 > 0 { sample_random_effects(data, priors, &mut state, &mut rng); }

        let cache = LinearCache::build(&state, data);
        for k in 0..data.n_breakpoints {
            let nc_k = nc_om.get(k).copied().unwrap_or(false);
            hmc_step_om(data, priors, &mut state, k, &cache, &mut adapt_om[k], &mut rng, nc_k);
            hmc_step_rho(data, priors, &mut state, k, &cache, &mut adapt_rho[k], &mut rng);
        }

        sample_gamma(data, priors, ss, &mut state, &cache, &mut rng);
        sample_linear_coefs(data, priors, &mut state, &mut rng);
        if learn_pi { sample_pi(ss, &mut state, &mut rng); }
        sample_sigma(data, priors, &mut state, &mut rng);
        if data.n_groups_b0 > 0 { sample_sigma_u(priors, &mut state, &mut rng); }

        sample_sigma_re_om(data, priors, &mut state, &mut rng);

        if iter < n_warmup {
            for k in 0..data.n_breakpoints {
                let nc_k = nc_om.get(k).copied().unwrap_or(false);
                let sigma_re = state.sigma_re_om[k];
                let q_obs = if nc_k && sigma_re > 1e-10 {
                    let mut z = state.beta_om[k].clone();
                    for j in 0..data.x_om[k].ncols() {
                        if data.re_mask_om[k][j] { z[j] /= sigma_re; }
                    }
                    z
                } else {
                    state.beta_om[k].clone()
                };
                adapt_om[k].observe(&q_obs);
                adapt_rho[k].observe(&state.beta_rho[k]);
                if (iter + 1) % 500 == 0 {
                    adapt_om[k].refresh_mass_matrix();
                    adapt_rho[k].refresh_mass_matrix();
                }
            }
        } else if iter == n_warmup {
            for k in 0..data.n_breakpoints {
                adapt_om[k].freeze();
                adapt_rho[k].freeze();
            }
        }

        if iter >= n_warmup {
            let row = iter - n_warmup;
            let draw = state.to_vec(true, learn_pi, true);
            for (col, &val) in draw.iter().enumerate() { draws[(row, col)] = val; }
        }
    }
    let n_div = adapt_om.iter().map(|h| h.n_divergent).sum::<usize>() + adapt_rho.iter().map(|h| h.n_divergent).sum::<usize>();
    (draws, n_div)
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
        beta_om.push(DVector::from_iterator(data.x_om[k].ncols(), (0..data.x_om[k].ncols()).map(|i| {
            let m = priors.om_mean[k][i];
            if priors.om_sd[k][i] > 0.0 { m + jitter.sample(rng) } else { m }
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
        sigma_re_om: vec![1.0; data.n_breakpoints],
    }
}
