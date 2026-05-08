use nalgebra::{DMatrix, DVector};

// ---------------------------------------------------------------------------
// Data passed in from R
// ---------------------------------------------------------------------------

pub struct ModelData {
    pub y: DVector<f64>,
    pub tau: DVector<f64>,
    pub x_b0: DMatrix<f64>,
    pub x_b1: DMatrix<f64>,
    /// List of design matrices for slope changes at each breakpoint
    pub x_deltas: Vec<DMatrix<f64>>,
    /// List of design matrices for breakpoint locations
    pub x_om: Vec<DMatrix<f64>>,
    /// List of design matrices for transition sharpness
    pub x_rho: Vec<DMatrix<f64>>,
    /// 0-based group indices for b0 random intercept; -1 if observation has no RE
    pub group_b0: Vec<i32>,
    pub n_groups_b0: usize,
    pub n: usize,
    pub n_breakpoints: usize,
    /// Indicates if a coefficient in x_om[k] is a random effect (hierarchical)
    pub re_mask_om: Vec<Vec<bool>>,
}

// ---------------------------------------------------------------------------
// Prior hyperparameters
// ---------------------------------------------------------------------------

/// Priors for all regression coefficients.
/// Organized by parameter group.
pub struct Priors {
    pub b0_mean: Vec<f64>,
    pub b0_sd: Vec<f64>,
    pub b0_lb: Vec<f64>,
    pub b0_ub: Vec<f64>,

    pub b1_mean: Vec<f64>,
    pub b1_sd: Vec<f64>,
    pub b1_lb: Vec<f64>,
    pub b1_ub: Vec<f64>,

    pub delta_mean: Vec<Vec<f64>>,
    pub delta_sd: Vec<Vec<f64>>,
    pub delta_lb: Vec<Vec<f64>>,
    pub delta_ub: Vec<Vec<f64>>,

    pub om_mean: Vec<Vec<f64>>,
    pub om_sd: Vec<Vec<f64>>,
    pub om_lb: Vec<Vec<f64>>,
    pub om_ub: Vec<Vec<f64>>,

    pub rho_mean: Vec<Vec<f64>>,
    pub rho_sd: Vec<Vec<f64>>,
    pub rho_lb: Vec<Vec<f64>>,
    pub rho_ub: Vec<Vec<f64>>,

    pub sigma_shape: f64,
    pub sigma_scale: f64,
    pub sigma_u_shape: f64,
    pub sigma_u_scale: f64,

    pub sigma_re_om_shape: f64,
    pub sigma_re_om_scale: f64,

    pub p_b0: usize,
    pub p_b1: usize,
    pub p_deltas: Vec<usize>,
    pub p_om: Vec<usize>,
    pub p_rho: Vec<usize>,
}

// ---------------------------------------------------------------------------
// Spike-and-slab configuration
// ---------------------------------------------------------------------------

pub struct SpikeSlabConfig {
    /// Whether b1 is eligible for spike-and-slab
    pub b1_spike_mask: Vec<bool>,
    /// Spike-and-slab for each breakpoint's delta coefficients
    pub delta_spike_mask: Vec<Vec<bool>>,

    pub pi_init: f64,

    /// Beta hyperprior shape parameters for pi (if learn_pi > 0)
    pub beta_a: f64,
    pub beta_b: f64,
}

// ---------------------------------------------------------------------------
// Sampler state
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct State {
    pub beta_b0: DVector<f64>,
    pub u_b0: DVector<f64>,
    pub beta_b1: DVector<f64>,
    pub beta_deltas: Vec<DVector<f64>>,
    pub beta_om: Vec<DVector<f64>>,
    pub beta_rho: Vec<DVector<f64>>,
    pub sigma: f64,
    pub sigma_u: f64,
    /// Inclusion indicators for b1
    pub gamma_b1: Vec<bool>,
    /// Inclusion indicators for each breakpoint's deltas
    pub gamma_deltas: Vec<Vec<bool>>,
    /// Current inclusion probability
    pub pi: f64,
    /// Learned standard deviation for omega random effects at each breakpoint
    pub sigma_re_om: Vec<f64>,
}

impl State {
    pub fn n_params(&self, include_gammas: bool, learn_pi: bool, hierarchical: bool) -> usize {
        let mut n = self.beta_b0.len() + self.u_b0.len() + self.beta_b1.len() + 2;
        for i in 0..self.beta_deltas.len() {
            n += self.beta_deltas[i].len();
            n += self.beta_om[i].len();
            n += self.beta_rho[i].len();
        }
        if include_gammas {
            // Inclusion indicators
            n += self.gamma_b1.len();
            for g_vec in &self.gamma_deltas {
                n += g_vec.len();
            }
        }
        if learn_pi { n += 1; }
        // One sigma_re_om per breakpoint
        if hierarchical {
            n += self.sigma_re_om.len();
        }
        n
    }

    pub fn to_vec(&self, include_gammas: bool, learn_pi: bool, hierarchical: bool) -> Vec<f64> {
        let mut v = Vec::with_capacity(self.n_params(include_gammas, learn_pi, hierarchical));
        v.extend_from_slice(self.beta_b0.as_slice());
        v.extend_from_slice(self.u_b0.as_slice());
        v.extend_from_slice(self.beta_b1.as_slice());
        for b in &self.beta_deltas { v.extend_from_slice(b.as_slice()); }
        for b in &self.beta_om { v.extend_from_slice(b.as_slice()); }
        for b in &self.beta_rho { v.extend_from_slice(b.as_slice()); }
        v.push(self.sigma);
        v.push(self.sigma_u);
        
        if include_gammas {
            // Gammas
            for &g in &self.gamma_b1 { v.push(if g { 1.0 } else { 0.0 }); }
            for g_vec in &self.gamma_deltas {
                for &g in g_vec { v.push(if g { 1.0 } else { 0.0 }); }
            }
        }
        
        if learn_pi {
            v.push(self.pi);
        }
        if hierarchical {
            for &s in &self.sigma_re_om { v.push(s); }
        }
        v
    }

    // ------------------------------------------------------------------
    // Derived quantities
    // ------------------------------------------------------------------

    pub fn omega_vec(&self, k: usize, x_om: &DMatrix<f64>) -> DVector<f64> {
        x_om * &self.beta_om[k]
    }

    pub fn rho_vec(&self, k: usize, x_rho: &DMatrix<f64>) -> DVector<f64> {
        x_rho * &self.beta_rho[k]
    }

    pub fn means(&self, data: &ModelData) -> DVector<f64> {
        let n = data.n;
        let mut mu = &data.x_b0 * &self.beta_b0;

        // Add random intercepts
        if data.n_groups_b0 > 0 {
            for i in 0..n {
                let g = data.group_b0[i];
                if g >= 0 { mu[i] += self.u_b0[g as usize]; }
            }
        }

        // Segment 1 (initial slope)
        let mut b1_eff = self.beta_b1.clone();
        for j in 0..b1_eff.len() {
            if !self.gamma_b1[j] { b1_eff[j] = 0.0; }
        }
        let b1_vals = &data.x_b1 * &b1_eff;

        if data.n_breakpoints > 0 {
            // Center at first breakpoint for segment 1
            let om1 = self.omega_vec(0, &data.x_om[0]);
            for i in 0..n {
                mu[i] += b1_vals[i] * (data.tau[i] - om1[i]);
            }
        } else {
            // Linear model fallback
            for i in 0..n {
                mu[i] += b1_vals[i] * data.tau[i];
            }
        }

        // Breakpoints
        for k in 0..data.n_breakpoints {
            let om = self.omega_vec(k, &data.x_om[k]);
            let rho = self.rho_vec(k, &data.x_rho[k]);
            
            let mut bd_eff = self.beta_deltas[k].clone();
            for j in 0..bd_eff.len() {
                if !self.gamma_deltas[k][j] { bd_eff[j] = 0.0; }
            }
            let b_delta = &data.x_deltas[k] * &bd_eff;
            
            for i in 0..n {
                let di = data.tau[i] - om[i];
                let si = sigmoid(di * rho[i]);
                mu[i] += b_delta[i] * di * si;
            }
        }

        mu
    }
}

// ---------------------------------------------------------------------------
// Math helpers
// ---------------------------------------------------------------------------

pub fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

pub fn log_truncated_normal_prior(
    values: &[f64],
    means: &[f64],
    sds: &[f64],
    lbs: &[f64],
    ubs: &[f64],
) -> f64 {
    let log_sqrt2pi = 0.5 * std::f64::consts::TAU.ln();
    let mut lp = 0.0;
    for i in 0..values.len() {
        let v = values[i];
        if v < lbs[i] || v > ubs[i] {
            return f64::NEG_INFINITY;
        }
        let z = (v - means[i]) / sds[i];
        lp -= 0.5 * z * z + sds[i].ln() + log_sqrt2pi;
    }
    lp
}
