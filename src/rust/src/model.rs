use nalgebra::{DMatrix, DVector};

// ---------------------------------------------------------------------------
// Data passed in from R
// ---------------------------------------------------------------------------

pub struct ModelData {
    pub y: DVector<f64>,
    pub tau: DVector<f64>,
    pub x_b0: DMatrix<f64>,
    pub x_b1: DMatrix<f64>,
    pub x_b2: DMatrix<f64>,
    pub x_om: DMatrix<f64>,
    pub x_rho: DMatrix<f64>,
    /// 0-based group indices for b0 random intercept; -1 if observation has no RE
    pub group_b0: Vec<i32>,
    pub n_groups_b0: usize,
    pub n: usize,
}

// ---------------------------------------------------------------------------
// Prior hyperparameters
// ---------------------------------------------------------------------------

/// Priors for all regression coefficients, stored as flat vectors ordered:
/// [beta_b0 (p_b0), beta_b1 (p_b1), beta_b2 (p_b2), beta_om (p_om), beta_rho (p_rho)]
pub struct Priors {
    pub mean: Vec<f64>,
    pub sd: Vec<f64>,
    /// Lower bound for each coefficient (-INF if unconstrained)
    pub lb: Vec<f64>,
    /// Upper bound for each coefficient (+INF if unconstrained)
    pub ub: Vec<f64>,
    pub sigma_shape: f64,
    pub sigma_scale: f64,
    pub sigma_u_shape: f64,
    pub sigma_u_scale: f64,
    pub p_b0: usize,
    pub p_b1: usize,
    pub p_b2: usize,
    pub p_om: usize,
    pub p_rho: usize,
    /// Cached flag: true if any b0, b1, or b2 coefficient has finite bounds
    pub has_lin_bounds: bool,
}

impl Priors {
    pub fn b0_range(&self) -> std::ops::Range<usize> {
        0..self.p_b0
    }
    pub fn b1_range(&self) -> std::ops::Range<usize> {
        self.p_b0..(self.p_b0 + self.p_b1)
    }
    pub fn b2_range(&self) -> std::ops::Range<usize> {
        (self.p_b0 + self.p_b1)..(self.p_b0 + self.p_b1 + self.p_b2)
    }
    pub fn om_range(&self) -> std::ops::Range<usize> {
        let s = self.p_b0 + self.p_b1 + self.p_b2;
        s..(s + self.p_om)
    }
    pub fn rho_range(&self) -> std::ops::Range<usize> {
        let s = self.p_b0 + self.p_b1 + self.p_b2 + self.p_om;
        s..(s + self.p_rho)
    }

    /// True if any linear coefficient (b0, b1, or b2) has a finite bound.
    /// This is precomputed at construction for O(1) lookup.
    pub fn lin_has_finite_bounds(&self) -> bool {
        self.has_lin_bounds
    }
}

// ---------------------------------------------------------------------------
// Spike-and-slab configuration (used by run_chain_ss)
// ---------------------------------------------------------------------------

/// Per-coefficient spike-and-slab configuration for b2 variable selection.
///
/// When `spike_mask[k]` is true, coefficient k of b2 has a spike-and-slab
/// prior: with probability `pi[k]`, it follows the slab (normal prior); with
/// probability `1 - pi[k]`, it is exactly zero.  When a b2 coefficient is
/// spiked to zero, the same-named coefficients in omega and rho (if they
/// exist) are also zeroed via `om_map` and `rho_map`.
pub struct SpikeSlabConfig {
    /// Which b2 coefficients are eligible for spike-and-slab (length p_b2)
    pub spike_mask: Vec<bool>,
    /// Initial/fixed inclusion probability for each b2 coefficient (length p_b2)
    pub pi_init: Vec<f64>,
    /// Mapping from b2 coefficient index → omega coefficient index (-1 if no match)
    pub om_map: Vec<i32>,
    /// Mapping from b2 coefficient index → rho coefficient index (-1 if no match)
    pub rho_map: Vec<i32>,
    /// Beta hyperprior shape parameter a (0 = fixed pi, >0 = learn pi)
    pub beta_a: f64,
    /// Beta hyperprior shape parameter b
    pub beta_b: f64,
}

// ---------------------------------------------------------------------------
// Sampler state
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct State {
    pub beta_b0: DVector<f64>,
    pub u_b0: DVector<f64>,   // random intercepts; length = n_groups_b0
    pub beta_b1: DVector<f64>,
    pub beta_b2: DVector<f64>,
    pub beta_om: DVector<f64>,
    pub beta_rho: DVector<f64>,
    pub sigma: f64,    // residual SD
    pub sigma_u: f64,  // random-effect SD (unused when n_groups == 0)
    /// Spike-and-slab inclusion indicators (length p_b2; all true in base model)
    pub gamma: Vec<bool>,
    /// Current inclusion probability (sampled when Beta hyperprior is active)
    pub pi: f64,
}

impl State {
    /// Number of scalar parameters stored per draw (base model, no gamma)
    pub fn n_params(&self) -> usize {
        self.beta_b0.len()
            + self.u_b0.len()
            + self.beta_b1.len()
            + self.beta_b2.len()
            + self.beta_om.len()
            + self.beta_rho.len()
            + 2 // sigma, sigma_u
    }

    /// Number of parameters including gamma indicators and pi
    pub fn n_params_ss(&self, learn_pi: bool) -> usize {
        self.n_params() + self.gamma.len() + if learn_pi { 1 } else { 0 }
    }

    /// Flatten to a contiguous Vec for storage in the draw matrix.
    /// Order: beta_b0 | u_b0 | beta_b1 | beta_b2 | beta_om | beta_rho | sigma | sigma_u
    pub fn to_vec(&self) -> Vec<f64> {
        let mut v = Vec::with_capacity(self.n_params());
        v.extend_from_slice(self.beta_b0.as_slice());
        v.extend_from_slice(self.u_b0.as_slice());
        v.extend_from_slice(self.beta_b1.as_slice());
        v.extend_from_slice(self.beta_b2.as_slice());
        v.extend_from_slice(self.beta_om.as_slice());
        v.extend_from_slice(self.beta_rho.as_slice());
        v.push(self.sigma);
        v.push(self.sigma_u);
        v
    }

    /// Flatten to Vec including gamma indicators and optionally pi.
    /// Order: beta_b0 | u_b0 | beta_b1 | beta_b2 | beta_om | beta_rho | sigma | sigma_u | gamma [| pi]
    pub fn to_vec_ss(&self, learn_pi: bool) -> Vec<f64> {
        let mut v = self.to_vec();
        for &g in &self.gamma {
            v.push(if g { 1.0 } else { 0.0 });
        }
        if learn_pi {
            v.push(self.pi);
        }
        v
    }

    // ------------------------------------------------------------------
    // Derived quantities
    // ------------------------------------------------------------------

    /// omega_i = X_om * beta_om  (n-vector)
    pub fn omega_vec(&self, x_om: &DMatrix<f64>) -> DVector<f64> {
        x_om * &self.beta_om
    }

    /// rho_i = X_rho * beta_rho  (n-vector)
    pub fn rho_vec(&self, x_rho: &DMatrix<f64>) -> DVector<f64> {
        x_rho * &self.beta_rho
    }

    /// Compute the model mean for every observation.
    /// mu_i = b0_i + b1_i * d_i + b2_i * d_i * sigma(d_i * rho_i)
    /// where d_i = tau_i - omega_i
    pub fn means(&self, data: &ModelData) -> DVector<f64> {
        let omega = self.omega_vec(&data.x_om);
        let rho = self.rho_vec(&data.x_rho);

        // d_i = tau_i - omega_i
        let d: DVector<f64> = data.tau.zip_map(&omega, |t, w| t - w);
        // s_i = sigmoid(d_i * rho_i)
        let s: DVector<f64> = d.zip_map(&rho, |di, ri| sigmoid(di * ri));

        let b0_fixed = &data.x_b0 * &self.beta_b0;
        let b1_vals = &data.x_b1 * &self.beta_b1;
        let b2_vals = &data.x_b2 * &self.beta_b2;

        // b1_i * d_i  and  b2_i * d_i * s_i
        let b1_contrib: DVector<f64> = d.zip_map(&b1_vals, |di, b1i| di * b1i);
        let b2_contrib: DVector<f64> =
            d.zip_map(&s, |di, si| di * si)
                .zip_map(&b2_vals, |ds, b2i| ds * b2i);

        let mut mu = b0_fixed + b1_contrib + b2_contrib;

        // Add random intercepts
        if data.n_groups_b0 > 0 {
            for i in 0..data.n {
                let g = data.group_b0[i];
                if g >= 0 {
                    mu[i] += self.u_b0[g as usize];
                }
            }
        }

        mu
    }

}

// ---------------------------------------------------------------------------
// Math helpers
// ---------------------------------------------------------------------------

/// Numerically stable logistic sigmoid
pub fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Log-density of an (optionally truncated) normal prior.
/// Returns -INF if any value violates its bounds.
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
