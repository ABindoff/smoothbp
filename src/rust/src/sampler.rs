use nalgebra::{DMatrix, DVector};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand::Rng;
use rand_distr::{Normal, Gamma, Distribution};

use crate::model::{ModelData, Priors, State, log_truncated_normal_prior, sigmoid};



// ---------------------------------------------------------------------------
// Entry point: run one chain, return (n_post × n_params) matrix
// ---------------------------------------------------------------------------

// progress_fn(chain_id, n_chains, iter, n_iter, in_warmup) — called every report_every
// iterations.  Passed as a closure from lib.rs where the extendr context is available.
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

    // draws stored column-major: rows = posterior samples, cols = parameters
    let mut draws = DMatrix::<f64>::zeros(n_post, n_params);

    // Adaptive MH tracking (omega and rho only; sigma_u uses conjugate Gibbs)
    let tune_window = 100usize;
    let mut step_om = step_om_init;
    let mut step_rho = step_rho_init;
    let mut n_accept_om = 0u32;
    let mut n_accept_rho = 0u32;

    // ll cached and refreshed after each block that modifies the mean function
    let mut ll = 0.0f64;
    let mut lp_om = 0.0f64;
    let mut lp_rho = 0.0f64;

    // Report intervals: print at 0%, 10%, 20%, ..., 100%
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

        // After linear update, refresh cached quantities
        ll = state.log_likelihood(data);
        lp_om = state.log_prior_om(priors);
        lp_rho = state.log_prior_rho(priors);

        // ---------------------------------------------------------------
        // Block 2a: MH for beta_om
        // ---------------------------------------------------------------
        let accepted = mh_step_om(data, priors, &mut state, &mut ll, &mut lp_om, step_om, &mut rng);
        if accepted {
            n_accept_om += 1;
        }

        // ---------------------------------------------------------------
        // Block 2b: MH for beta_rho
        // ---------------------------------------------------------------
        let accepted = mh_step_rho(data, priors, &mut state, &mut ll, &mut lp_rho, step_rho, &mut rng);
        if accepted {
            n_accept_rho += 1;
        }

        // ---------------------------------------------------------------
        // Block 3a: Sample sigma (Gibbs, InvGamma conjugate)
        // ---------------------------------------------------------------
        sample_sigma(data, priors, &mut state, &mut rng);
        ll = state.log_likelihood(data);

        // ---------------------------------------------------------------
        // Block 3b: Sample sigma_u (Gibbs, InvGamma conjugate).
        //
        // Key insight: after the NC step, we hold u_j = sigma_u * z_j in state.
        // The full conditional for sigma_u^2 given u_j is simply:
        //   p(sigma_u^2 | u) ∝ prod_j N(u_j; 0, sigma_u^2) * InvGamma(sigma_u^2; a, b)
        //                     = InvGamma(a + m/2,  b + sum_j u_j^2 / 2)
        //
        // This IS the marginalised update: z_j has been "integrated out" by
        // expressing the sufficient statistic as u_j = sigma_u * z_j, leaving
        // sigma_u^2 in a standard conjugate form.  No MH needed.
        // ---------------------------------------------------------------
        if data.n_groups_b0 > 0 {
            sample_sigma_u(priors, &mut state, &mut rng);
        }

        // ---------------------------------------------------------------
        // Adaptive step-size tuning during warmup (omega and rho only)
        // ---------------------------------------------------------------
        if iter < n_warmup && (iter + 1) % tune_window == 0 {
            let rate_om = n_accept_om as f64 / tune_window as f64;
            let rate_rho = n_accept_rho as f64 / tune_window as f64;
            step_om = adapt_step(step_om, rate_om);
            step_rho = adapt_step(step_rho, rate_rho);
            n_accept_om = 0;
            n_accept_rho = 0;
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

    draws
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
//
// This breaks the strong posterior correlation between the overall intercept
// (beta_b0) and the group deviations (u_j) that causes slow mixing in the
// centred parameterisation.  Concretely, when sigma_u is small (strong
// shrinkage), the centred sampler's variance of u_j is ~sigma^2 (large) while
// here var(z_j) = 1/tau_j → 1 regardless of sigma_u.
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

    // Sample z_j from its non-centred full conditional, then set u_j = sigma_u * z_j
    let normal = Normal::new(0.0, 1.0f64).unwrap();
    for j in 0..n_groups {
        let nj     = n_j[j] as f64;
        let tau_j  = nj * sigma_u2 / sigma2 + 1.0;   // posterior precision for z_j
        let v_j    = 1.0 / tau_j;                      // posterior variance
        let mu_j   = v_j * state.sigma_u / sigma2 * sum_r[j];
        let z_j    = mu_j + v_j.sqrt() * normal.sample(rng);
        state.u_b0[j] = state.sigma_u * z_j;
    }
}

// ---------------------------------------------------------------------------
// Block 1b: Sample (beta_b0, beta_b1, beta_b2) jointly | u, omega, rho, sigma
//
// After subtracting random effects: y_tilde_i = y_i - u_{group_i}
// Model: y_tilde_i = [X_b0_i | d_i * X_b1_i | d_i*s_i * X_b2_i] * [beta_b0; beta_b1; beta_b2]
//
// Normal-normal conjugate update:
//   Prec_post = X'X / sigma^2 + Prec_prior
//   mu_post   = Prec_post^{-1} * (X'y_tilde / sigma^2 + Prec_prior * mu_prior)
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

    // Precompute d_i and s_i
    let omega = state.omega_vec(&data.x_om);
    let rho = state.rho_vec(&data.x_rho);
    let d: Vec<f64> = (0..n).map(|i| data.tau[i] - omega[i]).collect();
    let s: Vec<f64> = (0..n).map(|i| sigmoid(d[i] * rho[i])).collect();

    // Build effective design matrix X_full (n × p_lin), column-major
    let mut x_full = DMatrix::<f64>::zeros(n, p_lin);
    for i in 0..n {
        // b0 columns
        for k in 0..p_b0 {
            x_full[(i, k)] = data.x_b0[(i, k)];
        }
        // b1 columns: d_i * X_b1_i
        for k in 0..p_b1 {
            x_full[(i, p_b0 + k)] = d[i] * data.x_b1[(i, k)];
        }
        // b2 columns: d_i * s_i * X_b2_i
        for k in 0..p_b2 {
            x_full[(i, p_b0 + p_b1 + k)] = d[i] * s[i] * data.x_b2[(i, k)];
        }
    }

    // y_tilde = y - u_{group_i}
    let mut y_tilde = data.y.clone();
    if data.n_groups_b0 > 0 {
        for i in 0..n {
            let g = data.group_b0[i];
            if g >= 0 {
                y_tilde[i] -= state.u_b0[g as usize];
            }
        }
    }

    // Prior precision (diagonal) and prior mean
    let b0_r = priors.b0_range();
    let b1_r = priors.b1_range();
    let b2_r = priors.b2_range();
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

    // Prec_post = X'X / sigma^2 + diag(prec_prior)
    let xt = x_full.transpose();
    let xtx = &xt * &x_full;
    let mut prec_post = xtx / sigma2;
    for k in 0..p_lin {
        prec_post[(k, k)] += prec_prior[k];
    }

    // rhs = X'y_tilde / sigma^2 + diag(prec_prior) * mu_prior
    let xty: DVector<f64> = &xt * &y_tilde / sigma2;
    let rhs: DVector<f64> = xty + prec_prior.zip_map(&mu_prior, |p, m| p * m);

    // Solve via Cholesky: Prec_post * theta = rhs  =>  theta_mean = Prec_post^{-1} * rhs
    let chol = prec_post.clone().cholesky().expect("Precision matrix not positive definite");
    let mu_post = chol.solve(&rhs);

    // Sample: theta ~ N(mu_post, Prec_post^{-1})
    // Use: z ~ N(0,I), solve L' * x = z, then theta = mu_post + x
    let normal = Normal::new(0.0, 1.0f64).unwrap();
    let z = DVector::from_iterator(p_lin, (0..p_lin).map(|_| normal.sample(rng)));
    // L is lower-triangular Cholesky factor of Prec_post
    let l = chol.l();
    // Solve L' * x = z  (back-substitution through upper triangular L')
    let x = l.transpose().solve_upper_triangular(&z)
        .expect("Triangular solve failed");
    let theta = mu_post + x;

    // Unpack
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
// Block 2a: Random-walk Metropolis for beta_om
// ---------------------------------------------------------------------------

fn mh_step_om(
    data: &ModelData,
    priors: &Priors,
    state: &mut State,
    ll: &mut f64,
    lp_om: &mut f64,
    step: f64,
    rng: &mut StdRng,
) -> bool {
    let p_om = priors.p_om;
    let r = priors.om_range();

    // Propose
    let normal = Normal::new(0.0, step).unwrap();
    let mut beta_prop = state.beta_om.clone();
    for k in 0..p_om {
        beta_prop[k] += normal.sample(rng);
    }

    // Check bounds on proposed values
    for (k, i) in r.clone().enumerate() {
        if beta_prop[k] < priors.lb[i] || beta_prop[k] > priors.ub[i] {
            return false;
        }
    }

    // Proposed log-prior
    let lp_prop = log_truncated_normal_prior(
        beta_prop.as_slice(),
        &priors.mean[r.clone()],
        &priors.sd[r.clone()],
        &priors.lb[r.clone()],
        &priors.ub[r],
    );
    if lp_prop == f64::NEG_INFINITY {
        return false;
    }

    // Proposed log-likelihood
    let mut state_prop = state.clone();
    state_prop.beta_om = beta_prop;
    let ll_prop = state_prop.log_likelihood(data);

    // Metropolis acceptance
    let log_alpha = (ll_prop + lp_prop) - (*ll + *lp_om);
    if rng.gen::<f64>().ln() < log_alpha {
        *state = state_prop;
        *ll = ll_prop;
        *lp_om = lp_prop;
        true
    } else {
        false
    }
}

// ---------------------------------------------------------------------------
// Block 2b: Random-walk Metropolis for beta_rho
// ---------------------------------------------------------------------------

fn mh_step_rho(
    data: &ModelData,
    priors: &Priors,
    state: &mut State,
    ll: &mut f64,
    lp_rho: &mut f64,
    step: f64,
    rng: &mut StdRng,
) -> bool {
    let p_rho = priors.p_rho;
    let r = priors.rho_range();

    let normal = Normal::new(0.0, step).unwrap();
    let mut beta_prop = state.beta_rho.clone();
    for k in 0..p_rho {
        beta_prop[k] += normal.sample(rng);
    }

    // Check bounds
    for (k, i) in r.clone().enumerate() {
        if beta_prop[k] < priors.lb[i] || beta_prop[k] > priors.ub[i] {
            return false;
        }
    }

    let lp_prop = log_truncated_normal_prior(
        beta_prop.as_slice(),
        &priors.mean[r.clone()],
        &priors.sd[r.clone()],
        &priors.lb[r.clone()],
        &priors.ub[r],
    );
    if lp_prop == f64::NEG_INFINITY {
        return false;
    }

    let mut state_prop = state.clone();
    state_prop.beta_rho = beta_prop;
    let ll_prop = state_prop.log_likelihood(data);

    let log_alpha = (ll_prop + lp_prop) - (*ll + *lp_rho);
    if rng.gen::<f64>().ln() < log_alpha {
        *state = state_prop;
        *ll = ll_prop;
        *lp_rho = lp_prop;
        true
    } else {
        false
    }
}

// ---------------------------------------------------------------------------
// Block 3a: Sample sigma^2 | everything  (InvGamma conjugate)
//
// Prior:  sigma^2 ~ InvGamma(shape_0, scale_0)
// Update: sigma^2 ~ InvGamma(shape_0 + n/2, scale_0 + SSR/2)
// Sample: X ~ Gamma(shape_post, rate_post), sigma^2 = 1/X
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

    // Gamma(shape, rate) sample: use rand_distr's Gamma(shape, scale=1/rate)
    let g = Gamma::new(shape_post, 1.0 / rate_post).unwrap();
    let x = g.sample(rng);
    state.sigma = (1.0 / x).sqrt();
}

// ---------------------------------------------------------------------------
// Block 3b: Sample sigma_u^2 | u  (InvGamma conjugate Gibbs)
//
// Prior:   sigma_u^2 ~ InvGamma(a, b)
// Likelihood: u_j | sigma_u^2 ~ N(0, sigma_u^2)  iid  (j = 1..m)
//
// Full conditional (by normal-inverse-gamma conjugacy):
//   sigma_u^2 | u ~ InvGamma(a + m/2,  b + sum_j u_j^2 / 2)
//
// After the NC Gibbs step, state.u_b0[j] = sigma_u * z_j, so sum_j u_j^2 is
// well-defined as the scaled sum of squares of z_j.  Sampling sigma_u^2 from
// this distribution then implicitly updates the scale of z_j for the next
// iteration without needing any MH step.
//
// Sampling X ~ Gamma(shape, rate=1/scale) then sigma_u^2 = 1/X.
// ---------------------------------------------------------------------------

fn sample_sigma_u(priors: &Priors, state: &mut State, rng: &mut StdRng) {
    let m       = state.u_b0.len() as f64;
    let ss_u: f64 = state.u_b0.iter().map(|u| u * u).sum();

    let shape_post = priors.sigma_u_shape + 0.5 * m;
    let rate_post  = priors.sigma_u_scale  + 0.5 * ss_u;  // rate = 1/scale for Gamma

    let g = Gamma::new(shape_post, 1.0 / rate_post).unwrap();
    state.sigma_u = (1.0 / g.sample(rng)).sqrt();
}

// ---------------------------------------------------------------------------
// Adaptive step-size tuning targeting 23.4% acceptance
// ---------------------------------------------------------------------------

fn adapt_step(step: f64, accept_rate: f64) -> f64 {
    // Multiplicative adjustment: scale up if accepting too much, down if too little
    let target = 0.234;
    let factor = (accept_rate / target).clamp(0.5, 2.0);
    (step * factor).max(1e-6)
}
