use nalgebra::{DMatrix, DVector};
use rand::rngs::StdRng;
use rand::SeedableRng;
use statrs::distribution::{Continuous, ContinuousCDF, Normal};
use statrs::function::gamma::ln_gamma;
use extendr_api::prelude::*;

use crate::model::{ModelData, Priors, State};

pub fn unconstrained_to_state(x: &DVector<f64>, data: &ModelData) -> State {
    let mut state = State {
        beta_b0: DVector::zeros(data.x_b0.ncols()),
        u_b0: DVector::zeros(data.n_groups_b0),
        beta_b1: DVector::zeros(data.x_b1.ncols()),
        beta_deltas: data.x_deltas.iter().map(|mat| DVector::zeros(mat.ncols())).collect(),
        beta_om: data.x_om.iter().map(|mat| DVector::zeros(mat.ncols())).collect(),
        beta_rho: data.x_rho.iter().map(|mat| DVector::zeros(mat.ncols())).collect(),
        sigma: 1.0,
        sigma_u: 1.0,
        gamma_b1: vec![true; data.x_b1.ncols()],
        gamma_deltas: data.x_deltas.iter().map(|mat| vec![true; mat.ncols()]).collect(),
        pi: 1.0,
        sigma_re_om: vec![],
    };
    
    let mut idx = 0;
    for i in 0..state.beta_b0.len() { state.beta_b0[i] = x[idx]; idx += 1; }
    for i in 0..state.u_b0.len() { state.u_b0[i] = x[idx]; idx += 1; }
    for i in 0..state.beta_b1.len() { state.beta_b1[i] = x[idx]; idx += 1; }
    
    for k in 0..data.n_breakpoints {
        for i in 0..state.beta_deltas[k].len() { state.beta_deltas[k][i] = x[idx]; idx += 1; }
    }
    for k in 0..data.n_breakpoints {
        for i in 0..state.beta_om[k].len() { state.beta_om[k][i] = x[idx]; idx += 1; }
    }
    for k in 0..data.n_breakpoints {
        for i in 0..state.beta_rho[k].len() { state.beta_rho[k][i] = x[idx]; idx += 1; }
    }
    
    state.sigma = x[idx].exp(); idx += 1;
    state.sigma_u = x[idx].exp();
    
    state
}

pub fn exact_log_posterior_unconstrained(x: &DVector<f64>, data: &ModelData, priors: &Priors) -> f64 {
    let state = unconstrained_to_state(x, data);
    let mut log_p = 0.0;
    
    // 1. Log Likelihood
    let mu = state.means(data);
    let r = &data.y - &mu;
    let sigma2 = state.sigma * state.sigma;
    let n = data.n as f64;
    log_p += -0.5 * n * (2.0 * std::f64::consts::PI * sigma2).ln() - 0.5 * r.dot(&r) / sigma2;

    // 2. Log Priors for linear coefficients (Truncated Normal)
    let log_tn = |values: &[f64], means: &[f64], sds: &[f64], lbs: &[f64], ubs: &[f64]| -> f64 {
        let mut lp = 0.0;
        for i in 0..values.len() {
            let v = values[i];
            if v < lbs[i] || v > ubs[i] { return f64::NEG_INFINITY; }
            let normal = Normal::new(means[i], sds[i]).unwrap();
            let cdf_ub = normal.cdf(ubs[i]);
            let cdf_lb = normal.cdf(lbs[i]);
            let z = (cdf_ub - cdf_lb).max(1e-30); 
            lp += normal.ln_pdf(v) - z.ln();
        }
        lp
    };

    log_p += log_tn(state.beta_b0.as_slice(), &priors.b0_mean, &priors.b0_sd, &priors.b0_lb, &priors.b0_ub);
    log_p += log_tn(state.beta_b1.as_slice(), &priors.b1_mean, &priors.b1_sd, &priors.b1_lb, &priors.b1_ub);
    
    for k in 0..data.n_breakpoints {
        log_p += log_tn(state.beta_deltas[k].as_slice(), &priors.delta_mean[k], &priors.delta_sd[k], &priors.delta_lb[k], &priors.delta_ub[k]);
        log_p += log_tn(state.beta_om[k].as_slice(), &priors.om_mean[k], &priors.om_sd[k], &priors.om_lb[k], &priors.om_ub[k]);
        log_p += log_tn(state.beta_rho[k].as_slice(), &priors.rho_mean[k], &priors.rho_sd[k], &priors.rho_lb[k], &priors.rho_ub[k]);
    }

    // 3. Random effects prior (if applicable)
    if data.n_groups_b0 > 0 {
        let sigma_u2 = state.sigma_u * state.sigma_u;
        let nu = state.u_b0.len() as f64;
        log_p += -0.5 * nu * (2.0 * std::f64::consts::PI * sigma_u2).ln() - 0.5 * state.u_b0.dot(&state.u_b0) / sigma_u2;
    }

    // 4. Variance priors (tau = 1/sigma^2 ~ Gamma)
    let log_gamma_pdf = |x: f64, shape: f64, rate: f64| -> f64 {
        shape * rate.ln() - ln_gamma(shape) + (shape - 1.0) * x.ln() - rate * x
    };

    let tau = 1.0 / sigma2;
    log_p += log_gamma_pdf(tau, priors.sigma_shape, priors.sigma_scale);
    log_p += std::f64::consts::LN_2 - 3.0 * state.sigma.ln(); // Jacobian tau -> sigma

    if data.n_groups_b0 > 0 {
        let tau_u = 1.0 / (state.sigma_u * state.sigma_u);
        log_p += log_gamma_pdf(tau_u, priors.sigma_u_shape, priors.sigma_u_scale);
        log_p += std::f64::consts::LN_2 - 3.0 * state.sigma_u.ln(); // Jacobian tau_u -> sigma_u
    }
    
    // Jacobian for unconstrained space mapping: sigma = exp(x_sigma)
    log_p += state.sigma.ln();
    log_p += state.sigma_u.ln();

    log_p
}

fn mvn_log_pdf(x: &DVector<f64>, mean: &DVector<f64>, cholesky: &nalgebra::Cholesky<f64, nalgebra::Dyn>) -> f64 {
    let p = x.len();
    let mut log_det = 0.0;
    let l = cholesky.l();
    for i in 0..p { log_det += 2.0 * l[(i, i)].ln(); }
    
    let diff = x - mean;
    let y = cholesky.l().solve_lower_triangular(&diff).unwrap();
    let quad = y.dot(&y);
    
    -0.5 * (p as f64) * std::f64::consts::TAU.ln() - 0.5 * log_det - 0.5 * quad
}

pub fn run_bridge_sampling(mcmc_draws: &DMatrix<f64>, data: &ModelData, priors: &Priors, seed: u64) -> f64 {
    let n_samples = mcmc_draws.nrows();
    let p = mcmc_draws.ncols();
    
    // 1. Transform bounded parameters to unconstrained
    let mut unconstrained_draws = mcmc_draws.clone();
    for i in 0..n_samples {
        unconstrained_draws[(i, p - 2)] = unconstrained_draws[(i, p - 2)].ln(); // log(sigma)
        unconstrained_draws[(i, p - 1)] = unconstrained_draws[(i, p - 1)].ln(); // log(sigma_u)
    }

    // Identify active parameters (variance > 1e-10)
    let mut active = Vec::new();
    for j in 0..p {
        let col = unconstrained_draws.column(j);
        let mean_j = col.sum() / (n_samples as f64);
        let mut var_j = 0.0;
        for i in 0..n_samples {
            let diff = col[i] - mean_j;
            var_j += diff * diff;
        }
        var_j /= (n_samples - 1) as f64;
        if var_j > 1e-10 { active.push(j); }
    }
    
    let p_active = active.len();
    if p_active == 0 { return f64::NEG_INFINITY; }

    let mut active_draws = DMatrix::<f64>::zeros(n_samples, p_active);
    for (idx, &j) in active.iter().enumerate() {
        active_draws.set_column(idx, &unconstrained_draws.column(j));
    }

    // 2. Compute Mean and Covariance
    let mut mean = DVector::<f64>::zeros(p_active);
    for i in 0..n_samples { mean += active_draws.row(i).transpose(); }
    mean /= n_samples as f64;

    let mut centered = active_draws.clone();
    for i in 0..n_samples {
        let mut row = centered.row_mut(i);
        row -= mean.transpose();
    }
    let mut cov = (&centered.transpose() * &centered) / ((n_samples - 1) as f64);
    // Add small jitter for numerical stability
    for i in 0..p_active { cov[(i, i)] += 1e-8; }

    let cholesky = cov.cholesky().expect("Covariance not positive definite");

    // 3. Generate Proposal Samples
    let mut proposal_draws_active = DMatrix::<f64>::zeros(n_samples, p_active);
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = rand_distr::Normal::new(0.0, 1.0).unwrap();
    use rand_distr::Distribution;
    for i in 0..n_samples {
        let mut z = DVector::<f64>::zeros(p_active);
        for j in 0..p_active { z[j] = normal.sample(&mut rng); }
        let sample = &mean + cholesky.l() * z;
        proposal_draws_active.set_row(i, &sample.transpose());
    }

    // Fill inactive columns with their constant mean value
    let mut proposal_draws = unconstrained_draws.clone();
    for i in 0..n_samples {
        for (idx, &j) in active.iter().enumerate() {
            proposal_draws[(i, j)] = proposal_draws_active[(i, idx)];
        }
    }

    // 4. Evaluate Log-Densities
    let mut l1 = Vec::with_capacity(n_samples);
    let mut l2 = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let x1 = unconstrained_draws.row(i).transpose();
        let log_p1 = exact_log_posterior_unconstrained(&x1, data, priors);
        
        let mut x1_active = DVector::<f64>::zeros(p_active);
        for (idx, &j) in active.iter().enumerate() { x1_active[idx] = x1[j]; }
        let log_q1 = mvn_log_pdf(&x1_active, &mean, &cholesky);
        l1.push(log_p1 - log_q1);

        let x2 = proposal_draws.row(i).transpose();
        let log_p2 = exact_log_posterior_unconstrained(&x2, data, priors);
        
        let mut x2_active = DVector::<f64>::zeros(p_active);
        for (idx, &j) in active.iter().enumerate() { x2_active[idx] = x2[j]; }
        let log_q2 = mvn_log_pdf(&x2_active, &mean, &cholesky);
        l2.push(log_p2 - log_q2);
    }

    // 5. Iterative Bridge Equation using log-sum-exp
    let mut log_marginal = l1[0]; // Good initial guess is often just the first l1
    let tol = 1e-10;
    let max_iter = 1000;
    
    // N1 = N2 = n_samples, so s1 = s2 = 0.5
    let log_s1 = 0.5f64.ln();
    let log_s2 = 0.5f64.ln();
    
    // Log-sum-exp helper
    let lse = |a: f64, b: f64| -> f64 {
        let m = a.max(b);
        if m == f64::NEG_INFINITY { return m; }
        m + ( (a - m).exp() + (b - m).exp() ).ln()
    };

    for _iter in 0..max_iter {
        // numerator = sum_{j} exp(l2_j) / (s1 exp(l2_j) + s2 exp(log_marginal))
        // we work on log scale: log(term_j) = l2_j - lse(log_s1 + l2_j, log_s2 + log_marginal)
        let mut log_num = f64::NEG_INFINITY;
        for &lj in &l2 {
            let log_denom_j = lse(log_s1 + lj, log_s2 + log_marginal);
            let log_term = lj - log_denom_j;
            log_num = lse(log_num, log_term);
        }
        
        let mut log_den = f64::NEG_INFINITY;
        for &li in &l1 {
            let log_denom_i = lse(log_s1 + li, log_s2 + log_marginal);
            let log_term = -log_denom_i;
            log_den = lse(log_den, log_term);
        }
        
        let new_log_marginal = log_num - log_den;
        if (new_log_marginal - log_marginal).abs() < tol {
            log_marginal = new_log_marginal;
            break;
        }
        log_marginal = new_log_marginal;
    }

    log_marginal
}
