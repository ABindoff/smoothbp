#' Robustify a smoothbp_fit object using a Bayesian Sandwich approach
#'
#' @param object A \code{smoothbp_fit} object.
#' @param cluster String naming the column in \code{object$data} defining the clusters (e.g., Subject ID).
#' @param ... Unused.
#'
#' @return A new \code{smoothbp_fit} object where the MCMC draws have been affinely transformed to match the robust clustered covariance matrix.
#' @export
robustify <- function(object, cluster, ...) {
  UseMethod("robustify")
}

#' @export
robustify.smoothbp_fit <- function(object, cluster, ...) {
  if (!cluster %in% names(object$data)) {
    stop(sprintf("Cluster column '%s' not found in data.", cluster))
  }
  
  cluster_id <- as.character(object$data[[cluster]])
  
  # 1. Extract the center (posterior mean)
  draw_mat <- posterior::as_draws_matrix(object$draws)
  if (nrow(draw_mat) == 0) {
    stop("No post-warmup draws found in the model fit. Please ensure iter > warmup when calling smoothbp().")
  }
  theta_bar <- colMeans(draw_mat)
  
  # 2. Calculate V_naive (original covariance)
  V_naive <- stats::cov(draw_mat)
  
  # Find 'slab' parameters (parameters that have non-zero variance)
  # EXCLUDE random effects (u[...]) and variance components (sigma_u, etc) 
  # from the robustification matrix to prevent rank-deficiency in large hierarchical models.
  var_theta <- diag(V_naive)
  param_names <- names(theta_bar)
  is_target <- !grepl("^(u\\[|sigma_u|sigma_re_om)", param_names)
  active_idx <- which(var_theta > 1e-12 & is_target)
  if (length(active_idx) < sum(is_target)) {
    message("Note: Some target parameters have effectively zero variance (e.g. spike). Robustification applied only to active parameters.")
  }
  
  # Subset to active parameters
  theta_bar_act <- theta_bar[active_idx]
  V_naive_act <- V_naive[active_idx, active_idx, drop = FALSE]
  
  y <- as.numeric(object$data[[object$response]])
  
  # Helper to compute mu given active parameters
  .compute_mu <- function(theta_act) {
    theta_full <- theta_bar
    theta_full[active_idx] <- theta_act
    
    dm <- object$dm
    tau <- as.numeric(object$data[[object$time]])
    n <- length(tau)
    n_bp <- length(dm$X_deltas)
    col_names <- names(theta_full)
    
    b0_cols  <- which(grepl("^b0_", col_names))
    b1_cols  <- which(grepl("^b1_", col_names))
    u_cols   <- which(grepl("^u\\[", col_names))
    
    mu_i <- as.vector(dm$X_b0 %*% as.numeric(theta_full[b0_cols]))
    beta_b1 <- as.numeric(theta_full[b1_cols])
    gamma_b1_cols <- which(grepl("^gamma_b1_", col_names))
    if (length(gamma_b1_cols) > 0) beta_b1 <- beta_b1 * as.numeric(theta_full[gamma_b1_cols])
    b1_vals <- as.vector(dm$X_b1 %*% beta_b1)
    
    if (n_bp > 0) {
      om_cols <- which(grepl("^omega1_", col_names))
      om1_i <- as.vector(dm$X_om[[1]] %*% as.numeric(theta_full[om_cols]))
      mu_i <- mu_i + b1_vals * (tau - om1_i)
    } else {
      mu_i <- mu_i + b1_vals * tau
    }
    
    if (n_bp > 0) {
      for (k in seq_len(n_bp)) {
        delta_cols <- which(grepl(paste0("^delta", k, "_"), col_names))
        om_cols    <- which(grepl(paste0("^omega", k, "_"), col_names))
        rho_cols   <- which(grepl(paste0("^rho", k, "_"), col_names))
        gamma_delta_cols <- which(grepl(paste0("^gamma_delta", k, "_"), col_names))
        
        b_delta <- as.numeric(theta_full[delta_cols])
        if (length(gamma_delta_cols) > 0) b_delta <- b_delta * as.numeric(theta_full[gamma_delta_cols])
        
        delta_i <- as.vector(dm$X_deltas[[k]] %*% b_delta)
        om_i    <- as.vector(dm$X_om[[k]] %*% as.numeric(theta_full[om_cols]))
        rho_i   <- as.vector(dm$X_rho[[k]] %*% as.numeric(theta_full[rho_cols]))
        
        di <- tau - om_i
        si <- 1 / (1 + exp(-di * rho_i))
        mu_i <- mu_i + delta_i * di * si
      }
    }
    
    if (dm$n_groups_b0 > 0) {
      u_b0 <- as.numeric(theta_full[u_cols])
      for (i in seq_len(n)) {
        g <- dm$group_b0[i]
        if (g >= 0L) mu_i[i] <- mu_i[i] + u_b0[g + 1L]
      }
    }
    return(mu_i)
  }
  
  # Numerical Jacobian for mu w.r.t theta_act
  eps <- 1e-5
  n_obs <- length(y)
  n_act <- length(theta_bar_act)
  J_mu <- matrix(0, nrow = n_obs, ncol = n_act)
  
  mu_base <- .compute_mu(theta_bar_act)
  sigma_base <- theta_bar["sigma"]
  
  for (p in seq_len(n_act)) {
    theta_eps <- theta_bar_act
    theta_eps[p] <- theta_eps[p] + eps
    mu_eps <- .compute_mu(theta_eps)
    J_mu[, p] <- (mu_eps - mu_base) / eps
  }
  
  # Gradient of log-likelihood for each observation
  dll_dmu <- (y - mu_base) / (sigma_base^2)
  
  # Score matrix (N x P) for mu parameters
  score_mat <- matrix(0, nrow = n_obs, ncol = n_act)
  for (p in seq_len(n_act)) {
    param_name <- names(theta_bar_act)[p]
    if (param_name == "sigma") {
      score_mat[, p] <- -1 / sigma_base + ((y - mu_base)^2) / (sigma_base^3)
    } else {
      score_mat[, p] <- dll_dmu * J_mu[, p]
    }
  }
  
  # 3. Calculate V_robust (Sandwich)
  unique_clusters <- unique(cluster_id)
  U_clust <- matrix(0, nrow = length(unique_clusters), ncol = n_act)
  for (i in seq_along(unique_clusters)) {
    idx <- which(cluster_id == unique_clusters[i])
    if (length(idx) == 1) {
      U_clust[i, ] <- score_mat[idx, ]
    } else {
      U_clust[i, ] <- colSums(score_mat[idx, , drop = FALSE])
    }
  }
  
  # Meat of the sandwich
  B <- t(U_clust) %*% U_clust
  
  # Finite sample correction
  n_clust <- length(unique_clusters)
  dfc <- (n_clust / (n_clust - 1)) * ((n_obs - 1) / (n_obs - n_act))
  B <- B * dfc
  
  # V_robust = V_naive * B * V_naive
  V_robust_act <- V_naive_act %*% B %*% V_naive_act
  
  # 4. Affine Transformation
  L_naive <- t(chol(V_naive_act))
  
  # Make V_robust symmetric and add a tiny ridge to ensure positive definiteness
  V_robust_act <- (V_robust_act + t(V_robust_act)) / 2
  diag(V_robust_act) <- diag(V_robust_act) + 1e-8
  L_rob <- t(chol(V_robust_act))
  
  # Transformation matrix T_mat = L_rob %*% solve(L_naive)
  T_mat <- L_rob %*% solve(L_naive)
  
  # Apply to array
  n_iter <- dim(object$draws)[1]
  n_chains <- dim(object$draws)[2]
  new_draws <- object$draws
  
  for (c in seq_len(n_chains)) {
    for (i in seq_len(n_iter)) {
      theta_s <- as.numeric(object$draws[i, c, active_idx])
      theta_new <- theta_bar_act + as.vector(T_mat %*% (theta_s - theta_bar_act))
      new_draws[i, c, active_idx] <- theta_new
    }
  }
  
  object$draws <- posterior::as_draws_array(new_draws)
  object$is_robust <- TRUE
  object$robust_cluster <- cluster
  
  return(object)
}
