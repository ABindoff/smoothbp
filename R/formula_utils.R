# Internal utilities for parsing formulas and building design matrices

# ---------------------------------------------------------------------------
# Parse random-effect terms from a formula, returning:
#   $fixed   : formula with RE terms removed
#   $re_group: character name of the grouping variable (or NULL)
# ---------------------------------------------------------------------------
.parse_re <- function(formula) {
  tt <- terms(formula, keep.order = TRUE)
  trm <- attr(tt, "term.labels")
  re_terms <- grep("^1\\s*\\|", trm, value = TRUE)

  if (length(re_terms) == 0L) {
    return(list(fixed = formula, re_group = NULL))
  }
  if (length(re_terms) > 1L) {
    stop("smoothbp supports at most one random-intercept term per parameter.")
  }

  group_name <- trimws(sub("^1\\s*\\|\\s*", "", re_terms[1]))

  # Rebuild fixed formula without the RE term
  fixed_trm <- setdiff(trm, re_terms)
  has_int <- attr(tt, "intercept") == 1L
  fixed_rhs <- if (length(fixed_trm) == 0L) {
    if (has_int) "1" else "0"
  } else {
    paste(c(if (!has_int) "0", fixed_trm), collapse = " + ")
  }
  fixed_fml <- stats::as.formula(paste("~", fixed_rhs))
  environment(fixed_fml) <- environment(formula)

  list(fixed = fixed_fml, re_group = group_name)
}

# ---------------------------------------------------------------------------
# Build design matrices for all parameters (including multiple segments).
# ---------------------------------------------------------------------------
.build_design_matrices <- function(
    b0_fml, b1_fml, deltas_fml, omega_fml, rho_fml,
    data
) {
  mk_mm <- function(fml, dat) {
    stats::model.matrix(fml, data = dat)
  }

  # Helper for processing lists of formulas
  mk_list_mm <- function(fml_list, dat) {
    if (!is.list(fml_list)) fml_list <- list(fml_list)
    lapply(fml_list, function(f) mk_mm(.parse_re(f)$fixed, dat))
  }

  # b0 random effects
  b0_parsed <- .parse_re(b0_fml)
  X_b0 <- mk_mm(b0_parsed$fixed, data)
  
  X_b1 <- mk_mm(.parse_re(b1_fml)$fixed, data)
  X_deltas <- mk_list_mm(deltas_fml, data)
  X_om     <- mk_list_mm(omega_fml, data)
  X_rho    <- mk_list_mm(rho_fml, data)

  n_bp <- length(X_deltas)
  if (length(X_om) != n_bp || length(X_rho) != n_bp) {
    stop("Number of formulas for deltas, omega, and rho must match.")
  }

  group_b0 <- integer(nrow(data)) - 1L
  n_groups_b0 <- 0L
  group_levels_b0 <- character(0)

  if (!is.null(b0_parsed$re_group)) {
    gfactor <- factor(data[[b0_parsed$re_group]])
    group_levels_b0 <- levels(gfactor)
    n_groups_b0 <- nlevels(gfactor)
    group_b0 <- as.integer(gfactor) - 1L
  }

  list(
    X_b0 = X_b0,
    X_b1 = X_b1,
    X_deltas = X_deltas,
    X_om     = X_om,
    X_rho    = X_rho,
    group_b0        = group_b0,
    n_groups_b0     = n_groups_b0,
    group_levels_b0 = group_levels_b0,
    col_names_b0    = colnames(X_b0),
    col_names_b1    = colnames(X_b1),
    col_names_deltas = lapply(X_deltas, colnames),
    col_names_om     = lapply(X_om, colnames),
    col_names_rho    = lapply(X_rho, colnames)
  )
}

# ---------------------------------------------------------------------------
# Build concatenated prior vectors
# ---------------------------------------------------------------------------
.build_prior_vectors <- function(priors, dm) {
  expand_list <- function(spec_list, nms_list) {
    if (!is.list(spec_list) || inherits(spec_list, "smoothbp_prior")) {
        spec_list <- rep(list(spec_list), length(nms_list))
    }
    lapply(seq_along(nms_list), function(i) .expand_prior(spec_list[[i]], nms_list[[i]]))
  }

  p_b0 <- .expand_prior(priors$b0, dm$col_names_b0)
  p_b1 <- .expand_prior(priors$b1, dm$col_names_b1)
  
  p_deltas <- expand_list(priors$deltas, dm$col_names_deltas)
  p_om     <- expand_list(priors$omega,  dm$col_names_om)
  p_rho    <- expand_list(priors$rho,    dm$col_names_rho)

  list(
    b0 = p_b0,
    b1 = p_b1,
    deltas = p_deltas,
    om = p_om,
    rho = p_rho
  )
}

# ---------------------------------------------------------------------------
# Build the full parameter name vector
# ---------------------------------------------------------------------------
.param_names <- function(dm, pv, learn_pi = FALSE) {
  names <- c(
    paste0("b0_", pv$b0$name),
    if (dm$n_groups_b0 > 0) paste0("u[", dm$group_levels_b0, "]") else character(0),
    paste0("b1_", pv$b1$name)
  )
  
  for (i in seq_along(pv$deltas)) {
    names <- c(names, paste0("delta", i, "_", pv$deltas[[i]]$name))
  }
  for (i in seq_along(pv$om)) {
    names <- c(names, paste0("omega", i, "_", pv$om[[i]]$name))
  }
  for (i in seq_along(pv$rho)) {
    names <- c(names, paste0("rho", i, "_", pv$rho[[i]]$name))
  }
  
  names <- c(names, "sigma", "sigma_u")
  
  # SS indicators
  # (Will be added by run_chain_ss wrapper if needed)
  
  names
}
