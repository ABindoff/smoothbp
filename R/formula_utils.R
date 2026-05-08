# Internal utilities for parsing formulas and building design matrices

# ---------------------------------------------------------------------------
# Parse random-effect terms from a formula, returning:
#   $fixed   : formula with RE terms removed (for fixed-effect part)
#   $re_group: character name of the grouping variable (or NULL)
# ---------------------------------------------------------------------------
.parse_re <- function(formula) {
  tt  <- terms(formula, keep.order = TRUE)
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
  has_int   <- attr(tt, "intercept") == 1L
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
# Build a single design matrix from a formula, returning it with a
# 're_mask' attribute (integer vector: 1 = random effect, 0 = fixed).
#
# For  ~ (1 | group):
#   - Column 1: intercept (grand mean, fixed, mask=0)
#   - Columns 2..K+1: one dummy per level of 'group' using identity coding
#     (all K levels present, mask=1).  Shrinking these to 0 = shrinking
#     toward the grand mean.
#
# For plain ~ x (no RE terms): standard model.matrix, all mask=0.
# ---------------------------------------------------------------------------
.build_mm <- function(formula, data) {
  parsed <- .parse_re(formula)

  if (is.null(parsed$re_group)) {
    # No random effects: standard design matrix
    X <- stats::model.matrix(parsed$fixed, data = data)
    attr(X, "re_mask") <- integer(ncol(X))          # all fixed
    return(X)
  }

  # ---- (1 | group) formula -----------------------------------------------
  # Fixed part (usually just an intercept)
  X_fixed <- stats::model.matrix(parsed$fixed, data = data)

  # Random part: one dummy column per level of group (identity coding)
  grp       <- factor(data[[parsed$re_group]])
  lvls      <- levels(grp)
  K         <- length(lvls)
  X_re      <- matrix(0L, nrow = nrow(data), ncol = K)
  colnames(X_re) <- paste0("re_", parsed$re_group, "_", lvls)
  for (j in seq_len(K)) X_re[grp == lvls[j], j] <- 1L

  X <- cbind(X_fixed, X_re)
  attr(X, "re_mask") <- c(integer(ncol(X_fixed)),   # fixed columns: 0
                           rep(1L, K))               # RE columns:    1
  X
}

# ---------------------------------------------------------------------------
# Build design matrices for all parameters (including multiple segments).
# ---------------------------------------------------------------------------
.build_design_matrices <- function(
    b0_fml, b1_fml, deltas_fml, omega_fml, rho_fml,
    data
) {
  mk_list_mm <- function(fml_list, dat) {
    if (!is.list(fml_list)) fml_list <- list(fml_list)
    lapply(fml_list, function(f) .build_mm(f, dat))
  }

  # b0: may carry a (1|group) RE for the random intercept.
  # We do NOT use .build_mm for b0 because the Rust sampler handles b0's random 
  # intercept natively via the group_b0 mechanism and sigma_u, rather than as 
  # identity-coded dummy columns in the design matrix.
  b0_parsed <- .parse_re(b0_fml)
  X_b0      <- stats::model.matrix(b0_parsed$fixed, data = data)
  attr(X_b0, "re_mask") <- integer(ncol(X_b0))
  X_b1     <- .build_mm(b1_fml,    data)
  X_deltas <- mk_list_mm(deltas_fml, data)
  X_om     <- mk_list_mm(omega_fml,  data)
  X_rho    <- mk_list_mm(rho_fml,   data)

  n_bp <- length(X_deltas)
  if (length(X_om) != n_bp || length(X_rho) != n_bp) {
    stop("Number of formulas for deltas, omega, and rho must match.")
  }

  # b0 classic RE grouping (for random intercepts via group_b0 mechanism)
  group_b0        <- integer(nrow(data)) - 1L
  n_groups_b0     <- 0L
  group_levels_b0 <- character(0)

  if (!is.null(b0_parsed$re_group)) {
    gfactor         <- factor(data[[b0_parsed$re_group]])
    group_levels_b0 <- levels(gfactor)
    n_groups_b0     <- nlevels(gfactor)
    group_b0        <- as.integer(gfactor) - 1L
  }

  list(
    X_b0     = X_b0,
    X_b1     = X_b1,
    X_deltas = X_deltas,
    X_om     = X_om,
    X_rho    = X_rho,
    group_b0        = group_b0,
    n_groups_b0     = n_groups_b0,
    group_levels_b0 = group_levels_b0,
    col_names_b0     = colnames(X_b0),
    col_names_b1     = colnames(X_b1),
    col_names_deltas = lapply(X_deltas, colnames),
    col_names_om     = lapply(X_om,     colnames),
    col_names_rho    = lapply(X_rho,    colnames)
  )
}

# ---------------------------------------------------------------------------
# Extract the re_mask for a list of design matrices (e.g. X_om).
# Returns a list of integer vectors, one per breakpoint.
# ---------------------------------------------------------------------------
.get_re_masks <- function(X_list) {
  lapply(X_list, function(X) {
    m <- attr(X, "re_mask")
    if (is.null(m)) integer(ncol(X)) else m
  })
}

# ---------------------------------------------------------------------------
# Check whether ANY design matrix in a list has random effects
# ---------------------------------------------------------------------------
.has_re <- function(X_list) {
  any(sapply(.get_re_masks(X_list), function(m) any(m == 1L)))
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

  names
}
