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
# Build design matrices for all five parameters.
# Returns a list with elements: X_b0, X_b1, X_b2, X_om, X_rho,
#                               group_b0 (integer vector, 0-based, -1 if no RE),
#                               n_groups_b0, group_levels_b0,
#                               col_names_b0, ..., col_names_rho
# ---------------------------------------------------------------------------
.build_design_matrices <- function(
    b0_fml, b1_fml, b2_fml, omega_fml, rho_fml,
    data
) {
  # Helper: build model matrix from a fixed formula
  mk_mm <- function(fml, dat) {
    mm <- stats::model.matrix(fml, data = dat)
    mm
  }

  # Parse random effects for b0
  b0_parsed <- .parse_re(b0_fml)
  b1_parsed <- .parse_re(b1_fml)
  b2_parsed <- .parse_re(b2_fml)
  om_parsed <- .parse_re(omega_fml)
  rho_parsed <- .parse_re(rho_fml)

  # Warn if RE specified for non-b0 parameters
  for (nm in c("b1", "b2", "om", "rho")) {
    p <- get(paste0(nm, "_parsed"))
    if (!is.null(p$re_group)) {
      warning(sprintf(
        "Random effects for '%s' are not yet supported; ignoring (1 | %s).",
        nm, p$re_group
      ))
    }
  }

  X_b0  <- mk_mm(b0_parsed$fixed, data)
  X_b1  <- mk_mm(b1_parsed$fixed, data)
  X_b2  <- mk_mm(b2_parsed$fixed, data)
  X_om  <- mk_mm(om_parsed$fixed, data)
  X_rho <- mk_mm(rho_parsed$fixed, data)

  # Random effect grouping for b0
  group_b0 <- integer(nrow(data)) - 1L  # default: no RE (-1)
  n_groups_b0 <- 0L
  group_levels_b0 <- character(0)

  if (!is.null(b0_parsed$re_group)) {
    gvar <- data[[b0_parsed$re_group]]
    if (is.null(gvar)) {
      stop(sprintf("Variable '%s' not found in data for random effect.", b0_parsed$re_group))
    }
    gfactor <- factor(gvar)
    group_levels_b0 <- levels(gfactor)
    n_groups_b0 <- nlevels(gfactor)
    group_b0 <- as.integer(gfactor) - 1L  # 0-based
  }

  list(
    X_b0  = X_b0,
    X_b1  = X_b1,
    X_b2  = X_b2,
    X_om  = X_om,
    X_rho = X_rho,
    group_b0        = group_b0,
    n_groups_b0     = n_groups_b0,
    group_levels_b0 = group_levels_b0,
    col_names_b0    = colnames(X_b0),
    col_names_b1    = colnames(X_b1),
    col_names_b2    = colnames(X_b2),
    col_names_om    = colnames(X_om),
    col_names_rho   = colnames(X_rho)
  )
}

# ---------------------------------------------------------------------------
# Build concatenated prior vectors from a smoothbp_priors object and
# the column names for each parameter's design matrix.
# Returns: list(mean, sd, lb, ub) each of length p_total
# ---------------------------------------------------------------------------
.build_prior_vectors <- function(priors, dm) {
  expand <- function(spec, nms) .expand_prior(spec, nms)

  p_b0  <- expand(priors$b0,    dm$col_names_b0)
  p_b1  <- expand(priors$b1,    dm$col_names_b1)
  p_b2  <- expand(priors$b2,    dm$col_names_b2)
  p_om  <- expand(priors$omega, dm$col_names_om)
  p_rho <- expand(priors$rho,   dm$col_names_rho)

  all_rows <- rbind(p_b0, p_b1, p_b2, p_om, p_rho)

  list(
    mean = all_rows$mean,
    sd   = all_rows$sd,
    lb   = all_rows$lb,
    ub   = all_rows$ub,
    names = all_rows$name,
    # Also return per-parameter names for labelling
    names_b0  = p_b0$name,
    names_b1  = p_b1$name,
    names_b2  = p_b2$name,
    names_om  = p_om$name,
    names_rho = p_rho$name
  )
}

# ---------------------------------------------------------------------------
# Build the full parameter name vector for the draw matrix columns.
# Order: beta_b0 | u_b0 | beta_b1 | beta_b2 | beta_om | beta_rho | sigma | sigma_u
# ---------------------------------------------------------------------------
.param_names <- function(dm, pv) {
  b0_names  <- paste0("b0_", pv$names_b0)
  u_names   <- if (dm$n_groups_b0 > 0) paste0("u[", dm$group_levels_b0, "]") else character(0)
  b1_names  <- paste0("b1_", pv$names_b1)
  b2_names  <- paste0("b2_", pv$names_b2)
  om_names  <- paste0("omega_", pv$names_om)
  rho_names <- paste0("rho_", pv$names_rho)
  c(b0_names, u_names, b1_names, b2_names, om_names, rho_names, "sigma", "sigma_u")
}
