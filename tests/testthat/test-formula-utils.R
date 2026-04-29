test_that(".parse_re extracts random-effect group correctly", {
  fml <- ~ 1 + age + sex + (1 | subject)
  result <- smoothbp:::.parse_re(fml)

  expect_equal(result$re_group, "subject")
  # Fixed formula should not contain the RE term
  fixed_terms <- attr(terms(result$fixed), "term.labels")
  expect_false(any(grepl("\\|", fixed_terms)))
  expect_true("age" %in% fixed_terms)
  expect_true("sex" %in% fixed_terms)
})

test_that(".parse_re returns NULL re_group when no RE present", {
  fml <- ~ 1 + age
  result <- smoothbp:::.parse_re(fml)
  expect_null(result$re_group)
})

test_that(".parse_re errors on multiple RE terms", {
  fml <- ~ 1 + (1 | a) + (1 | b)
  expect_error(smoothbp:::.parse_re(fml), regexp = "at most one")
})

test_that(".build_design_matrices returns correct dimensions", {
  set.seed(1)
  n <- 30
  dat <- data.frame(
    y   = rnorm(n),
    tau = seq(0, 5, length.out = n),
    grp = rep(c("A", "B"), each = n / 2),
    sub = rep(1:10, times = 3)
  )
  dm <- smoothbp:::.build_design_matrices(
    b0_fml    = ~ 1 + grp + (1 | sub),
    b1_fml    = ~ 1 + grp,
    b2_fml    = ~ 1,
    omega_fml = ~ 1,
    rho_fml   = ~ 1,
    data = dat
  )
  expect_equal(nrow(dm$X_b0), n)
  expect_equal(ncol(dm$X_b0), 2)   # intercept + grpB
  expect_equal(ncol(dm$X_b1), 2)
  expect_equal(ncol(dm$X_b2), 1)
  expect_equal(dm$n_groups_b0, 10)
  expect_length(dm$group_b0, n)
  expect_true(all(dm$group_b0 >= 0L & dm$group_b0 < 10L))
})
