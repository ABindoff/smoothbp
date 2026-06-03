# Peer Review: *smoothbp: Fast Bayesian Hierarchical Piecewise Regression with Smoothed Transitions and Spike-and-Slab Model Selection*

**Reviewer:** colleague pre-submission review  
**Target journal:** Journal of Statistical Software  
**Date:** 2026-05-28  
**Package version reviewed:** 0.2.1  

---

## Summary

This paper introduces **smoothbp**, an R package for Bayesian hierarchical piecewise regression with logistic-smoothed change-points. The package occupies a genuine and well-argued niche between general-purpose probabilistic programming languages (brms/Stan) and existing specialised tools (mcp/JAGS): it offers a turnkey formula interface for models with multiple smoothed transitions, random effects on change-point timing, and Kuo & Mallick spike-and-slab breakpoint selection, backed by a compiled Rust sampler that exploits conjugate structure for linear parameters.

The core technical contribution is real and the implementation is clearly well-engineered. The vignettes in particular are of high quality — the four-scenario validation against brms and mcp is exactly the kind of reproducible comparison that JSS referees will want to see. The package tests are thoughtful and include parameter-recovery tests that are often absent from statistical software papers.

That said, there are several issues — one of them critical — that must be addressed before submission. The most important is a discrepancy between the manuscript's central model equation and the actual implementation. I detail all issues below in roughly decreasing order of urgency.

---

## Major Comments

### 1. The manuscript's key equation does not match the implementation (CRITICAL)

The manuscript's equation (Section 3.1, Eq. 1) writes the baseline slope term as:

$$b_1 \tau_{ij}$$

But the actual implementation in `src/rust/src/model.rs`, `src/rust/src/sampler.rs`, `simulate.R`, and the getting-started vignette all use:

$$b_1 (\tau_{ij} - \omega_{1j})$$

That is, the pre-break slope is *centred at the first breakpoint*, not at the origin. The simulate.R roxygen documents this explicitly: *"d_{ij} = τ_{ij} − ω"* is the centred displacement that enters both the b1 and δ terms. The vignette's equation is also correct:

$$\mu_i = b_{0i} + b_{1i}(\tau_i - \omega_{1i}) + \sum_{k=1}^K \delta_{ki}(\tau_i - \omega_{ki})\,\text{logistic}\!\left[(\tau_i - \omega_{ki})\rho_{ki}\right]$$

The consequence of this centering is that `b0` represents the conditional mean at the *first change-point* (when all δ terms vanish there), not the intercept at τ = 0. This is a deliberate and sensible parameterisation choice, but the manuscript equation 1 must be corrected to reflect it. The discrepancy would confuse any reader who tries to reproduce the model from the paper, and it would constitute an error of record in a JSS article.

**Required action:** Correct Eq. 1 and add a sentence explaining the centering choice and its interpretation. Note that this also means the `b0` interpretation stated in Section 3.1 ("b_0 is the intercept") is imprecise — it is the level of the mean function at τ = ω_{1j}, which is worth clarifying explicitly.

---

### 2. Section 6 (Benchmarking) reports no actual numbers

Section 6 makes quantitative claims — "delta mean ≈ 0.1%", "consistently generates higher Effective Sample Sizes per second" — without a single table or figure. These numbers are presented as if they were fixed facts, but ESS/sec is highly machine-, dataset-, and configuration-dependent. The detailed quantitative comparison *does* exist, in the excellent `brms-comparison` vignette, but the manuscript itself provides no reproducible evidence.

JSS papers must be self-contained enough that a referee can evaluate the scientific claims from the paper alone. A table summarising the scenario-1 comparison (point estimates, ESS/sec, wall-clock time) taken directly from the vignette output is the minimum needed here.

**Required action:** Add at least one reproducible comparison table (or figure) to Section 6. The brms-comparison vignette already contains the code; the manuscript just needs a representative summary of its output. The "0.1%" figure should either be grounded in a shown table or removed — it is unusually precise for a claim made without citation.

---

### 3. No figures anywhere in the manuscript

The manuscript contains no figures or tables at all. For a software paper introducing a new visualisation-capable package, this is a notable omission. JSS papers typically include at minimum: (a) a figure demonstrating the model fit on representative data, (b) a convergence diagnostic panel, and (c) the comparison plot from benchmarking.

The vignettes contain beautiful plots (overlaid posteriors, PIP bar charts, trajectory comparisons). Several of these could be included in the manuscript with minimal effort.

**Required action:** Add at least one figure to the manuscript. Suggested candidates: the posterior-density overlay from Scenario 1 of the brms comparison, or the PIP plot from the spike-and-slab vignette.

---

### 4. Ordering constraints on multiple change-points are not discussed

For multi-breakpoint models the manuscript does not address the label-switching / ordering problem: if the model has two change-points ω₁ and ω₂ with no ordering constraint, the posterior can be multimodal with the two ωs swapped. This is a classical identification challenge in mixture and change-point models. The advanced-modeling vignette uses tightly constrained priors (`lb`/`ub`) to avoid this, but there is no discussion of the issue in the manuscript and no user-facing guidance about when ordering constraints are necessary.

**Required action:** Add a short paragraph (or sidebar) in Section 3 or Section 5 explaining the label-switching issue and the recommended mitigation (non-overlapping bounded priors via `space_omega_priors()` or explicit `lb`/`ub` settings). The warning already present in the `brms-comparison` vignette about the "unidentifiable omega > max(tau) trap" is a related point worth surfacing in the paper.

---

## Moderate Comments

### 5. The claim of uniqueness in Appendix A is too strong

Appendix A concludes that "Bayesian multi-breakpoint hierarchical models with built-in spike-and-slab are uniquely realized in smoothbp." This is only true of *turnkey, formula-driven* implementations. A user of brms, PyMC, or Turing.jl can trivially implement such a model. The claim should be scoped to something like: "no existing package provides a turnkey formula interface combining all three features (hierarchical timing, logistic smoothing, and spike-and-slab selection) without requiring the user to code the model geometry by hand." That is both accurate and sufficient to motivate the package.

---

### 6. Default prior for rho is not discussed or justified

The default `rho = prior_normal(3, 2, lb = 0)` places substantial prior mass on rho < 1, which corresponds to very gradual, spread-out transitions. For many applications (intervention analyses, sudden regime shifts) this prior will be strongly regularising toward implausibly diffuse transitions. Conversely, the default lower bound of 0 is correct but the upper boundary is Inf, meaning the prior assigns non-trivial probability to rho > 10 (near-instantaneous kinks).

JSS expects Bayesian software papers to justify or at least characterise their default priors. A brief sensitivity analysis or a figure showing the implied transition shape under the default prior would be valuable here.

---

### 7. `update.smoothbp_fit` silently resets tuning parameters

In `R/smoothbp.R`, the `update()` method documents that `step_om`, `step_rho`, and `target_accept` are not stored in the fit object and silently revert to their defaults (0.3, 0.3, 0.65) on re-fit. This is a usability pitfall: a user who carefully tuned the sampler for a difficult model will unknowingly lose that tuning when using `update()`. The code comment acknowledges this but it is not mentioned in any documentation or vignette.

**Required action:** Either store these tuning parameters in the fit object (preferred) or add an explicit warning to the `update()` documentation noting that sampler tuning is reset.

---

### 8. `simulate_smoothbp()` only supports a single breakpoint

The package is explicitly designed for *multiple* change-points, but `simulate_smoothbp()` only accepts a single `omega`, `rho`, and `delta`. Users wanting to validate a two-breakpoint model — as done in the brms-comparison vignette — must write their own data-generating code by hand (as the vignette does). This inconsistency is particularly noticeable because the vignette's multi-breakpoint simulation code is duplicated in two separate vignettes.

**Suggested action:** Extend `simulate_smoothbp()` to accept vectors/lists of `omega`, `rho`, and `delta` for multi-breakpoint scenarios, or document the limitation explicitly and point users to the vignette code.

---

### 9. The "conjugate Gibbs eliminates the funnel" claim needs qualification

Section 4 states that the joint conjugate update for b0 and all u_{0j} "completely eliminates the infamous hierarchical funnel pathology." This is partially correct — sampling b0 and u jointly does avoid the slow-mixing ridge between them — but the funnel pathology in Neal's sense arises from the non-centred parameterisation, not merely from blocked vs. element-wise updates. If sigma_u is small (strong shrinkage), the centred parameterisation used here (u ~ N(0, sigma_u²)) can still exhibit funnel-like geometry in the (u, sigma_u) joint space. The claim should be softened to something like "avoids the slow-mixing correlation between the population intercept and group-level deviations."

---

## Minor Comments

### 10. Mathematical notation inconsistencies

- In Eq. 1 of the manuscript, the sharpness is written as `ρ_k(τ_{ij} − ω_{kj})` inside the sigmoid argument, but in Section 3.3 the Kuo & Mallick specification writes the slab prior for `β_{δk}`, introducing a new symbol (`β`) not used elsewhere. Aligning notation across sections would help.
- The sigmoid is denoted `σ(·)` in the model equation, but the same symbol is conventionally used for the standard deviation in many Bayesian contexts. Using `S(·)` or `logistic(·)` would avoid ambiguity.
- The subscription convention mixes `ij` (for subject × time) and `k` (for breakpoint). A single notation paragraph at the start of Section 3 defining all indices would aid readability.

### 11. The Bacon and Watts (1971) citation is missing from the manuscript

The DESCRIPTION file credits Bacon and Watts (1971) as the original source for the smooth transition piecewise regression model, but this citation does not appear in the manuscript's reference list. JSS papers must be self-contained; the intellectual lineage of the core model should be acknowledged in the text.

### 12. Installation requirements should be mentioned prominently

`smoothbp` requires Cargo (Rust's package manager) and `rustc >= 1.65.0` at install time — a non-trivial system requirement that distinguishes it from pure-R packages. The manuscript should mention this in Section 5 or an installation section. The DESCRIPTION already lists it under `SystemRequirements`, but many users will encounter the requirement for the first time when `install.packages()` fails.

### 13. The `hierarchical` argument is described as legacy but is still in the public API

The comment in `smoothbp.R` notes that `hierarchical` is a "legacy argument, now mostly auto-detected from (1|group) formula syntax." If it is legacy, it should be formally deprecated with a `.Deprecated()` call, or removed from the API entirely before JSS submission. Leaving undocumented legacy arguments in a published API creates long-term maintenance debt.

### 14. "Infinite-acceleration kinks" is non-standard terminology

The Discussion refers to "mathematically unrealistic infinite-acceleration kinks" for hard change-points. The standard term is simply "instantaneous transitions" or "hard change-points." "Infinite-acceleration" is evocative but physicists might object, and the phrase does not appear in the literature being cited.

### 15. The manuscript references the `pip()` function without explaining the output format

Section 5 shows `pip(fit_ss)` but does not explain what the returned object is or how to interpret the column names (e.g., `gamma_delta1_(Intercept)` is an unusual label for a publication-facing output). The vignette provides this interpretation table; a brief sentence in Section 5 would suffice.

### 16. Model comparison via LOO/WAIC is absent from the manuscript

Section 7 (Discussion) does not mention the LOO-IC or bridge-sampling model comparison tools that are part of the package API and demonstrated in the vignettes. Given that the spike-and-slab PIP and LOO are complementary model-selection tools, at least one sentence about this should appear in the manuscript.

### 17. The `b2` naming in simulation code may confuse readers

The brms-comparison vignette uses `b2` as the brms non-linear parameter name for what smoothbp calls `delta1`. Meanwhile, `simulate_smoothbp()` uses `delta` (not `b2`). The getting-started vignette then uses `b2` again in `simulate_smoothbp(b2 = 1.2, ...)` which is not a documented argument. Searching the vignette for `b2 = ` in the `smoothbp()` call to `simulate_smoothbp` reveals this is actually calling the `delta` argument with a positional mismatch in the getting-started vignette — this should be verified and corrected if so.

---

## Comments on the Code

The code is generally well-structured and the Rust implementation is clean. A few specific observations:

**model.rs / sampler.rs** — The numerically stable sigmoid implementation (using the `e/(1+e)` branch for negative inputs) is a good practice. The `log_truncated_normal_prior` function correctly handles the truncation normalisation constant, including the degenerate case where `cdf_u - cdf_l ≈ 0`.

**formula_utils.R** — This is the most complex R-side component. Thorough unit tests exist for it (`test-formula-utils.R`), which is important. The `test-gradients.R` file is particularly valuable for a bespoke HMC implementation; confirming that the Rust automatic differentiation agrees with finite differences is exactly the kind of validation reviewers will ask about.

**postprocess.R / postprocess_ss.R** — The use of the `posterior` package's `draws_array` format as the canonical storage type is an excellent design choice that gives users immediate access to the full posterior ecosystem.

**update.smoothbp_fit** — As noted above, `seed` is not stored in the fit object but is included in the `missing()` check. The fallback silently draws a new random seed, which is correct behaviour (you don't want to reproduce the same chain), but the missing `seed` slot means `object$seed` returns NULL rather than the original seed. Worth storing for reproducibility.

---

## Comments on the Vignettes

The vignette suite is the strongest part of the submission. The `brms-comparison.Rmd` vignette is exemplary: it specifies matching priors for both packages, accounts for the chain-initialisation pitfall that causes brms to go bimodal on this model, provides a quantitative ESS/sec comparison, and includes plots with truth overlaid. This vignette alone substantially de-risks the computational claims.

One suggestion: the `advanced-modeling.Rmd` vignette's market data example (NASDAQ stock prices) is engaging but uses unbalanced, real-world data with no ground truth. It would be more persuasive if accompanied by a simulation study at the same complexity level, or if the market example were moved to a "case study" section and the vignette's primary focus kept on a controlled synthetic example.

The `NOT_CRAN` pattern for conditional chunk evaluation is the right approach. Confirm that all long-running chunks are guarded by this flag before CRAN submission.

---

## Overall Recommendation

**Major revision required before submission.**

The package itself is technically sound and the niche is well-motivated. The core issues are presentational: the model equation in the manuscript must be corrected to match the implementation (item 1 above), the benchmarking section requires actual numbers (item 2), and at least one figure should be added (item 3). The label-switching discussion (item 4) is the only issue of scientific completeness beyond the equation error.

Items 5–17 are all addressable in a single revision pass and none are showstoppers. With those changes made, this paper would make a solid contribution to JSS and to the R ecosystem for interrupted time-series and change-point analysis.

---

*Review prepared after reading: manuscript, R source (all files in R/), Rust source (src/rust/src/), vignettes (getting-started, spike-and-slab, brms-comparison, advanced-modeling), testthat suite, DESCRIPTION, and NAMESPACE.*
