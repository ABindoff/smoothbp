library(tidyverse)

demographics <- readRDS('demographic.rds')
d1 <- readRDS('raw_long.rds')

d2 <- d1 |>
  dplyr::filter(name %in% c(
    'bnt_raw',
    'ravlt_t15_tot',
    'lmi_unit_tot',
    'lmii_unit_tot',
    'wais_ds_raw',
    'wais_lns_raw',
    'wais_voc_raw',
    'wais_com_raw',
    'stroop_c_time',
    'tmt_b_time',
    'ssp_len',
    'swm_be',
    'rvp_a',
    'pal_ftm_score'
  )) |>
  mutate(
    IDCode = factor(IDCode),
    name = factor(name)
  ) |>
  dplyr::filter(!is.na(value)) |>
  select(
    IDCode, Age_t1, Sex, wtar_fsiq_baseline, education_total, Group,
    name, value, phase, years_since_baseline,
    hads_anx_baseline, hads_dep_baseline, drs_tot_raw_baseline
  )

d3 <- d2 |>
  pivot_wider(
    id_cols = c(
      'IDCode', 'Age_t1', 'Sex', 'wtar_fsiq_baseline', 'education_total',
      'Group', 'phase', 'years_since_baseline',
      'hads_anx_baseline', 'hads_dep_baseline', 'drs_tot_raw_baseline'
    ),
    names_from = name,
    values_from = value
  )

# Z-score using baseline assessment means and SDs from Thow 2018
zsc <- function(x, m, sd) (x - m) / sd

mean_thow <- c(53.14, 48.31, 30.15, 18.77, 11.67, 56.90,
               26.15, 57.68, 25.94, 59.02, 18.35,  5.76, 25.63, .91)

sd_thow   <- c( 8.86,  8.30,  6.41,  3.91,  2.39,  5.78,
                3.41,  2.90,  7.53, 19.67,  3.35,  1.20, 18.58, .05)

test_cols <- names(d3)[12:25]
for (i in seq_along(test_cols)) {
  d3[[test_cols[i]]] <- zsc(d3[[test_cols[i]]], m = mean_thow[i], sd = sd_thow[i])
}

# Composite cognitive domain scores
d3$episodic  <-  .356 * d3$lmi_unit_tot  +
  .346 * d3$lmii_unit_tot  +
  .305 * d3$ravlt_t15_tot  +
  .245 * d3$pal_ftm_score

d3$working   <-  .397 * d3$wais_lns_raw  +
  .376 * d3$wais_ds_raw    +
  .325 * d3$ssp_len        +
  -.306 * d3$swm_be

d3$executive <-  .439 * d3$stroop_c_time +
  .424 * d3$tmt_b_time     +
  -.460 * d3$rvp_a

d3$language  <-  .360 * d3$bnt_raw       +
  .442 * d3$wais_com_raw   +
  .477 * d3$wais_voc_raw

d4 <- d3 |>
  select(
    IDCode, Age_t1, Sex, wtar_fsiq_baseline, education_total, Group,
    phase, years_since_baseline,
    hads_anx_baseline, hads_dep_baseline, drs_tot_raw_baseline,
    episodic, working, executive, language
  ) |>
  pivot_longer(cols = c('episodic', 'working', 'executive', 'language')) |>
  dplyr::filter(!is.na(value)) |>
  mutate(name = factor(name))

episodic  <- d4 |> dplyr::filter(name == 'episodic')
language  <- d4 |> dplyr::filter(name == 'language')
working   <- d4 |> dplyr::filter(name == 'working')
# Flip executive so that higher = better
executive <- d4 |> dplyr::filter(name == 'executive') |> mutate(value = -value)


library(smoothbp)

episodic %>% filter(is.na(years_since_baseline)) %>% View()
episodic <- episodic %>%
  mutate( y = value,
          tau = years_since_baseline,
          subject = IDCode) %>%
  na.omit('years_since_baseline')

fit <- smoothbp(
  formula = value ~ years_since_baseline,
  b0      = ~ 1 + Group + (1 | IDCode),   # random intercept per subject
  b1      = ~ 1 + Group,
  b2      = ~ 1 + Group,
  omega   = ~ 1,
  rho     = ~ 1,
  data    = episodic,
  priors  = smoothbp_priors(
    b0    = prior_normal(0, 10),
    b1    = prior_normal(0, 5),
    b2    = prior_normal(0, 5),
    omega = list(
      "(Intercept)"    = prior_normal(3, 2, lb = 0.1, ub = 6)
    ),
    rho   = prior_normal(3, 1, lb = 1),
    sigma = prior_invgamma(shape = 2, scale = 1)
  ),
  chains  = 4L,
  cores = 2L,
  iter    = 5000L,
  warmup  = 1500L,
  seed    = 42L
)

summary(fit, effects = 'fixed')
plot(fit)
hypothesis(fit,
           c('b1_GroupExperimental > 0',
             'b1_GroupExperimental + b2_GroupExperimental > 0',
             '`omega_(Intercept)` > 0'))
fig(fit)



fig <- function(m, d = episodic, ylab = 'Estimate', pal = c('#277975', '#863E17')) {
  p <- expand.grid(
    Group = c('Control', 'Experimental'),
    years_since_baseline = seq(0, 9, length.out = 36),
    Age_t1 = mean(d$Age_t1, na.rm = TRUE),
    hads_dep_baseline = mean(d$hads_dep_baseline, na.rm = TRUE),
    hads_anx_baseline = mean(d$hads_anx_baseline, na.rm = TRUE),
    drs_tot_raw_baseline = mean(d$drs_tot_raw_baseline, na.rm = TRUE),
    wtar_fsiq_baseline = mean(d$wtar_fsiq_baseline, na.rm = TRUE),
    Sex = 'Female',
    education_total = mean(d$education_total, na.rm = TRUE)
  )
  
  fit <- fitted(m, newdata = p, re_formula = NA, robust = TRUE)
  p <- cbind(p, fit)
  p$Group <- factor(p$Group,
                    levels = c('Control', 'Experimental'),
                    labels = c('Control', 'Education'))
  offset <- p$fitted_mean[1]
  p <- p |>
    mutate(
      Estimate = fitted_mean - offset,
      Q2.5 = fitted_Q2.5 - offset,
      Q97.5 = fitted_Q97.5 - offset
    )
  
  ggplot(
    p,
    aes(
      x = years_since_baseline,
      y = Estimate,
      ymin = Q2.5,
      ymax = Q97.5,
      group = Group,
      colour = Group,
      fill = Group
    )
  ) +
    geom_ribbon(colour = NA, alpha = .2) +
    geom_line() +
    theme_bw() +
    ylab(ylab) +
    xlab('Assessment stage (years)') +
    scale_colour_manual(values = pal) +
    scale_fill_manual(values = pal) +
    scale_x_continuous(
      breaks = c(0, 1, 2, 3, 5, 7, 9),
      labels = c(1, 2, 3, 4, 6, 8, 10)
    )
}
