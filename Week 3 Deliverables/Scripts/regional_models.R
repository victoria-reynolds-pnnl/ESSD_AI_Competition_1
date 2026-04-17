# this code chunk is not AI (lines 1-37)
## if you do not have these packages
## you will have to install them
# uncomment line 5-6 and run
#install.packages("tidyverse", "rstudioapi", "glmmTMB", 
#                 "DHARMa", "broom.mixed")
#broom_mixed requires package rlang version 1.1.7

# libraries
#R packages needed for code to run
library(tidyverse) # data wrangling 
library(rstudioapi) # helps set path so works on any machine

# fit mixed models
library(glmmTMB) # fit models
library(DHARMa) # fit checks
library(broom.mixed) # extracts coefficients cleanly

## set working directory & get data
setwd(dirname(getActiveDocumentContext()$path))

train_df <- read.csv("../Output/cold_train_df.csv")
val_df <- read.csv("../Output/cold_val_df.csv")

train_df$data <- "train"
val_df$data <- "validate"

nerc_levels <- c(
  "NERC1","NERC2","NERC3","NERC4","NERC5","NERC6","NERC7","NERC8",
  "NERC9","NERC10","NERC11","NERC12","NERC15","NERC17","NERC18","NERC20"
)
train_df$NERC_ID <- factor(train_df$NERC_ID, levels = nerc_levels)
val_df$NERC_ID   <- factor(val_df$NERC_ID,   levels = nerc_levels)

train_df$c_year <- train_df$start_yr - mean(train_df$start_yr, na.rm=TRUE)
val_df$c_year <- val_df$start_yr - mean(val_df$start_yr, na.rm=TRUE)

# evaluate distinct regional models vs contuining combine model approach
## AI from here down
# gpt 5.2
# Fit one Negative Binomial model per NERC region (NERC_ID) using glmmTMB

## multiple failures to converge 
#form_mult <- n_extreme ~ c_year * extreme_type

form <- n_extreme ~ c_year + extreme_type

fit_region_clean <- function(dat, min_nonzero = 1) {
  dat <- dat %>%
    group_by(extreme_type) %>%
    filter(sum(n_extreme > 0, na.rm = TRUE) >= min_nonzero) %>%  # keep types with signal
    ungroup() %>%
    droplevels()
  
  glmmTMB(form, data = dat, family = nbinom2(link = "log"))
}

regional_models2 <- train_df %>%
  mutate(NERC_ID = factor(NERC_ID), extreme_type = factor(extreme_type)) %>%
  group_split(NERC_ID) %>%
  set_names(levels(factor(train_df$NERC_ID))) %>%
  map(~ fit_region_clean(.x, min_nonzero = 1))

## model convergence problem - some regions have too few observations

### try to fix by using regional quartile approach, rather than categorical
# treat as factors, not integers
train_df <- train_df %>%
  mutate(
    code   = str_extract(extreme_type, "\\d+"),
    dur_q  = as.factor(str_sub(code, 1, 1)),
    temp_q = as.factor(str_sub(code, 2, 2))
  )

val_df <- val_df %>%
  mutate(
    code   = str_extract(extreme_type, "\\d+"),
    dur_q  = as.factor(str_sub(code, 1, 1)),
    temp_q = as.factor(str_sub(code, 2, 2))
  )

###
form_final <- n_extreme ~ c_year + dur_q + temp_q
form_final2 <- n_extreme ~ c_year + I(c_year^2) + dur_q + temp_q

# create 16 models based on training dataset
regional_models_final <- train_df %>%
  mutate(NERC_ID = factor(NERC_ID)) %>%
  group_split(NERC_ID) %>%
  set_names(levels(factor(train_df$NERC_ID))) %>%
  map(~ glmmTMB(form_final, data = .x, family = nbinom2(link = "log")))

# create 16 models based on training dataset - quadratic form
regional_models_final2 <- train_df %>%
  mutate(NERC_ID = factor(NERC_ID)) %>%
  group_split(NERC_ID) %>%
  set_names(levels(factor(train_df$NERC_ID))) %>%
  map(~ glmmTMB(form_final2, data = .x, family = nbinom2(link = "log")))

# exponentiate coefficients from models
coef_rr_tbl <- imap_dfr(regional_models_final, \(mod, region) {
  broom.mixed::tidy(mod, effects = "fixed", conf.int = TRUE, exponentiate = TRUE) %>%
    mutate(NERC_ID = region, .before = 1)
})

###
val_pred <- purrr::imap_dfr(split(val_df, val_df$NERC_ID), \(dat, id) {
  m <- regional_models_final[[as.character(id)]]
  if (is.null(m)) return(NULL)
  dat$predicted_n <- as.numeric(predict(m, newdata = dat, type = "response"))
  dat
})

val_pred2 <- purrr::imap_dfr(split(val_df, val_df$NERC_ID), \(dat, id) {
  m <- regional_models_final2[[as.character(id)]]
  if (is.null(m)) return(NULL)
  dat$predicted_n <- as.numeric(predict(m, newdata = dat, type = "response"))
  dat
})

# val_pred (or val_df) must contain: NERC_ID, start_yr, n_extreme, predicted_n
plot_df <- val_pred %>%   # change to val_df if that's where predicted_n lives
  group_by(NERC_ID, start_yr) %>%
  summarise(
    observed  = sum(n_extreme,   na.rm = TRUE),
    predicted = sum(predicted_n, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  pivot_longer(c(observed, predicted), names_to = "series", values_to = "count")

ggplot(plot_df, aes(x = start_yr, y = count, color = series, linetype = series)) +
  geom_line(linewidth = 0.8) +
  facet_wrap(~ NERC_ID, scales = "fixed") +
  scale_color_manual(values = c(observed = "black", predicted = "steelblue")) +
  scale_linetype_manual(values = c(observed = "solid", predicted = "dashed")) +
  labs(x = "Year", y = "Total extremes", color = NULL, linetype = NULL) +
  theme_bw()

ggsave("../Visualizations/extremes_val.png",
       width = 9, height =8)

### MAE review on the validation dataset - no quadratic
# 1) MAE at the row level (each observation)
mae_overall <- val_pred %>%
  summarise(MAE = mean(abs(n_extreme - predicted_n), na.rm = TRUE))

mae_by_region <- val_pred %>%
  group_by(NERC_ID) %>%
  summarise(
    MAE = mean(abs(n_extreme - predicted_n), na.rm = TRUE),
    n = sum(!is.na(n_extreme) & !is.na(predicted_n)),
    .groups = "drop"
  ) %>%
  arrange(desc(MAE))

mae_overall
mae_by_region

### quadratic
### MAE review on the validation dataset - no quadratic
# mae_overall2 same as mae_overall
#  MAE at the row level (each observation)
mae_overall2 <- val_pred2 %>%
  summarise(MAE = mean(abs(n_extreme - predicted_n), na.rm = TRUE))

mae_overall2$MAE - mae_overall$MAE # zero - exactly same

### only running on regional models no quadratic
# Run DHARMa simulated residuals + standard tests per region
dharma_res <- imap(regional_models_final, function(mod, region){
  
  sim <- simulateResiduals(fittedModel = mod, n = 1000)  # increase n for more power
  
  list(
    region = region,
    sim = sim,
    tests = list(
      uniformity   = testUniformity(sim),
      dispersion   = testDispersion(sim),
      zeroInflation= testZeroInflation(sim),
      outliers     = testOutliers(sim)
    )
  )
})

# Look at p-values summary
dharma_pvals <- imap_dfr(dharma_res, function(x, nm){
  data.frame(
    NERC_ID = nm,
    uniformity_p    = x$tests$uniformity$p.value,
    dispersion_p    = x$tests$dispersion$p.value,
    zeroInflation_p = x$tests$zeroInflation$p.value,
    outliers_p      = x$tests$outliers$p.value
  )
})

# continue refining combined model approach
# compare random effect to fixed factor for region
m1 <- glmmTMB(n_extreme ~ c_year + code + (1 | NERC_ID) ,
              data = train_df, family = nbinom2(link="log"))

m1.1 <- glmmTMB(n_extreme ~ c_year * code + (1 | NERC_ID) ,
                 data = train_df, family = nbinom2(link="log"))
m2 <- glmmTMB(n_extreme ~ c_year + code + as.factor(NERC_ID) ,
              data = train_df, family = nbinom2(link="log"))

m2.1 <- glmmTMB(n_extreme ~ c_year * code + as.factor(NERC_ID) ,
              data = train_df, family = nbinom2(link="log"))

# m3 did not converge
m3 <- glmmTMB(n_extreme ~ c_year * code + (1 + c_year | NERC_ID),
                data=train_df, family=nbinom2(link="log"))
# m3.1 did not converge
m3.1 <- glmmTMB(n_extreme ~ c_year + code + (1 + c_year | NERC_ID),
              data=train_df, family=nbinom2(link="log"))

# models with quadratic terms
# m4 version of m1: linear + quadratic year, additive with code, random intercept by region
m4 <- glmmTMB(
  n_extreme ~ c_year + I(c_year^2) + code + (1 | NERC_ID),
  data = train_df,
  family = nbinom2(link = "log")
)

# m4 version of m1.1: allow code-specific linear and quadratic trends, random intercept by region
#failed to converge
m4.1 <- glmmTMB(
  n_extreme ~ (c_year + I(c_year^2)) * code + (1 | NERC_ID),
  data = train_df,
  family = nbinom2(link = "log")
)

# m4 version of m2: linear + quadratic year, additive with code, fixed effects for region
m4.2 <- glmmTMB(
  n_extreme ~ c_year + I(c_year^2) + code + as.factor(NERC_ID),
  data = train_df,
  family = nbinom2(link = "log")
)

# m4 version of m2.1: code-specific linear and quadratic trends, fixed effects for region
#failed to converge
m4.3 <- glmmTMB(
  n_extreme ~ (c_year + I(c_year^2)) * code + as.factor(NERC_ID),
  data = train_df,
  family = nbinom2(link = "log")
)

### m5 models
m5 <- glmmTMB(
     n_extreme ~ c_year * code + (1 | NERC_ID) + (1 | NERC_ID:start_yr),
     data = train_df,
   family = nbinom2(link = "log")
)

m5.1 <- glmmTMB(
  n_extreme ~ c_year + code + (1 | NERC_ID) + (1 | NERC_ID:start_yr),
  data = train_df,
  family = nbinom2(link = "log")
)

m5.2 <- glmmTMB(
     n_extreme ~ c_year * code + (1 | NERC_ID) + (1 | start_yr),
     data = train_df,
     family = nbinom2(link = "log")
) 

m5.3 <- glmmTMB(
  n_extreme ~ c_year + code + (1 | NERC_ID) + (1 | start_yr),
  data = train_df,
  family = nbinom2(link = "log")
) 

m5.4 <- glmmTMB(
  n_extreme ~ c_year * code + as.factor(NERC_ID) + (1 | start_yr),
  data = train_df,
  family = nbinom2(link = "log")
) 

## compare m1 and m2 and m3 versions
mods <- list(m1 = m1, `m1.1` = m1.1, m2 = m2, `m2.1` = m2.1, 
             m3 = m3, `m3.1` = m3.1, 
             `m4` = m4,`m4.1` = m4.1,`m4.2` = m4.2,`m4.3` = m4.3,
             `m5` = m5,`m5.1` = m5.1, `m5.2` = m5.2,`m5.3` = m5.3,
             `m5.4` = m5.4)

a <- AIC(m1, m1.1, m2, m2.1, m3, m3.1, 
         m4, m4.1,m4.2, m4.3, m5, m5.1, m5.2, m5.3, m5.4) 
b <- BIC(m1, m1.1, m2, m2.1, m3, m3.1, 
         m4, m4.1,m4.2, m4.3, m5, m5.1, m5.2, m5.3, m5.4) 

ic_df <- merge(
  data.frame(model = rownames(a), df = a$df, AIC = a$AIC, row.names = NULL),
  data.frame(model = rownames(b), df = b$df, BIC = b$BIC, row.names = NULL),
  by = c("model", "df"),
  all = TRUE
)
# add model specification string
spec_tbl <- imap_dfr(mods, ~ tibble(
  model = .y,
  fit   = paste(deparse(formula(.x)), collapse = " ")
))
ic_df <- ic_df %>% left_join(spec_tbl, by = "model")

# add preference columns
best_aic_model <- ic_df$model[which.min(ic_df$AIC)]
best_bic_model <- ic_df$model[which.min(ic_df$BIC)]

ic_df <- ic_df %>%
  mutate(
    AIC_pref = if_else(model == best_aic_model, "best model", ""),
    BIC_pref = if_else(model == best_bic_model, "best model", "")
  )

## not AI
fail_cnv <- c("m3","m3.1","m4.1")
ic_df$notes <- ifelse(ic_df$model %in% fail_cnv, 
                            "failed to converge", NA)
# AI
#### compare 4 candidate models
val_comp <- val_df %>%
  mutate(
    pred_m5.2 = predict(m5.2, newdata = ., type = "response", 
                        allow.new.levels = TRUE),
        pred_m5.4 = predict(m5.4, newdata = ., type = "response", 
                            allow.new.levels = TRUE),
    pred_m2   = predict(m2.1,   newdata = ., type = "response"),
    pred_m1.1 = predict(m1.1, newdata = ., type = "response")
  )

# mae 5.4 slightly better than mae 5.2
# mae 5.2 slightly better than mae 2
#mae 2 slightly better than MAE m1.1
val_comp_overall <- val_comp %>%
  summarise(
    MAE_m5.4   = mean(abs(n_extreme - pred_m5.4),   na.rm = TRUE),
    MAE_m5.2   = mean(abs(n_extreme - pred_m5.2),   na.rm = TRUE),
    MAE_m2   = mean(abs(n_extreme - pred_m2),   na.rm = TRUE),
    MAE_m1.1 = mean(abs(n_extreme - pred_m1.1), na.rm = TRUE)
  )

val_comp_reg <- val_df %>%
  mutate(
    pred_m5.2 = as.numeric(predict(m5.2, newdata = ., type = "response",
                                   allow.new.levels = TRUE)),
    pred_m5.4 = as.numeric(predict(m5.4, newdata = ., type = "response",
                                   allow.new.levels = TRUE))
  )

mae_by_region <- val_comp_reg %>%
  group_by(NERC_ID) %>%
  summarise(
    MAE_m5.2 = mean(abs(n_extreme - pred_m5.2), na.rm = TRUE),
    MAE_m5.4 = mean(abs(n_extreme - pred_m5.4), na.rm = TRUE),
    n = sum(!is.na(n_extreme)),
    .groups = "drop"
  ) %>%
  mutate(delta_MAE = MAE_m5.4 - MAE_m5.2) %>%  # negative => m5.4 better
  arrange(MAE_m5.4)

mae_by_region

### write output
write.csv(mae_by_region, 
          "../Output/mae_regional_models.csv",
          row.names = FALSE)
write.csv(dharma_pvals, 
          "../Output/dharma_pvals_val_df.csv",
          row.names = FALSE)
write.csv(coef_rr_tbl, "../Output/coefficients_rr_tbl.csv",
          row.names = FALSE)
write.csv(val_comp, 
          "../Output/val_comp_mae.csv",
          row.names = FALSE)

write.csv(val_comp_overall, 
          "../Output/val_comp_mae_overall.csv",
          row.names = FALSE)

write.csv(ic_df, 
          "../Output/ic_df_combined_model.csv",
          row.names = FALSE)

write.csv(mae_by_region, 
          "../Output/va_comp_mae_region.csv",
          row.names = FALSE)

