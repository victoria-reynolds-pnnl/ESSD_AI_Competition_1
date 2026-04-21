# this code chunk is not AI (lines 1-28)
## if you do not have these packages
## you will have to install them
# uncomment line 5-6 and run
#install.packages("tidyverse", "rstudioapi", "lme4",
#                 "glmmTMB", "DHARMa")

# libraries
#R packages needed for code to run

library(tidyverse) # data wrangling 
library(rstudioapi) # helps set path so works on any machine

# fit mixed models
library(lme4) # fit poisson 
library(glmmTMB) # fit neg bin & zero-inflated 
library(DHARMa) # fit diagnostics

## set working directory & get data
setwd(dirname(getActiveDocumentContext()$path))

df <- read.csv("../Data/region_yr_extreme_count.csv")

# set seed for reproducibility for dharma
x <- 123
set.seed(x) # check dharma code as well

# from here down is AI
# written with chatgpt gpt 5.2

# get cold snap rows only
# 1) Filter to cold types
df_cold <- df %>%
  mutate(
    NERC_ID      = factor(NERC_ID),
    extreme_type = factor(extreme_type),
    start_yr   = as.integer(start_yr)
  ) %>%
  filter(grepl("cold", as.character(extreme_type), 
               ignore.case = TRUE))

# Keep rows that do NOT contain "heat" in any factor column
df_cold <- df_cold[
  !apply(df_cold, 1, 
         function(row) any(grepl("heat", row, 
                                 ignore.case = TRUE))),
]

# Drop unused factor levels
df_cold <- droplevels(df_cold)

# 2) Create the full panel of NERC_ID x year x cold extreme_type
#    and fill missing combinations with n_extreme = 0
all_years <- 1980:2024

df_cold_complete <- df_cold %>%
  # ensure one row per cell before completing (in case of duplicates)
  group_by(NERC_ID, start_yr, extreme_type) %>%
  summarise(n_extreme = sum(n_extreme), .groups = "drop") %>%
  complete(
    NERC_ID,
    start_yr = all_years,
    extreme_type,
    fill = list(n_extreme = 0)
  ) %>%
  arrange(NERC_ID, extreme_type, start_yr) 

# 3) Recreate splits from the completed panel
train_df <- df_cold_complete %>% filter(start_yr >= 1980, start_yr <= 2012)
val_df   <- df_cold_complete %>% filter(start_yr >= 2013, start_yr <= 2018)
test_df  <- df_cold_complete %>% filter(start_yr >= 2019, start_yr <= 2024)

# Quick sanity checks
mean(train_df$n_extreme == 0)     # proportion zeros in training
range(train_df$n_extreme)         # should now include 0

# Hierarchical model: random intercept for region
form <- n_extreme ~ 1 + extreme_type + (1 | NERC_ID)
form_yr <- n_extreme ~ 1 + start_yr + extreme_type + (1 | NERC_ID)

### fit models
# ----------------------------
# 1) Fit four model types
# ----------------------------
fit_pois <- glmer(
  form, data = train_df,
  family = poisson(link = "log") #poisson regression
)

fit_nb <- glmmTMB(
  form, data = train_df,
  family = nbinom2(link = "log") # negative binomial
)

fit_zip <- glmmTMB(
  form, data = train_df,
  family = poisson(link = "log"), #zero inflated poisson
  ziformula = ~ 1
)

# Optional 4th (often useful): zero-inflated NB
fit_zinb <- glmmTMB(
  form, data = train_df,
  family = nbinom2(link = "log"),
  ziformula = ~ 1
)

# ----------------------------
# 2) DHARMa comparison helper
# ----------------------------
dharma_tests <- function(fit, 
                         name, 
                         n_sims = 1000, 
                         seed = x) {
  set.seed(seed)
  sim_res <- simulateResiduals(fittedModel = fit, 
                               n = n_sims)
  
  data.frame(
    model = name,
    dispersion_stat = unname(testDispersion(sim_res)$statistic),
    dispersion_p    = testDispersion(sim_res)$p.value,
    zi_stat         = unname(testZeroInflation(sim_res)$statistic),
    zi_p            = testZeroInflation(sim_res)$p.value,
    uniformity_stat = unname(testUniformity(sim_res)$statistic),
    uniformity_p    = testUniformity(sim_res)$p.value
  )
}

# ----------------------------
# 3) Run DHARMa tests for each
# ----------------------------
fits <- list(
  poisson      = fit_pois,
  negbin       = fit_nb,
  zi_poisson   = fit_zip,
  zi_negbin    = fit_zinb
)

diag_table <- imap_dfr(fits, ~ dharma_tests(.x, .y, 
                                            n_sims = 1000, seed = x))

diag_table %>%
  arrange(uniformity_p, dispersion_p, zi_p)  # small p = worse fit


# ----------------------------
# 4) (Optional) Save diagnostic plots for each model
# ----------------------------
save_dharma_plot <- function(fit, filename, n_sims = 1000, seed = 123,
                             width = 1600, height = 700, res = 200) {
  set.seed(seed)
  sim_res <- simulateResiduals(fittedModel = fit, n = n_sims)
  
  png(filename, width = width, height = height, res = res)
  plot(sim_res)
  dev.off()
  
  invisible(sim_res)
}

# Creates four PNGs 
# updated code for visualizations folder
save_dharma_plot(fit_pois, 
                 "../Visualizations/dharma_poisson.png")
save_dharma_plot(fit_nb,   
                 "../Visualizations/dharma_negbin.png")
save_dharma_plot(fit_zip,  
                 "../Visualizations/dharma_zip.png")
save_dharma_plot(fit_zinb, 
                 "../Visualizations/dharma_zinb.png")

### AIC and BIC model fit checks
## not written by AI
AIC(fit_nb, fit_zip, fit_pois, fit_zinb) # neg bin smallest
BIC(fit_nb, fit_zip, fit_pois, fit_zinb) #neg bin smallest

#----------------------------
  # Walk-forward evaluator
  # ----------------------------

form <- n_extreme ~ 1 + extreme_type + (1 | NERC_ID)

val_years <- 2013:2018

walkforward_mae <- function(data, model = c("nb", "zip"), val_years = 2013:2018) {
  model <- match.arg(model)
  
  preds <- map_dfr(val_years, function(y) {
    wf_train <- data %>% filter(start_yr >= 1980, start_yr <= (y - 1))
    wf_val   <- data %>% filter(start_yr == y)
    
    fit <- if (model == "nb") {
      glmmTMB(form, data = wf_train, family = nbinom2(link = "log"))
    } else {
      glmmTMB(form, data = wf_train, family = poisson(link = "log"), ziformula = ~ 1)
    }
    
    wf_val %>%
      mutate(pred_mean = predict(fit, newdata = wf_val, type = "response",
                                 allow.new.levels = TRUE))
  })
  
  tibble(
    model = model,
    wf_mae = mean(abs(preds$n_extreme - preds$pred_mean))
  )
}

mae_table <- bind_rows(
  walkforward_mae(df_cold_complete, model = "nb",  val_years = val_years),
  walkforward_mae(df_cold_complete, model = "zip", val_years = val_years)
) %>%
  arrange(wf_mae)

mae_table

# best model form: negative binomial

### now let's deal with thinking about time
### evaluate negative binomial models

# Hierarchical models: random intercept for region

## center time
train_df$c_year <- train_df$start_yr - mean(train_df$start_yr, na.rm=TRUE)
val_df$c_year <- val_df$start_yr - mean(val_df$start_yr, na.rm=TRUE)


form <- n_extreme ~ 1 + extreme_type + (1 | NERC_ID)
form_yr <- n_extreme ~ c_year + extreme_type + (1 | NERC_ID)
form_yr_rs <- n_extreme ~ c_year + extreme_type + (c_year | NERC_ID)
form_yr_rs2 <- n_extreme ~ c_year + extreme_type + (1+ c_year || NERC_ID)


fit_nb <- glmmTMB(form, 
                  data = train_df, 
                  family = nbinom2(link="log"))
fit_nb_yr <- glmmTMB(form_yr, 
                     data = train_df, 
                     family = nbinom2(link="log"))
fit_nb_yr_rs <- glmmTMB(form_yr_rs, 
                        data=train_df, 
                        family=nbinom2())

fit_nb_yr_rs2 <- glmmTMB(form_yr_rs2, 
                        data=train_df, 
                        family=nbinom2())

# fit checks
AIC(fit_nb, fit_nb_yr, fit_nb_yr_rs,fit_nb_yr_rs2) # fit_nb_yr smallest
BIC(fit_nb, fit_nb_yr, fit_nb_yr_rs, fit_nb_yr_rs2) # fit_nb smallest

anova(fit_nb_yr, fit_nb_yr_rs) # accept fit_nb_yr p = 0.6
anova(fit_nb, fit_nb_yr) # accept fit_nb p = 0.136

#### mae checks for all 3 models
mae <- function(y, yhat) mean(abs(y - yhat), na.rm = TRUE)

# put all data in one place for easy filtering
all_df <- bind_rows(
  train_df %>% mutate(split = "train"),
  val_df   %>% mutate(split = "val")
)

val_years <- 2013:2018

wf_mae <- lapply(val_years, function(y) {
  
  train_y <- all_df %>% filter(start_yr <= y - 1)   # expands forward each year
  val_y   <- val_df %>% filter(start_yr == y)
  
  # refit each model on the current training window
  fit_nb_y <- glmmTMB(form, data = train_y, family = nbinom2(link="log"))
  fit_nb_yr_y <- glmmTMB(form_yr, data = train_y, family = nbinom2(link="log"))
  fit_nb_yr_rs_y <- glmmTMB(form_yr_rs, data = train_y, family = nbinom2(link="log"))
  
  # population-level predictions (random effects set to 0)
  pred_nb       <- predict(fit_nb_y,       newdata = val_y, type = "response", re.form = NA)
  pred_nb_yr    <- predict(fit_nb_yr_y,    newdata = val_y, type = "response", re.form = NA)
  pred_nb_yr_rs <- predict(fit_nb_yr_rs_y, newdata = val_y, type = "response", re.form = NA)
  
  tibble(
    val_year = y,
    model = c("fit_nb", "fit_nb_yr", "fit_nb_yr_rs", "fit_nb_yr_rs2"),
    mae = c(
      mae(val_y$n_extreme, pred_nb),
      mae(val_y$n_extreme, pred_nb_yr),
      mae(val_y$n_extreme, pred_nb_yr_rs),
      mae(val_y$n_extreme, pred_nb_yr_rs)
    )
  )
})

mae_table <- bind_rows(wf_mae)

# optional: overall MAE across 2013-2018 for each model
mae_overall <- mae_table %>%
  group_by(model) %>%
  summarise(val_year = "2013-2018", 
            mae = mean(mae, na.rm = TRUE), 
            .groups = "drop")

# not written by AI
### write DHARMa diag table to Output folder
write.csv(diag_table, 
          "../Output/cold_model_dispersion.csv", 
          row.names = FALSE)
write.csv(mae_table,
          "../Output/cold_mae_val.csv",
          row.names = FALSE)
write.csv(mae_overall,
          "../Output/cold_mae_overall_val.csv",
          row.names = FALSE)

# write df_cold_complete datasets to output
# by split
write.csv(train_df,
          "../Output/cold_train_df.csv",
          row.names = FALSE)
write.csv(val_df,
          "../Output/cold_val_df.csv",
          row.names = FALSE)
write.csv(test_df,
          "../Output/cold_test_df.csv",
          row.names = FALSE)

