# save_coefs_m54.R
# Called by train.py after test.R has run.
# Fits the final m5.4 model on train+val combined data,
# saves RDS and coefficient CSV for use by Python.
# Working directory must be Scripts/ when this is called.
# No rstudioapi dependency — safe to call from subprocess.

library(glmmTMB)
library(tidyverse)

# Load split data written by regional_time_cold_zeros.R
train_df <- read.csv("../Output/cold_train_df.csv")
val_df   <- read.csv("../Output/cold_val_df.csv")

# Set NERC factor levels (matches test.R)
nerc_levels <- c(
  "NERC1","NERC2","NERC3","NERC4","NERC5","NERC6","NERC7","NERC8",
  "NERC9","NERC10","NERC11","NERC12","NERC15","NERC17","NERC18","NERC20"
)
train_df$NERC_ID <- factor(train_df$NERC_ID, levels = nerc_levels)
val_df$NERC_ID   <- factor(val_df$NERC_ID,   levels = nerc_levels)

# Combine train + val (mirrors m5.4tv in test.R)
longer_df        <- rbind(train_df, val_df)
c_year_mean      <- mean(longer_df$start_yr, na.rm = TRUE)
longer_df$c_year <- longer_df$start_yr - c_year_mean

cat("Fitting m5.4tv on train+val data ...\n")
fit_m54tv <- glmmTMB(
  n_extreme ~ c_year * extreme_type + as.factor(NERC_ID) + (1 | start_yr),
  data   = longer_df,
  family = nbinom2(link = "log")
)

# Save R model object
dir.create("../Models", showWarnings = FALSE)
saveRDS(fit_m54tv, "../Models/fit_m54tv.rds")
cat("Saved: Models/fit_m54tv.rds\n")

# Extract fixed-effect coefficients
coef_tab             <- as.data.frame(coef(summary(fit_m54tv))$cond)
coef_tab$term        <- rownames(coef_tab)
coef_tab$c_year_mean <- c_year_mean
coef_tab$dispersion  <- sigma(fit_m54tv)
coef_tab$ref_extreme <- levels(factor(longer_df$extreme_type))[1]
coef_tab$ref_nerc    <- nerc_levels[1]

write.csv(coef_tab, "../Output/fit_m54_coefs.csv", row.names = FALSE)
cat("Saved: Output/fit_m54_coefs.csv\n")
cat("c_year_mean:", c_year_mean, "\n")
cat("Dispersion (size):", sigma(fit_m54tv), "\n")
cat("Reference extreme_type:", levels(factor(longer_df$extreme_type))[1], "\n")
cat("Reference NERC_ID:", nerc_levels[1], "\n")
