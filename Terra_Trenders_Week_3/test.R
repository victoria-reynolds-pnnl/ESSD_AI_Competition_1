# save_coefs.R
# Called by train.py after fit_nb_cold_train_val.R has run.
# Loads the train+val combined data, refits the final NB model,
# saves the RDS and coefficient table CSV for use by Python.
# Working directory must be Scripts/ when this is called.

library(glmmTMB)
library(tidyverse)

# Load split data written by regional_time_cold_zeros.R
train_df <- read.csv("../Output/cold_train_df.csv")
val_df   <- read.csv("../Output/cold_val_df.csv")

# Combine train + val (mirrors fit_nb_cold_train_val.R)
tr_val <- rbind(train_df, val_df)

# Rename extreme_type → cold and shorten labels (mirrors fit_nb_cold_train_val.R)
names(tr_val)[names(tr_val) == "extreme_type"] <- "cold"
tr_val$cold <- gsub("cold_dur_temp", "dt", tr_val$cold)

# Final model formula
form <- n_extreme ~ 1 + cold + (1 | NERC_ID)
fit_nb <- glmmTMB(form, data = tr_val, family = nbinom2(link = "log"))

# Save R model object
dir.create("../Models", showWarnings = FALSE)
saveRDS(fit_nb, "../Models/fit_nb.rds")
cat("Saved: Models/fit_nb.rds\n")

# Save coefficient table
coef_tab <- as.data.frame(coef(summary(fit_nb))$cond)
coef_tab$term       <- rownames(coef_tab)
coef_tab$dispersion <- sigma(fit_nb)
coef_tab$ref_level  <- levels(factor(tr_val$cold))[1]

write.csv(coef_tab, "../Output/fit_nb_coefs.csv", row.names = FALSE)
cat("Saved: Output/fit_nb_coefs.csv\n")
cat("Reference level:", levels(factor(tr_val$cold))[1], "\n")
cat("Dispersion (size):", sigma(fit_nb), "\n")
