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
library(glmmTMB) # fit neg bin & zero-inflated 

## set working directory & get data
setwd(dirname(getActiveDocumentContext()$path))

train_df <- read.csv("../Output/cold_train_df.csv")

val_df <- read.csv("../Output/cold_val_df.csv")

train_df$c_year <- train_df$start_yr - mean(train_df$start_yr, na.rm=TRUE)
val_df$c_year <- val_df$start_yr - mean(val_df$start_yr, na.rm=TRUE)


# from here down is AI
# written with chatgpt gpt 5.2

# split into quartiles again


train_df <- train_df %>%
  mutate(
    code   = str_extract(extreme_type, "\\d+"),
    dur_q  = as.integer(str_sub(code, 1, 1)),
    temp_q = as.integer(str_sub(code, 2, 2))
  )


#### NB check
# use extreme_type or fixed effects for duration & temp quartiles?

#### text form vs form_q
form <- n_extreme ~ 1 + extreme_type + (1 | NERC_ID)
form_q <- n_extreme ~ 1 + dur_q * temp_q + (1 | NERC_ID)
form_yr <- n_extreme ~ c_year + extreme_type + (1 | NERC_ID)
form_yr_q <- n_extreme ~ c_year * dur_q + temp_q + (1 | NERC_ID)


fit_nb <- glmmTMB(
  form, data = train_df,
  family = nbinom2(link = "log") # negative binomial
)

fit_nb_q <- glmmTMB(
  form_q, data = train_df,
  family = nbinom2(link = "log") # negative binomial
)

fit_nb_yr <- glmmTMB(
  form_yr, data = train_df,
  family = nbinom2()
)

fit_nb_yr_q <- glmmTMB(
  form_yr_q, 
  data = train_df,
  family = nbinom2()
)


# fit checks
AIC(fit_nb, fit_nb_q, fit_nb_yr, fit_nb_yr_q) # fit_nb smallest
BIC(fit_nb, fit_nb_q, fit_nb_yr, fit_nb_yr_q) # fit_nb smallest

### walkforward mae
mae <- function(y, yhat) mean(abs(y - yhat), na.rm = TRUE)

# formulas for the two models
form_nb    <- n_extreme ~ 1 + extreme_type + (1 | NERC_ID)
form_nb_yr <- n_extreme ~ c_year + extreme_type + (1 | NERC_ID)

# put all data in one place for expanding-window training
all_df <- bind_rows(
  train_df %>% mutate(split = "train"),
  val_df   %>% mutate(split = "val")
)

val_years <- sort(unique(val_df$start_yr))

mae_table <- map_dfr(val_years, function(y) {
  
  train_y <- all_df %>% filter(start_yr <= y - 1)
  val_y   <- val_df %>% filter(start_yr == y)
  
  fit_nb_y    <- glmmTMB(form_nb,    data = train_y, family = nbinom2(link = "log"))
  fit_nb_yr_y <- glmmTMB(form_nb_yr, data = train_y, family = nbinom2(link = "log"))
  
  pred_nb    <- predict(fit_nb_y,    newdata = val_y, type = "response", allow.new.levels = TRUE)
  pred_nb_yr <- predict(fit_nb_yr_y, newdata = val_y, type = "response", allow.new.levels = TRUE)
  
  tibble(
    val_year = y,
    model = c("fit_nb", "fit_nb_yr"),
    mae = c(mae(val_y$n_extreme, pred_nb),
            mae(val_y$n_extreme, pred_nb_yr)),
    n = nrow(val_y)
  )
}) %>% arrange(val_year, model)

mae_table

# overall MAE across validation years (weighted by # rows each year)
mae_overall <- mae_table %>%
  group_by(model) %>%
  summarise(
    val_year = paste0(min(val_year), "-", max(val_year)),
    mae = weighted.mean(mae, w = n, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(mae)


# output file not written by AI
write.csv(mae_table, 
          "../Output/cold_mae_nb_val2.csv", 
          row.names = FALSE)

write.csv(mae_overall, 
          "../Output/cold_mae_nb_overall_val2.csv", 
          row.names = FALSE)
