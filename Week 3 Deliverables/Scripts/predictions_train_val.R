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
library(viridis) # for plotting colors

## set working directory & get data
setwd(dirname(getActiveDocumentContext()$path))

train_df <- read.csv("../Output/cold_train_df.csv")
val_df <- read.csv("../Output/cold_val_df.csv")

train_df$data <- "train"
val_df$data <- "validate"

# from here down is AI
# written with chatgpt gpt 5.2
nerc_levels <- c(
  "NERC1","NERC2","NERC3","NERC4","NERC5","NERC6","NERC7","NERC8",
  "NERC9","NERC10","NERC11","NERC12","NERC15","NERC17","NERC18","NERC20"
)
train_df$NERC_ID <- factor(train_df$NERC_ID, levels = nerc_levels)
val_df$NERC_ID   <- factor(val_df$NERC_ID,   levels = nerc_levels)

train_df$c_year <- train_df$start_yr - mean(train_df$start_yr, na.rm=TRUE)
val_df$c_year <- val_df$start_yr - mean(val_df$start_yr, na.rm=TRUE)

df <- bind_rows(train_df, val_df)

# define form & model - includes year
form <- n_extreme ~ c_year + extreme_type + (1 | NERC_ID)

fit_nb <- glmmTMB(
  form, data = train_df,
  family = nbinom2(link = "log") # negative binomial
)

# Predict on val data 
df_predict <- val_df %>%
  distinct(NERC_ID, start_yr,c_year,extreme_type) %>%
  mutate(
    pred_n_extreme = predict(
      fit_nb,
      newdata = .,
      type = "response",
      allow.new.levels = TRUE
    )
  )

df_merged <- val_df %>%
  left_join(df_predict, by = c("NERC_ID", "start_yr", "extreme_type"))

df_merged <- df_merged %>%
  mutate(code = substr(extreme_type, 
                       nchar(extreme_type) - 1, nchar(extreme_type)))

###
df_tot <- df_merged %>%
  group_by(NERC_ID, start_yr) %>%
  summarise(n_extreme = sum(n_extreme, na.rm = TRUE), .groups = "drop")

p <- ggplot(df_tot, aes(start_yr, n_extreme, group = NERC_ID, color = NERC_ID)) +
  geom_line(alpha = 0.7) +
  geom_point(size = 1.4, alpha = 0.9) +
  scale_x_continuous(breaks = sort(unique(df_tot$start_yr))) +
  scale_y_continuous(trans = "log1p") +
  scale_color_viridis_d(option = "D", end = 0.9) +
  labs(x = "Year", y = "Total observed extremes (log1p scale)", color = "NERC_ID") +
  theme_bw()

df_tot_pred <- df_merged %>%
  group_by(NERC_ID, start_yr) %>%
  summarise(pred_n_extreme = sum(pred_n_extreme, na.rm = TRUE), .groups = "drop")

p1 <- ggplot(df_tot_pred, aes(start_yr, pred_n_extreme, group = NERC_ID, color = NERC_ID)) +
  geom_line(alpha = 0.7) +
  geom_point(size = 1.4, alpha = 0.9) +
  scale_x_continuous(breaks = sort(unique(df_tot$start_yr))) +
  scale_y_continuous(trans = "log1p") +
  scale_color_viridis_d(option = "D", end = 0.9) +
  labs(x = "Year", y = "Total predicted extremes (log1p scale)", color = "NERC_ID") +
  theme_bw()

