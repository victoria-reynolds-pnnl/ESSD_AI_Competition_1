# this code chunk is not AI (lines 1-28)
## if you do not have these packages
## you will have to install them
# uncomment line 5-6 and run
#install.packages("tidyverse", "rstudioapi",
#                 "glmmTMB", "sjPlot")

# libraries
#R packages needed for code to run

library(tidyverse) # data wrangling 
library(rstudioapi) # helps set path so works on any machine

# fit mixed models
library(glmmTMB) # fit neg bin & zero-inflated 
library(sjPlot) # make nice model tables

## set working directory & get data
setwd(dirname(getActiveDocumentContext()$path))

#get MAE val
mae_val <- read.csv("../Output/cold_mae_val.csv")

# get datasets for coding
train_df <- read.csv("../Output/cold_train_df.csv")
val_df <- read.csv("../Output/cold_val_df.csv")

# written by AI
# gpt-5.2 
# plot mae val as panels
ggplot(mae_val, aes(x = val_year, y = mae)) +
  geom_line(linewidth = 1) +
  geom_point(size = 2) +
  facet_wrap(~ model, ncol = 1) +
  scale_x_continuous(breaks = sort(unique(mae_val$val_year))) +
  labs(x = "Validation year", y = "MAE") +
  theme_minimal()



# mae_val: columns val_year, model, mae

# 1) Wide table (one row per year, one column per model)
mae_wide <- mae_val %>%
  pivot_wider(names_from = model, values_from = mae) %>%
  arrange(val_year)

# 2) Year-by-year MAE differences vs baseline (fit_nb)
mae_diff <- mae_wide %>%
  mutate(
    diff_fit_nb_yr    = fit_nb_yr    - fit_nb,
    diff_fit_nb_yr_rs = fit_nb_yr_rs - fit_nb,
    diff_fit_nb_yr_rs2 = fit_nb_yr_rs2 - fit_nb
  )

mae_diff %>% select(val_year, 
                    diff_fit_nb_yr, 
                    diff_fit_nb_yr_rs,
                    diff_fit_nb_yr_rs2)

# 3) Summary: average difference and how often each model beats fit_nb
diff_summary <- mae_diff %>%
  summarise(
    mean_diff_fit_nb_yr      = mean(diff_fit_nb_yr, na.rm = TRUE),
    mean_diff_fit_nb_yr_rs   = mean(diff_fit_nb_yr_rs, na.rm = TRUE),
    mean_diff_fit_nb_yr_rs2  = mean(diff_fit_nb_yr_rs2, na.rm = TRUE),
    
    n_years_yr_better        = sum(diff_fit_nb_yr < 0, na.rm = TRUE),
    n_years_yr_rs_better     = sum(diff_fit_nb_yr_rs < 0, na.rm = TRUE),
    n_years_yr_rs2_better    = sum(diff_fit_nb_yr_rs2 < 0, na.rm = TRUE),
    
    n_years_total            = n()
  )
diff_summary

# 4) Optional: plot the differences (below 0 means better than fit_nb)
mae_diff_long <- mae_diff %>%
  select(val_year, 
         diff_fit_nb_yr, diff_fit_nb_yr_rs, diff_fit_nb_yr_rs2) %>%
  pivot_longer(-val_year, names_to = "comparison", values_to = "mae_diff")


p <- ggplot(mae_diff_long, aes(val_year, mae_diff, color = comparison)) +
  geom_hline(yintercept = 0, linetype = 2) +
  geom_line(linewidth = 1) +
  geom_point(size = 2) +
  scale_x_continuous(breaks = sort(unique(mae_diff_long$val_year))) +
  scale_y_continuous(labels = scales::label_number(accuracy = 0.0001)) +
  scale_color_manual(values = c(
    diff_fit_nb_yr    = "magenta",
    diff_fit_nb_yr_rs = "black",
    diff_fit_nb_yr_rs2 = "blue"
  )) +
  labs(
    x = "Validation year",
    y = "MAE difference (events per year)\n(model − fit_nb)",
    color = NULL
  ) +
  theme_minimal()+
  ggtitle("Comparison of Year Models to Negative Binomial Model")

p

ggsave("../Visualizations/mae_plot.png",
       height = 6, width = 8)

# ## create tr_val & center time
tr_val <- rbind(train_df, 
                val_df)
tr_val$c_year <- tr_val$start_yr - mean(tr_val$start_yr, 
                                        na.rm=TRUE)
names(tr_val)[3] <- "cold"

tr_val$cold <- gsub("cold_dur_temp",
                    "dt",
                    tr_val$cold)

### seems to be overfitting
## model not converging
form_yr_rs <- n_extreme ~ c_year + cold + (c_year | NERC_ID)

fit_nb_yr_rs <- glmmTMB(form_yr_rs, 
                        data=tr_val, 
                        family=nbinom2())

###
fit_nb_yr_rs2 <- glmmTMB(
  n_extreme ~ c_year + cold + (1 + c_year || NERC_ID),
  data = tr_val,
  family = nbinom2()
)

## simpler best model
form <- n_extreme ~ 1 + cold + (1 | NERC_ID)
fit_nb <- glmmTMB(form, 
                  data = tr_val, 
                  family = nbinom2(link="log"))

## AIC, BIC - not written by AI
a <- as.data.frame(AIC(fit_nb, 
                       fit_nb_yr_rs, fit_nb_yr_rs2))
b <- as.data.frame(BIC(fit_nb, 
                       fit_nb_yr_rs, fit_nb_yr_rs2))

fit.check <- data.frame(
  model = rownames(a),
  df    = a$df,
  AIC   = a$AIC,
  BIC   = b$BIC,
  row.names = NULL
)

pnb <- plot_model(fit_nb, 
           type = "est", 
           transform = "exp", 
           show.values = TRUE)  # IRR scale
pnb <- pnb + geom_hline(yintercept = 1, 
                        linetype = "dashed", 
                        color = "black")+
  theme_minimal()+
  ggtitle("Cold Snaps")+
  scale_y_log10(limits = c(0.1, 25))

pnb

ggsave("../Visualizations/fixed_nb_val.png",
       height = 7.5, width = 5)

###
# add fitted mean (incidence) on response scale
df_plot <- train_df %>%
  mutate(pred_incidence = predict(fit_nb, type = "response"))

# aggregate by region (and optionally year/type if you want finer resolution)
by_region <- df_plot %>%
  group_by(NERC_ID) %>%
  summarise(
    actual = mean(n_extreme, na.rm = TRUE),
    predicted = mean(pred_incidence, na.rm = TRUE),
    n = dplyr::n(),
    .groups = "drop"
  )

# predicted vs actual scatter
ggplot(by_region, aes(x = actual, y = predicted, label = NERC_ID)) +
  geom_abline(slope = 1, intercept = 0, linetype = 2, color = "grey50") +
  geom_point(size = 2) +
  ggrepel::geom_text_repel(max.overlaps = Inf) +
  labs(x = "Actual mean incidence", y = "Predicted mean incidence") +
  theme_bw()

# Random intercepts for NERC_ID (BLUPs)
pr <- plot_model(fit_nb, 
           type = "re", 
           group.terms = "NERC_ID",
           sort.est = TRUE, 
           show.values = TRUE)+ geom_hline(yintercept = 1, 
                                                            linetype = "dashed", 
                                                            color = "black")+
  theme_minimal()+
  ggtitle("Region Random Effects")
pr

z <- pr$data

z <- z %>%
  mutate(term = fct_reorder(term, estimate, .desc = FALSE)) 

ggplot(z, aes(estimate, term, colour = group)) +
  geom_vline(xintercept = 1, linetype = 2, color = "grey50") +
  geom_pointrange(aes(xmin = conf.low, xmax = conf.high)) +
  theme_minimal()+
  ylab("NERC Subregion")+
  xlab("Region Specific Rate Multiplier")
  scale_colour_manual(values = c(neg = "red", pos = "blue"), guide = "none") 

# not written by AI
write.csv(fit.check,
          "../Output/val_fit_check.csv")
