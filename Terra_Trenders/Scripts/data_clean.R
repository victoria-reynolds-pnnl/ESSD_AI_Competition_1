# this code chunk is not AI (lines 1-16)
## if you do not have tidyverse or rstudioapi
## you will have to install them
# uncomment line 5 and run
# install.packages("tidyverse", "rstudioapi")

#libraries - need these packages for code to run
library(tidyverse) # data wrangling 
library(rstudioapi) # helps set path so works on any machine

## set working directory
setwd(dirname(getActiveDocumentContext()$path))

data_path <- "../Data/extreme_data_raw"
output_path <- "../Data/extreme_data_clean/"

# generated in AI 
#(lines 17-101 except comments)
# model garden gpt-5.2
# comments added after adding AI code to script
files <- list.files(data_path, pattern = "\\.csv$", full.names = TRUE)

dfs <- files %>%
  set_names(basename(.)) %>%
  map(~ read_csv(.x, show_col_types = FALSE))

df_heat <- dfs[names(dfs) %>% str_detect(regex("heat", ignore_case = TRUE))] %>%
  bind_rows(.id = "source_file")

df_cold <- dfs[names(dfs) %>% str_detect(regex("cold", ignore_case = TRUE))] %>%
  bind_rows(.id = "source_file")

df_cold <- df_cold %>% rename(extr_temp = lowest_temperature)
df_heat <- df_heat %>% rename(extr_temp = highest_temperature)

df_cold <- df_cold %>% mutate(type = "cold_snap")
df_heat <- df_heat %>% mutate(type = "heat_wave")

# combine files

df <- bind_rows(df_heat, df_cold) %>%
  select(-source_file)

# assign each row a unique number
df$perm_num <- seq_len(nrow(df))

### create two new features
# start_yr = year event began
# impact = spatial area * duration * extreme temp

df <- df %>%
  mutate(
    start_yr = year(ymd(start_date)),
    impact   = case_when(
      type == 'heat_wave' ~ spatial_coverage * duration * (extr_temp - 290), # extr_temp > 290K
      type == 'cold_snap' ~ spatial_coverage * duration * (290 - extr_temp)  # extr_temp < 290K
    )
  )

### update df with region specific quartiles
### for both event types
df <- df %>%
  mutate(start_yr = year(ymd(start_date))) %>%
  group_by(NERC_ID, type) %>%   # compute quartiles within each region + event type
  mutate(
    qrt_dur_heat = if_else(type == "heat_wave", ntile(duration, 4L), NA_integer_),
    qrt_dur_cold = if_else(type == "cold_snap", ntile(duration, 4L), NA_integer_)
  ) %>%
  ungroup()

# create region-specific extreme temp quartiles
df <- df %>%
  group_by(NERC_ID, type) %>%
  mutate(
    qrt_temp_heat = if_else(type == "heat_wave", ntile(extr_temp, 4L), NA_integer_),
    qrt_temp_cold = if_else(type == "cold_snap", ntile(extr_temp, 4L), NA_integer_)
  ) %>%
  ungroup()

# create extreme type - new feature
# by appending quartiles to type
# heat_dur_temp
# cold_dur_temp
# then drop quartile variables
df <- df %>%
  mutate(
    extreme_type = case_when(
      type == "heat_wave" ~ paste0("heat_dur_temp", qrt_dur_heat, qrt_temp_heat),
      type == "cold_snap" ~ paste0("cold_dur_temp", qrt_dur_cold, qrt_temp_cold),
      TRUE ~ NA_character_
    )
  )%>%
  select(-qrt_dur_heat:-qrt_temp_cold)

## rearrange so that perm_num and start_yr first
df <- df %>%
  relocate(perm_num, start_yr, 
           type, NERC_ID, .before = 1)

### new feature: # count of extreme types 
# by region & year
df_long <- df %>%
  group_by(NERC_ID, start_yr, extreme_type) %>%
  summarise(n_extreme = n(), .groups = "drop")

## output not written by AI
write.csv(df, 
          file = file.path(output_path, 
                 "data_clean.csv"),
          row.names = FALSE, na = "")
write.csv(df_long, 
          file = file.path(output_path,
                           "region_yr_extreme_count.csv"),
          row.names = FALSE, na = "")
