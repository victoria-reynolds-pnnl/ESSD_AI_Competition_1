### extreme data dictionary creation

# this code chunk is not AI (lines 1-19)
## if you do not have tidyverse, or rstudioapi
## you will have to install them
# uncomment line 5 and run
# install.packages("tidyverse", "rstudioapi")

#libraries  - need these packages for code to run
library(tidyverse) # data wrangling 
library(rstudioapi) # helps set path so works on any machine

## set working directory
setwd(dirname(getActiveDocumentContext()$path))

data_path <- "../Data/extreme_data_raw/"
output_path <- "../Data/extreme_data_clean/"

source("data_clean.R") # runs R script that cleans data

### # generated in AI 
#(lines 21-190 except comments)
# model garden gpt-5.2
# comments added after adding AI code to script
dd <- data.frame(
  file_name = c(
    list.files(data_path, full.names = FALSE),
    list.files(output_path, full.names = FALSE)
  ),
  stringsAsFactors = FALSE
)



## assign file type & folder
dd$file_type <- tools::file_ext(dd$file_name)

dd$folder <- ifelse(
  dd$file_name %in% list.files(data_path, full.names = FALSE),
  "extreme_data_raw",
  "extreme_data_clean"
)

## start populating description
dd$description <- NA

dd$description <- ifelse(grepl("clean", dd$file_name, ignore.case = TRUE),
                         "clean data with added features",
                         dd$description)

dd$description <- ifelse(grepl("count", dd$file_name, ignore.case = TRUE),
                         "clean data summary, new feature counts of extreme events by region, year, and type",
                         dd$description)
dd$description <- ifelse(grepl("\\.tif$", dd$file_name, ignore.case = TRUE),
                         "geopackage - map of NERC regions, for checking map outputs from analyses",
                         dd$description)
dd$description <- ifelse(grepl("definitions", dd$file_name, ignore.case = TRUE),
                         "area and event type descriptions from original data source",
                         dd$description)

dd$description <- ifelse(grepl("\\.gpkg$", dd$file_name, ignore.case = TRUE),
                         "geopackage of NERC subregions for output mapping",
                         dd$description)
dd$description <- ifelse(grepl("heat_wave", dd$file_name, ignore.case = TRUE),
                         "heat wave extreme event raw data for definition 9, area as average population",
                         dd$description)
dd$description <- ifelse(grepl("cold_snap", dd$file_name, ignore.case = TRUE),
                         "cold snap extreme event raw data for definition 9, area as average population",
                         dd$description)

dd$description <- ifelse(grepl("heat_wave_impact_timeseries", dd$file_name, ignore.case = TRUE),
                         "impact timeseries of heat wave extreme event by region",
                         dd$description)
dd$description <- ifelse(grepl("cold_snap_impact_timeseries", dd$file_name, ignore.case = TRUE),
                         "impact timeseries of cold snap extreme event by region",
                         dd$description)

dd$description <- dplyr::case_when(
  grepl("dictionary", dd$file_name, ignore.case = TRUE) ~ "Data dictionary for project.",
  grepl("README", dd$file_name, ignore.case = TRUE)     ~ "Describes all the files in the project.",
  TRUE                                                 ~ dd$description
)

# add description of R scripts to dd
dd <- dplyr::bind_rows(
  dd,
  data.frame(
    file_name    = c("data_clean.R", "extreme_DD.R"),
    folder       = "extreme_scripts_R",
    file_type    = ".R",
    description  = c(
      "An R script that takes the raw data csv files, cleans them, adds new features, and writes to extreme_data_clean_folder.",
      "An R script that makes the data dictionary from the raw data files, clean data files, and R scripts."
    ),
    stringsAsFactors = FALSE
  )
)
#### nerc subregion description
nerc <- data.frame(
  file_name  = "jurisdiction_nerc_subregion_v1.gpkg",
  field_name = c("id", "name", "address", "city", "state", "zip", "country",
                 "source", "sourcedate", "val_method", "val_date", "website", "subname"),
  gem_alias  = c("Id", "Name", "Address", "City", "State", "Zip", "Country",
                 "Source", "Source Date", "Value Method", "Value Date", "Website", "Subregion Name"),
  stringsAsFactors = FALSE
)
nerc$data_type <- "text"
nerc$info <- "fields directly from original geopackage dataset"

## get raw data description: cold extremes
cold_names <- data.frame(
  file_name = "cold_snap_library_NERC_average_pop_def9.csv",
  name  = names(df_cold),
  class = vapply(df_cold, function(x) paste(class(x), collapse = "/"), character(1)),
  stringsAsFactors = FALSE
)

#change temp name back to original
cold_names$name[cold_names$name == "extr_temp"] <- "lowest_temperature"

# drop source_file after creating cold_names, and reset row names
cold_names <- dplyr::filter(cold_names, !name %in% c("source_file", "type"))
row.names(cold_names) <- seq_len(nrow(cold_names))

# Add/update a description column in cold_names based on the variable name
cold_names$description <- dplyr::case_when(
  cold_names$name == "start_date" ~ "Start date of the event.",
  cold_names$name == "end_date" ~ "End date of the event.",
  cold_names$name == "centroid_date" ~ "The centroid date, calculated as the midpoint between the start and end dates.",
  cold_names$name =="lowest_temperature" ~ "The lowest daily minimum temperature (for cold snaps), in Kelvin.",
  cold_names$name == "duration" ~ "Duration of the event in days.",
  cold_names$name == "NERC_ID" ~ "Identifier for the NERC subregion.",
  cold_names$name == "spatial_coverage" ~
    "The spatial coverage of the event within the NERC subregion, expressed as the percentage of counties experiencing the event relative to the total number of counties in the subregion.",
  TRUE ~ cold_names$description %||% NA_character_
)

# make hot_names using cold_names as template
hot_names <- cold_names
hot_names$name[hot_names$name == "lowest_temperature"] <- "highest_temperature"
hot_names$description[hot_names$name == "highest_temperature"] <-
  "The highest daily maximum temperature (for heat waves) in Kelvin."
hot_names$file_name <- "heat_wave_library_NERC_average_pop_def9.csv"

# make df_names
df_names <- data.frame(
  file_name = "data_clean.csv",
  name  = names(df),
  class = vapply(df, function(x) paste(class(x), collapse = "/"), character(1)),
  stringsAsFactors = FALSE
)

df_names <- dplyr::left_join(
  df_names,
  dplyr::select(hot_names, name, description),
  by = "name"
)

df_names$description[df_names$name == "perm_num"] <-
  "Unique integer to identify rows in dataset."

df_names$description[df_names$name == "start_yr"] <-
  "The year the extreme event started."

df_names$description[df_names$name == "type"] <-
  "Extreme event type (heat wave or cold snap)."
df_names$description[df_names$name == "extr_temp"] <-
  "Temperature in Kelvin of the extreme event (highest for heat waves, lowest for cold snaps)."

df_names$description[df_names$name == "impact"] <-
  "Product of spatial_coverage, duration, and absolute departure of extr_temp from 290K (not yet normalized)."
df_names$description[df_names$name == "extreme_type"] <-
  "Event type (heat, cold) pasted to the regional duration quartile for the event type, pasted to the regional extr_temp quartile for the event type."

# make df_long_names
df_long_names <- data.frame(
  file_name = "region_yr_extreme_count.csv",
  name  = names(df_long),
  class = vapply(df_long, function(x) paste(class(x), collapse = "/"), character(1)),
  stringsAsFactors = FALSE
)

df_long_names <- dplyr::left_join(
  df_long_names,
  dplyr::select(df_names, name, description),
  by = "name"
)

df_long_names$description[df_long_names$name == "n_extreme"] <-
  "Count of extreme events by type (extreme_type) within NERC region and year."

# manually added for "heat_wave_impact_timeseries.csv" and "cold_snap_impact_timeseries.csv",
c_nerc <- c('NERC1', 'NERC2', 'NERC3', 'NERC4', 'NERC5', 'NERC6', 'NERC7', 'NERC8', 'NERC9', 'NERC10', 'NERC11', 'NERC12', 'NERC15', 'NERC17', 'NERC18', 'NERC20')
df_cold_ts_names <- data.frame(
  file_name = "cold_snap_impact_timeseries.csv",
  name  = c(c('date'), c_nerc),
  class = c(c('Date'), rep('numeric', times = length(c_nerc))),
  description = c('Timeseries date', paste(c_nerc, 'region impact (spatial_coverage * duration * (290K - extr_temp)) timeseries; -1 represents no event'))
)
df_heat_ts_names <- data.frame(
  file_name = "heat_wave_impact_timeseries.csv",
  name  = c(c('date'), c_nerc),
  class = c(c('Date'), rep('numeric', times = length(c_nerc))),
  description = c('Timeseries date', paste(c_nerc, 'region impact (spatial_coverage * duration * (extr_temp - 290K)) timeseries; -1 represents no event'))
)

df_long_names$description[df_long_names$name == "n_extreme"] <-
  "Count of extreme events by type (extreme_type) within NERC region and year."

# make dd_details
dd_details <- dplyr::bind_rows(cold_names, 
                               hot_names, 
                               df_names, 
                               df_long_names,
                               df_cold_ts_names,
                               df_heat_ts_names)
names(dd_details)[names(dd_details) == "name"] <- "variable_name"

## output not written by AI
write.csv(dd, 
          file = file.path(output_path, 
                           "README.csv"),
          row.names = FALSE, na = "")
write.csv(dd_details, 
          file = file.path(output_path,
                           "data_dictionary.csv"),
          row.names = FALSE, na = "")
