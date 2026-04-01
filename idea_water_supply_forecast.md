# Idea - water supply forecasting

## Overall concept
Using streamflow data and other public resources, develop a water supply forecast for the Columbia Eiver basin. Build a machine learning model that uses streamflow data and other sources of climate and show data adn vlidates the model using. 

## Resources

### Contest materials
- README.md
- week_2_deliverables.md
- 

### Potential training data

| Name | Format | Time range (Years) | Link |
| --- | --- | --- | --- |
| USGS Water Data for Washington (water conditions and monitoring locations) | Web portal (various data types) | Varies (site dependent) | [USGS](https://waterdata.usgs.gov/state/Washington/) |
| Climate teleconnection indicies | Web portal (various data types) | Varies (site dependent) | [NOAA](https://psl.noaa.gov/data/climateindices/list/) |
| Snow data | Web portal (various data types) | Varies (site dependent) | [NRCS](https://www.nrcs.usda.gov/programs-initiatives/sswsf-snow-survey-and-water-supply-forecasting-program/snow-and-water-products) |
| Natural flow data | TBD | TBD | TBD |


### Other forcasts to compare against

| Name | Format | Time range (Years) | Link |
| --- | --- | --- | --- |
| NRCS Forecasts | Web portal (various data types) | Varies (site dependent) | [NRCS](https://www.nrcs.usda.gov/resources/data-and-reports/water-supply-forecast-predefined-reports) |
| NWRFC Forecasts | Web portal (various data types) | Varies (site dependent) | [NWRFC](https://www.nwrfc.noaa.gov/ws/) |
| BPA Forecasts | Web portal (various data types) | Varies (site dependent) | [NWRFC](https://www.nwrfc.noaa.gov/ws/) |


## Implementation

Process data, develop features and dependent variables, train model and validate the performance.


## Considerations
- Which locations to produce forecasts for?
  - The Dalles is the most important location and would be a good plance to start.
  - Are there other key streamflow locations to produce forecasts for?
- Which snow gages to use?
- Where to get natural flow data?

\\