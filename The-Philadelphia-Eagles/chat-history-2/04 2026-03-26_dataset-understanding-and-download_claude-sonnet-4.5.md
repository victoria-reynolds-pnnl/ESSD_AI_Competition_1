# Dataset Understanding, Biases, and Download Strategy

**Date:** 2026-03-26  
**Model:** Claude Sonnet 4.5 (claude-sonnet-4-5-20250929-v1-birthright)  
**Tokens Used** ↑ 563.3k ↓ 14.1k     Size 1.06 MB  
**Owner:** Team The Philadelphia Eagles  
**Purpose:** Deep dive into ComExDBM dataset structure, CONUS vs WUS comparison, biases/limitations analysis, and practical guidance for downloading 200GB+ dataset  
**Related Week / Deliverable:** Week 1 - Data Discovery and Setup  
**Related Docs:**
- [../wiki/01.02 Week 1 Deliverables.md](../wiki/01.02%20Week%201%20Deliverables.md)
- [../wiki/02.01 Dataset Details.md](../wiki/02.01%20Dataset%20Details.md)
- [../wiki/00.01 Dataset Download Guide.md](../wiki/00.01%20Dataset%20Download%20Guide.md)
- [../data/ComExDBM_CONUS/README.md](../data/ComExDBM_CONUS/README.md)


_This is a curated transcript with tool execution details removed for readability._

---

## Summary

This chat session focused on three main objectives:

1. **Understanding the ComExDBM dataset structure** - Explaining what CONUS and WUS mean and their key differences
2. **Analyzing biases and limitations** - Critical evaluation of dataset constraints for ML modeling
3. **Practical download guidance** - Best practices for downloading and managing 200GB+ of data

The conversation resulted in comprehensive documentation of dataset characteristics and a detailed download guide to help the team acquire the full CONUS dataset efficiently.

---

## Cleaned Transcript

### User

I am working on the Week 1 deliverables for the ESSD AI Challenge. My teammate has already done this using GPT Codex, so I am using you and a new model (claude sonnet 4-5). The deliverables for Week 1 are defined in the markdown file, `wiki/01.02 Week 1 Deliverables.md`. To start, using the data and README files found in `data/ComExDBM_CONUS/`, can you help me understand the data? What is the difference between CONUS and WUS?

### Assistant

_[Reviewed Week 1 deliverables, dataset README files, and existing team documentation]_

Based on the documentation I've reviewed, I can now provide you with a comprehensive understanding of the ComExDBM dataset and the key differences between CONUS and WUS.

## Understanding the ComExDBM Dataset

**ComExDBM** stands for **Compound Extremes Data Benchmarking**. It's an AI-ready benchmark dataset that integrates multiple extreme weather phenomena across the United States from 2001-2020, specifically designed for machine learning research.

The dataset combines:
- **Heatwave events** with multiple heat indices (HI02, HI04, HI05, HI06, HI09, HI10)
- **Fire-related features** (hotspot counts, fire radiative power, burned area)
- **Drought indices** (PDSI, SPI, SPEI at multiple timescales)
- **Meteorological variables** (temperature, precipitation, humidity, wind, soil moisture, vegetation indices, etc.)

All data is harmonized to **daily temporal resolution** and stored in CF-compliant NetCDF files.

## CONUS vs WUS: Key Differences

| **Aspect** | **CONUS** | **WUS** |
|------------|-----------|---------|
| **Geographic Coverage** | Contiguous United States (entire lower 48 states) | Western United States (regional subset) |
| **Spatial Resolution** | **0.5° × 0.5°** (~55 km × 55 km) | **1 km × 1 km** (much finer detail) |
| **Grid Cells** | Coarser, fewer cells | Fine-scale, many more cells |
| **Data Volume** | More manageable | Significantly larger |
| **Meteorological Variables** | **Richer documented set** including NARR and MERRA-2 variables | More compact set |
| **Computational Complexity** | Lower | Higher |
| **Use Case** | Broad regional analysis, easier ML benchmarking | High-resolution regional analysis, more detailed spatial patterns |
| **Challenge Suitability** | **Better for 5-week timeline** | Stretch goal, higher risk |

## Why Your Team Chose CONUS

According to your team's documentation, you selected **CONUS fire occurrence classification** because:

1. **Clear binary target** - Easy to explain and evaluate (fire/no-fire)
2. **Manageable scope** - Fits the 5-week challenge better than fine-scale WUS
3. **Dual approach support** - Works for both traditional ML and LLM-based methods
4. **Coarser grid advantages**:
   - Easier preprocessing
   - Reduced computational cost
   - Lower class imbalance issues
   - Simpler feature engineering
   - Better for compact LLM prompts in Week 4

## Data Structure in Your Repository

The CONUS data at `data/ComExDBM_CONUS/` contains:

- **`01_Input/`** - Source data descriptions and shapefiles
- **`04_OutputData/`** - Processed NetCDF files organized by type:
  - `01_HeatWave/` - Temperature and heatwave labels (2001-2020)
  - `02_FireData/` - Fire hotspots, FRP, burned area (2001-2020)
  - `03_DroughtData/` - PDSI, SPI, SPEI indices (2001-2020)
  - `04_Meteorological_Variables/` - Met vars from NARR/MERRA-2 (2001-2020)

Each year has separate NetCDF files following the naming convention: `ComExDBM_YYYY_DataType_V02.nc`

## Your Week 1 Problem Statement

From your team's documentation, you're focused on:

> "Predict whether a CONUS grid cell will experience fire activity on a given day or week using co-located drought, meteorological, vegetation, and lagged fire features."

This leverages the CONUS dataset's strengths while keeping the scope manageable for the challenge timeline.

---

### User

With this information, what are some biases and limitations of using this dataset? If we were to only use the WUS data, since it has a smaller scope, would accomplishing our problem statement be more difficult or easier to create an accurate prediction model?

### Assistant

_[Analyzed dataset documentation and team research notes on biases and limitations]_

## Biases and Limitations of the ComExDBM Dataset

Based on the documentation, the ComExDBM dataset has several important biases and limitations:

### 1. **Multi-Source Integration Biases**
The benchmark assembles data from multiple source systems with different native resolutions and measurement assumptions (MODIS, CPC, gridMET, NARR, MERRA-2, Daymet). Each source has its own:
- Measurement errors and uncertainties
- Spatial/temporal coverage gaps
- Processing assumptions
- Calibration differences

### 2. **Operational Definition Limitations**
Event labels are operational definitions, not absolute ground truth:
- Heatwave labels depend on specific index definitions (HI02-HI10) and percentile thresholds
- Fire detection depends on MODIS confidence levels and may miss small fires or be affected by cloud cover
- Drought indices are model-derived, not direct observations

### 3. **Severe Class Imbalance**
Fire occurrence and compound extremes are rare events, leading to:
- Highly imbalanced datasets (likely <1% fire occurrence rate)
- Difficulty training models without specialized techniques
- Risk of models that simply predict "no fire" and achieve high accuracy
- Need for metrics like F1, PR-AUC instead of accuracy

### 4. **Spatial and Temporal Correlation**
- Adjacent grid cells and consecutive days are highly correlated
- Random row splits would create invalid evaluation (data leakage)
- Requires careful time-based or spatial-block splits

### 5. **Spatial Coverage Differences**
Signal quality and coverage differ between CONUS and WUS, and across fire, drought, and meteorological source systems.

### 6. **Temporal Consistency Issues**
Data quality and satellite coverage may vary across the 2001-2020 period, with earlier years potentially having:
- Lower satellite resolution
- Fewer validation sources
- Different processing algorithms

## WUS vs CONUS for Fire Prediction: Difficulty Analysis

**Using WUS would make the problem SIGNIFICANTLY MORE DIFFICULT**, not easier, despite the smaller geographic scope:

### Why WUS is Harder:

**1. Extreme Class Imbalance (Much Worse)**
- 1 km × 1 km resolution = **3,025× more grid cells** than 0.5° × 0.5° for the same area
- Fire events don't scale proportionally → even rarer positive cases per cell
- Class imbalance ratio could be 1000:1 or worse
- Requires aggressive sampling or specialized loss functions

**2. Computational Complexity**
- **~3,000× more data points** to process
- Monthly file structure (vs yearly for CONUS) = more file I/O
- Memory and processing time become major bottlenecks
- Harder to iterate quickly during model development

**3. Feature Engineering Complexity**
- Finer spatial resolution requires more sophisticated spatial context features
- Neighborhood effects become more important but harder to compute
- Risk of spatial leakage increases with fine-grained spatial features

**4. Reduced Meteorological Feature Set**
WUS has a more compact documented meteorological variable set compared to CONUS's richer NARR/MERRA-2 variables, giving fewer feature engineering options.

**5. Overfitting Risk**
- More parameters needed to capture fine-scale patterns
- Higher risk of learning noise rather than signal
- Requires more sophisticated regularization

**6. LLM Integration Challenges**
- Cannot practically prompt an LLM with individual 1km cell data
- Requires aggressive aggregation, losing the resolution advantage
- Makes Week 4 LLM comparison much harder

### Why CONUS is Better for Accurate Prediction:

**1. Better Signal-to-Noise Ratio**
- Coarser aggregation smooths out measurement noise
- Fire events are more consistently captured at 55km scale
- More stable patterns for models to learn

**2. More Balanced Classes**
- Higher fire occurrence rate per grid cell
- Easier to train without extreme sampling
- More reliable evaluation metrics

**3. Richer Feature Set**
- Broader documented meteorological variables
- More feature engineering opportunities
- Better predictive power

**4. Manageable Scope**
- Faster iteration cycles
- Easier to experiment with different approaches
- Better fit for 5-week timeline

**5. Easier Validation**
- Simpler to implement proper spatial/temporal splits
- More interpretable results at regional scale
- Better for comparing ML vs LLM approaches

## Conclusion

For your fire occurrence prediction problem, **CONUS is the superior choice** for building an accurate model. While WUS offers finer spatial detail, that detail comes at the cost of:
- Extreme computational burden
- Much worse class imbalance
- Reduced feature richness
- Higher overfitting risk
- Incompatibility with LLM comparison (Week 4)

The coarser CONUS grid provides better signal aggregation, more manageable class balance, richer features, and a more realistic scope for the 5-week challenge timeline. Your team made the right decision selecting CONUS for the competition.

---

### User

The full CONUS dataset is 200GB+ of data, stored in a tar file accessible online. What is the best way to download this data?

### Assistant

_[Created comprehensive download guide considering devcontainer setup and large file best practices]_

I've created a comprehensive dataset download guide at `wiki/00.01 Dataset Download Guide.md` with best practices for downloading the 200GB+ CONUS dataset.

## Key Recommendations for Downloading 200GB+ Data:

### Best Method: `wget` with Resume Capability

```bash
cd /path/to/parent-folder/dataset

wget --continue --progress=bar:force \
  --timeout=60 --tries=0 \
  "https://data.pnnl.gov/dataset/download/[URL]" \
  -O ComExDBM_CONUS.tar.gz
```

**Why `wget`:**
- Automatic resume on interruption (`--continue`)
- Infinite retries (`--tries=0`)
- Robust for large files
- Pre-installed on most systems

### Alternative: `aria2c` (Fastest)

For maximum speed, use `aria2c` with multiple parallel connections:

```bash
aria2c --continue=true \
  --max-connection-per-server=8 \
  --split=8 \
  "https://data.pnnl.gov/dataset/download/[URL]"
```

This can be 2-5× faster than single-connection downloads.

### Critical Points:

1. **Download Location**: Download to `../dataset/` (sibling folder), NOT inside the repo
2. **Disk Space**: Need ~500GB free (200GB compressed + 200GB extracted + buffer)
3. **Network Stability**: Use wired connection; estimated 4-10 hours depending on speed
4. **Use `screen` or `tmux`**: Prevents interruption if terminal closes
5. **Resume Capability**: All recommended methods support automatic resume

### Extraction:

```bash
# Standard extraction
tar -xzvf ComExDBM_CONUS.tar.gz

# Or with progress indicator
pv ComExDBM_CONUS.tar.gz | tar -xz
```

### Verification:

After extraction, verify you have:
- 80 NetCDF files (20 years × 4 data types)
- Expected directory structure with `01_HeatWave/`, `02_FireData/`, `03_DroughtData/`, `04_Meteorological_Variables/`

The guide includes troubleshooting tips, disk space calculations, network time estimates, and post-download verification steps. Get the actual download URL from the DOI link: <https://doi.org/10.25584/2004956>

---

### User

Can you include these chat logs into a new markdown file in the 'saved-chats' folder?

### Assistant

_[Creating this curated chat log file]_

---

## Key Outcomes

### 1. Dataset Understanding Documentation

**CONUS (Contiguous United States):**
- 0.5° × 0.5° resolution (~55 km grid cells)
- Nationwide coverage of lower 48 states
- Richer meteorological variable set
- Yearly NetCDF file structure
- Better for ML benchmarking and LLM comparison

**WUS (Western United States):**
- 1 km × 1 km resolution (55× finer)
- Regional western US coverage
- More compact variable set
- Monthly NetCDF file structure
- Higher computational complexity

### 2. Biases and Limitations Analysis

**Critical Issues Identified:**
- Multi-source integration inconsistencies
- Operational definitions vs. ground truth
- Severe class imbalance for fire events
- Spatial/temporal autocorrelation requiring careful validation
- Temporal consistency variations across 2001-2020
- Coarse resolution masking fine-scale events

**Validation Requirements:**
- Time-based or spatial-block splits (not random)
- Metrics appropriate for imbalanced data (F1, PR-AUC)
- Explicit leakage prevention in feature engineering
- Careful handling of lagged variables

### 3. CONUS vs WUS Modeling Difficulty

**Conclusion:** CONUS is significantly easier and more appropriate for the 5-week challenge:

**CONUS Advantages:**
- Better signal-to-noise ratio
- More manageable class balance
- Richer feature set
- Faster iteration cycles
- Compatible with LLM comparison

**WUS Challenges:**
- 3,000× more data points
- Extreme class imbalance (worse than CONUS)
- Higher computational cost
- More complex feature engineering
- Difficult LLM integration

### 4. Download Strategy Documentation

**Created:** [`wiki/00.01 Dataset Download Guide.md`](../wiki/00.01%20Dataset%20Download%20Guide.md)

**Key Recommendations:**
- Use `wget --continue` for robustness
- Use `aria2c` for maximum speed (8 parallel connections)
- Download to `../dataset/` sibling folder
- Require ~500GB free disk space
- Use `screen`/`tmux` for long downloads
- Verify 80 NetCDF files after extraction

**Download Time Estimates:**
- 100 Mbps: 4-5 hours
- 50 Mbps: 8-10 hours
- 10 Mbps: 40-50 hours

---

## Technical Insights

### Dataset Structure

```text
ComExDBM_CONUS/
├── 01_Input/              # Source data descriptions
├── 04_OutputData/
│   ├── 01_HeatWave/       # 20 yearly files (2001-2020)
│   ├── 02_FireData/       # 20 yearly files
│   ├── 03_DroughtData/    # 20 yearly files
│   └── 04_Meteorological_Variables/  # 20 yearly files
└── README.md
```

### File Naming Convention

```
ComExDBM_YYYY_DataType_V02.nc
```

Where:
- `YYYY` = year (2001-2020)
- `DataType` = HeatWave, Fires, Drought, MetVars
- `V02` = version 2

### Expected Data Volume

- **Compressed tar**: ~200GB
- **Extracted data**: ~200GB+
- **Total during extraction**: ~400GB
- **Recommended free space**: 500GB+

---

## Alignment with Week 1 Deliverables

This conversation directly supports the Week 1 requirement for:

> "A single paragraph detailing data discovery that describes the data and addresses potential biases and data limitations."

The analysis provided:
- ✅ Comprehensive dataset description
- ✅ Explicit bias identification
- ✅ Limitation analysis
- ✅ Validation strategy recommendations
- ✅ Practical download guidance

---

## Files Created/Modified

- **Created:** `wiki/00.01 Dataset Download Guide.md` - Comprehensive guide for downloading and managing 200GB+ dataset
- **Created:** `saved-chats/04 2026-03-26_dataset-understanding-and-download_claude-sonnet-4.5.md` - This chat log

---

## Next Steps for the Team

1. **Download the full CONUS dataset** using the guide
2. **Verify data integrity** (80 NetCDF files, correct structure)
3. **Begin exploratory data analysis** to confirm class imbalance and data characteristics
4. **Design validation strategy** with time-based or spatial-block splits
5. **Start feature engineering** for Week 2 deliverables

---

## Comparison with Previous Sessions

This Claude Sonnet 4.5 session complemented earlier work by:

- **Codex Session 01:** Dataset discovery and problem selection
- **Codex Session 02:** Fire occurrence decision and app direction
- **Claude Session 03:** Week 1 deliverables and 3D app vision
- **Claude Session 04 (this):** Deep dive into biases, limitations, and practical download strategy

Each session built on the previous work, progressively refining the team's understanding and documentation.

---

## Metadata

**Session Duration:** ~20 minutes  
**Primary Tools Used:** read_file, search_files, write_to_file, attempt_completion  
**Key Documents Referenced:**
- Week 1 deliverables template
- ComExDBM_CONUS README files
- Dataset details documentation
- Devcontainer setup guide

**Key Concepts Covered:**
- Dataset structure and organization
- CONUS vs WUS comparison
- Biases and limitations analysis
- Class imbalance implications
- Validation strategy requirements
- Large file download best practices
- Disk space and network considerations
