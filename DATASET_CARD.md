# Dataset Card: UCI Covertype

## Identity and provenance

- Name: Covertype
- Creator: Jock Blackard
- Institution/source: UCI Machine Learning Repository; source attributes originate from USFS and USGS data
- UCI identifier: 31
- URL: https://archive.ics.uci.edu/dataset/31/covertype
- DOI: https://doi.org/10.24432/C50K5N
- License: Creative Commons Attribution 4.0 International (CC BY 4.0)
- Data type: multivariate tabular classification
- Size: 581,012 observations, 54 input features, one target
- Target: forest `Cover_Type`, integers 1 through 7
- Missing values reported by UCI: none
- Synthetic status: real cartographic observations; not synthetic

## Features

The first 10 columns are quantitative terrain and distance measurements. The remaining 44 columns encode four wilderness areas and 40 soil types as binary indicators. The target is not included in preprocessing features.

## Use in this repository

The pipeline downloads the dataset through `ucimlrepo`, validates 54 numeric input columns, seven classes, row alignment, and absence of nulls, then stores a compressed local cache excluded from Git.

Quick mode uses a seeded stratified sample of 20,000 observations. Full mode uses all observations. Both modes use stratified train, validation, and test partitions.

## Leakage controls

- The test set is isolated before model training and early stopping.
- The validation set drives early stopping; it does not replace the test set.
- Standardization is fitted on training data only.
- The target column is removed before feature preparation.
- Binary indicator columns are not fitted through the continuous scaler.

## Known limitations

- Classes are imbalanced, so macro-F1 is reported with accuracy.
- The four wilderness areas come from northern Colorado and do not represent every ecological region.
- Spatially nearby observations may share characteristics. The provided experiment uses stratified random splitting, not a spatial holdout, so it does not establish geographic transfer.
- Dataset age and collection context limit claims about current land-cover conditions.

## Attribution

Blackard, J. (1998). *Covertype* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C50K5N
