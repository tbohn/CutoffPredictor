# CutoffPredictor
Tool for water utilities to monitor and predict customers' risk of service interruption

For details, please read the [User Guide](Documentation/CutoffPredictor_v2.0_Description_and_User_Guide.pdf) [Utility database](Documentation/database_tables.md)

## Overview

CutoffPredictor consists of:
1. Back end
  - A machine learning model is trained periodically, via the following steps:
    - query the utility's database
    - clean the data and prepare features
    - train several machine learning models over a range of parameters
    - select the best-performing model
  - On a monthly or daily basis, the user can make a prediction:
    - query the utility's database (to get recent records)
    - clean the data and prepare features
    - apply the best-performing model parameters from the training phase
2. Dashboard
  - Plotly/Dash app accessible via a web browser (127.0.0.1:8050)
  - Reads the prediction and displays interactive analytics

## Requirements

Accounts:
- An account to access the utility database
- Google Maps
- MapBox

Software requirements:
- PostgreSQL
- Flask
- Java 64-bit JDK, version 11 (required by Python h2o package)
- Plotly/Dash
- Python 3.4 or later; with the following packages:
  - pandas
  - numpy
  - scipy
  - math
  - datetime
  - requests
  - psycopg2 (PostgreSQL interface)
  - sklearn (for ML utilities; h2o ML models are favored over sklearn)
  - imblearn (for SMOTE oversampling)
  - h2o (for ML models)
  - shutil (for copying files)
  - argparse
  - os
  - flask
  - plotly
  - dash
 
## Code

Code is stored in the following directory structure:
- CutoffPredictor.py
  - This is the back-end app
- CPdashboard.py
  - This is the dashboard app 
- backend/
  - back-end functions
- config/
  - template for input config file
- dashboard/
  - dashboard functions
- Documentation/
  - documentation files

## Data
The user must supply CutoffPredictor with a top-level directory for storing data, which we'll call `DATA_DIR`.  CutoffPredictor expects the following data directories to exist under `DATA_DIR`:
- data_tables/
  - tables queried from the utility database are stored here as .csv files
- data_tables_clean/
  - cleaned versions of the database tables
- feature_tables.train/
  - tables of features computed for the training/testing period
- predictions.train/
  - tables of predictions and probabilities for the training/testing period
- saved_models/
  - best-performing models saved here as json files
- model_perf/
  - model performance statistics
- feature_tables.pred/
  - tables of features computed for the prediction period
- predictions.pred/
  - tables of predictions and probabilities for the prediction period

## [Inputs](Documentation/inputs.md)

1. [Utility database](Documentation/database_tables.md)

  - this is a SQL database (CutoffPredictor uses PostgreSQL)

2. [Configuration file](Documentation/config.md)

  - this can be derived from the template under config/

## Usage
1. Back end

        python CutoffPredictor.py config_file >& log_file

    where

  - config_file = input config file, derived from the template.
  - log_file = log file to store progress messages

2. Dashboard

        python CPdashboard.py config_file >& log_file

    where

  - config_file = input config file, derived from the template
  - log_file = log file to store progress messages

Both the back end and the dashboard use the same [config file](Documentation/config.md).

## Recommended Process Flow
### A. Update/Retrain Models (monthly or less frequently)
1. Prepare model inputs (this can be done in a single step, with a single config file).
  - Stages (`[STAGES]` section of config file):
    - Query database (`DOWNLOAD` = `TRUE`)
    - Prepare/clean data (`PREP_DATA` = `TRUE`)
    - Prepare features (`PREP_FEATURES` = `TRUE`); this will prepare features for a range of values of window length `nSamples`, controlled by `N_SAMPLE_LIST`. These feature tables will be saved as csv files in the `feature_tables.train` subdirectory under `DATA_DIR`.
    - All other options in the `STAGES` section should be set to `FALSE`.
  - Training options (`[TRAINING]` section of config file):
    - `N_SAMPLE_LIST`: list of window lengths (`nSamples`) to consider

2. Train models (this must be done separately for each desired set of model features)
  - Stages (`[STAGES]` section of config file):
    - Train models (`TRAIN_MODELS` = `TRUE`); this will do a search over all values of `nSamples` and across all specified model types to find the best-performing model and `nSamples` to use for predictions. In addition, for `random_forest`, the optimal value of `max_depth` will be found.
    - All other options in the `STAGES` section should be set to `FALSE`.
  - Training options (`[TRAINING]` section of config file):
    - `REF_DATE`: indicates the final date of the training period; all records prior to and including this date will be used in training the models.
    - `N_SAMPLE_LIST`: list of window lengths (`nSamples`) to consider
    - `MODEL_TYPES`: list of model types to explore; currently only `logistic_regression` and `random_forest` are supported.
    - `MAX_DEPTH_LIST`: list of values of `max_depth` (used in `random_forest` model) to explore; the minimum value should generally be set to 3 and the maximum should be somewhere between 5 and 20.
    - `FEATURES_CUT_PRIOR`: indicates whether to include among the feature set a boolean flag signifying whether a customer has had a prior cutoff; this is only possible for utilities that have recorded such information (not all do this); valid values are `'no_cut_prior'` and `'with_cut_prior'`.
    - `FEATURES_METADATA`: indicates whether to include among the feature set the three customer metadata variables `'cust_type_code'`, `'municipality'`, and `'meter_size'`; valid values are `'no_meta'` and `'with_meta'`.
    - `FEATURES_ANOM`: indicates which volume anomaly metric to include among the feature set; valid values are `'anom'` (use the simple anomaly feature `'f_anom3_vol'`), `'manom'` (use the monthly anomaly feature `'f_manom3_vol'`), and `'none'`.

### B. Model Predictions (monthly to daily)
This stage can be performed separately for each reference date and feature set over which models were trained in part A.
  - Stages (`[STAGES]` section of config file):
    - Make prediction (`PREDICT` = `TRUE`); this will use the best model/nSamples combination found in part A.2. for the given feature set
    - All other options in the `STAGES` section should be set to `FALSE`.
  - Training options (`[TRAINING]` section of config file):
    - `REF_DATE`: indicates the final date of the training period; the model and value of `nSamples` used will be those from stage A.2. for the best-performing model for this reference date and the given feature set.
  - Prediction options (`[PREDICTION]` section of config file):
    - `REF_DATE`: indicates the 'current' date (normally this would be the actual current date, but it can be set to a date in the past to compare previous predictions to actual outcomes); predictions will be made based on values of metrics computed from the `nSamples` months prior to `REF_DATE`, where `nSamples` is the best-performing value found in stage A.2. for the given feature set.
    - `FEATURES_CUT_PRIOR`: this should be set to the value used in part A.2.
    - `FEATURES_METADATA`: this should be set to the value used in part A.2.
    - `FEATURES_ANOM`: this should be set to the value used in part A.2.

### C. Dashboard (any time)
This stage can be performed separately for each reference date and feature set over which models were trained in part A and for which a prediction was made in part B.
  - Stages (`[STAGES]` section of config file):
    - All options in the `STAGES` section will be ignored
  - Prediction options (`[PREDICTION]` section of config file):
    - `REF_DATE`: indicates the 'current' date (normally this would be the actual current date, but it can be set to a date in the past to compare previous predictions to actual outcomes); predictions must have been made in stage B for this date and feature set.
    - `FEATURES_CUT_PRIOR`: this should be set to the value used in part B.
    - `FEATURES_METADATA`: this should be set to the value used in part B.
    - `FEATURES_ANOM`: this should be set to the value used in part B.

