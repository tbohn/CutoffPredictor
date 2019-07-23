# CutoffPredictor
Tool for water utilities to monitor and predict customers' risk of service interruption

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
  - this is an Amazon Redshift database
2. [Configuration file](Documentation/config.md)
  - this can be derived from the template under config/

## Usage
1. Back end
    python CutoffPredictor.py config_file >& log_file
  where
    config_file = input config file, derived from the template
    log_file = log file to store progress messages

2. Dashboard
    python CPdashboard.py config_file >& log_file
  where
    config_file = input config file, derived from the template
    log_file = log file to store progress messages

Both the back end and the dashboard use the same [config file](Documentation/config.md).
