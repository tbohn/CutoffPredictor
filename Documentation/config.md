# Configuration File

## Formatting of the file

The format of the configuration file is:

    [SECTION1]
    KEY1 : value1
    KEY2 : value2
    ...
    
    [SECTION2]
    ...

CutoffPredictor will store this information in a dictionary called `config`, which can be accessed as follows:
    `config[SECTION1][KEY1]` gives `value1`

Keys defined on one line of the config file can be referenced on subsequent lines via `%(KEY)s`, where `KEY` is the key. I.e., `%(KEY1)s` will be interpreted as `value1`.

## Major key-value options

The keys that need to be defined are:
1. `[STAGES]` section:
- These options tell CutoffPredictor which processing stages to run.
  - `DOWNLOAD` = True/False indicating whether to query the database
  - `PREP_DATA` = True/False indicating whether to prepare/clean the data
  - `PREP_FEATURES` = True/False indicating whether to prepare feature tables for model training
  - `TRAIN_MODELS` = True/False indicating whether to train the models
  - `PREDICT` = True/False indicating whether to make a prediction
  a. For periodic training of the models, run the following stages (set these to `True`): [`DOWNLOAD`, `PREP_DATA`, `PREP_FEATURES`, `TRAIN_MODELS`]
  b. For more frequent predictions, run: [`DOWNLOAD`, `PREP_DATA`, `PREDICT`]

2. `[PATHS]` section:
- Only 2 paths need to be set; all others will be derived from them:
  - `INSTALL_DIR` = location of CutoffPredictor code; just replace `install_path` with the corresponding path on your machine
  - `DATA_DIR` = top-level directory where data will be stored; just replace `data_path` with the corresponding path on your machine

3. `[DATABASE]` section:
- These are the database access parameters
  - `DBNAME` = database name
  - `SCHEMA` = schema name
  - `USER` = username
  - `PWD` = password
  - `HOST` = database hostname
  - `PORT` = port
  - `DATA_SET_ID` = the value of data_set_id column to use to get the correct subset of the data in queries

4. `[MAPPING]` section:
- Parameters for anything map-related
  - `GOOGLE_MAPS_API_KEY` = key for accessing the google maps api, used in the `DATA_PREP` stage to use geocoding to get lat/lon values from customer addresses; NOTE: an active google maps account is needed for this
  - `MAPBOX_ACCESS_TOKEN` = access token for the MapBox service, used in the `dashboard` to provide the map layers in the dashboard; NOTE: an active MapBox account is needed for this
  - `MAP_CENTER_LAT` = latitude of the center of the maps in the dashboard
  - `MAP_CENTER_LON` = longitude of the center of the maps in the dashboard

5. `[TRAINING]` section:
- Parameters for model training
  - `N_SAMPLE_LIST` = comma-separated list of sample window lengths to consider (in months)
  - `N_REALIZATIONS` = number of realizations of random window positions
  - `MODEL_TYPES` = comma-separated list of model types (no spaces); valid values are 'logistic_regression' or 'random_forest'
  - `COMBO_TYPES` = comma-separated list of feature combination types (no spaces); valid values are 'with_cutoff_prior' or 'omit_cutoff_prior'
  - `REF_DAY` = reference day (yyyy-mm-dd; records after this day are omitted)

6. `[PREDICTION]` section:
- Parameters for prediction
  - `COMBO_TYPE` = feature combination type desired for the prediction; valid values are 'with_cutoff_prior' or 'omit_cutoff_prior'
  - `REF_DAY` = reference day (yyyy-mm-dd; prediction is relative to this date)

