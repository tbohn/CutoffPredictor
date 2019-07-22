# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from CutoffPredictor import config
from CutoffPredictor import io
from CutoffPredictor import sql_queries
from CutoffPredictor import prep_data
from CutoffPredictor import prep_features
from CutoffPredictor import train_models
from CutoffPredictor import predict

def process_wrapper(config_file):

    # Read config_file
    if isinstance(config_file, dict):
        config = config_file
    else:
        config = read_config(config_file)

    # Download the data from the database
    if config['STAGES']['DOWNLOAD']:
        # Download the data from database
        [df_meter, df_location, df_occupant,
         df_volume, df_charge, df_cutoffs] = fetch_data(config)

        # Store in .csv files
        save_tables(config['PATHS']['DATA_TABLE_DIR'], df_meter, df_location,
                    df_occupant, df_volume, df_charge, df_cutoffs)


    # Prepare the data for analysis
    if config['STAGES']['PREP_DATA']:
        # Read from .csv files
        [df_meter, df_location,
         df_occupant, df_volume,
         df_charge, df_cutoffs] = read_tables(config['PATHS']['DATA_TABLE_DIR'])

        # Prepare the data for analysis
        [df_meter, df_location, df_occupant,
         df_volume, df_charge, df_cutoffs] = prep_data(config, df_meter,
                                                       df_location, df_occupant,
                                                       df_volume, df_charge,
                                                       df_cutoffs)

        # Store in .csv files
        save_tables(config['PATHS']['DATA_TABLE_CLEAN_DIR'], df_meter,
                    df_location, df_occupant, df_volume_align_clean,
                    df_charge_align_clean, df_cutoffs)


    # Prepare feature tables for the models
    if config['STAGES']['PREP_FEATURES']:
        # Read from .csv files
        [df_meter, df_location,
         df_occupant, df_volume,
         df_charge, df_cutoffs] = read_tables(config['PATHS']['DATA_TABLE_CLEAN_DIR'])

        # Prepare feature tables for the models
        prep_features(config, df_meter, df_location, df_occupant,
                      df_volume, df_charge, df_cutoffs, 'train')


    # Train models and find the best one
    if config['STAGES']['TRAIN_MODELS']:
        # Train models on the feature tables and select the best one
        train_and_compare_models(config)


    # Make prediction for current day
    if config['STAGES']['PREDICT']:
        # Generate a feature table for current day
        feature_table = prep_features(config, df_meter, df_location,
                                      df_occupant, df_volume, df_charge,
                                      df_cutoffs, 'predict')
        # Run the model, saving predictions to a place that the dashboard knows about
        make_prediction(config, feature_table)


    return 0


