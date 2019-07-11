# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from CutoffPredictor import io
from CutoffPredictor import sql_queries
from CutoffPredictor import prep_data
from CutoffPredictor import prep_features
from CutoffPredictor import train_models
from CutoffPredictor import predictions

def process_wrapper(config):

    # Download the data from the database
    if config['download']:
        # Download the data from database
        [df_meter, df_location, df_occupant,
         df_volume, df_charge, df_cutoffs] = fetch_data(config)

        # Store in .csv files
        save_tables(config['table_dir'], df_meter, df_location, df_occupant,
                    df_volume, df_charge, df_cutoffs)


    # Prepare the data for analysis
    if config['prep_data']:
        # Read from .csv files
        [df_meter, df_location, df_occupant,
         df_volume, df_charge, df_cutoffs] = read_tables(config['table_dir'])

        # Prepare the data for analysis
        [df_meter, df_location, df_occupant,
         df_volume, df_charge, df_cutoffs] = prep_data(config, df_meter,
                                                       df_location, df_occupant,
                                                       df_volume, df_charge,
                                                       df_cutoffs)

        # Store in .csv files
        save_tables(config['clean_dir'], df_meter, df_location, df_occupant,
                    df_volume_align_clean, df_charge_align_clean, df_cutoffs)


    # Prepare feature tables for the models
    if config['prep_features']:
        # Read from .csv files
        [df_meter, df_location, df_occupant,
         df_volume, df_charge, df_cutoffs] = read_tables(config['clean_dir'])

        # Prepare feature tables for the models
        prep_features(config, df_meter, df_location, df_occupant,
                      df_volume, df_charge, df_cutoffs, 'train')


    # Train models and find the best one
    if config['train_models']:
        # Train models on the feature tables and select the best one
# maybe this should return the best model and parameters? But should save the info about them somewhere (which model type, which N_sample, which r)
        xxxx


    # Make prediction for current day
    if config['predict']:
        # Generate a feature table for current day
# This needs to read the saved N_sample and run prep_features_train(), and return the feature table
        xxxx
        feature_table = prep_features(config, df_meter, df_location,
                                      df_occupant, df_volume, df_charge,
                                      df_cutoffs, 'predict')
# run the model, saving predictions to a place that the dashboard knows about
        xxxx


    return 0


