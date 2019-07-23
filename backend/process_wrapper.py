# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from . import config as cfg
from . import io
from . import sql_queries as sq
from . import prep_data as prda
from . import prep_features as prfe
from . import train_models as trmo
from . import predict as pred

def process_wrapper(config_file):

    # Read config_file
    if isinstance(config_file, dict):
        config = config_file
    else:
        config = cfg.read_config(config_file)

    # Download the data from the database
    if config['STAGES']['DOWNLOAD']:

        # Download the data from database
        print('fetching data')
        [df_meter, df_location, df_occupant,
         df_volume, df_charge, df_cutoffs] = sq.fetch_data(config)

        # Store in .csv files
        print('saving to .csv files')
        io.save_tables(config['PATHS']['DATA_TABLE_DIR'], df_meter, df_location,
                       df_occupant, df_volume, df_charge, df_cutoffs)


    # Prepare the data for analysis
    if config['STAGES']['PREP_DATA']:
        
        # Read from .csv files
        print('reading from .csv files')
        [df_meter, df_location,
         df_occupant, df_volume,
         df_charge, df_cutoffs] = \
            io.read_tables(config['PATHS']['DATA_TABLE_DIR'])

        # Prepare the data for analysis
        print('preparing data for analysis')
        [df_meter, df_location, df_occupant,
         df_volume, df_charge, df_cutoffs] = \
            prda.prep_data(config, df_meter, df_location, df_occupant,
                           df_volume, df_charge, df_cutoffs)

        # Store in .csv files
        print('saving to .csv files')
        io.save_tables(config['PATHS']['DATA_TABLE_CLEAN_DIR'], df_meter,
                       df_location, df_occupant, df_volume,
                       df_charge, df_cutoffs)


    # Prepare feature tables for the models
    if config['STAGES']['PREP_FEATURES']:

        # Read from .csv files
        print('reading from .csv files')
        [df_meter, df_location,
         df_occupant, df_volume,
         df_charge, df_cutoffs] = \
            io.read_tables(config['PATHS']['DATA_TABLE_CLEAN_DIR'])

        # Prepare feature tables for the models
        print('preparing feature tables')
        prfe.prep_features(config, df_meter, df_location, df_occupant,
                           df_volume, df_charge, df_cutoffs, 'train')


    # Train models and find the best one
    if config['STAGES']['TRAIN_MODELS']:

        # Train models on the feature tables and select the best one
        print('training models')
        trmo.train_and_compare_models(config)


    # Make prediction for current day
    if config['STAGES']['PREDICT']:

        # Generate a feature table for current day
        print('preparing feature table')
        feature_table = prfe.prep_features(config, df_meter, df_location,
                                           df_occupant, df_volume, df_charge,
                                           df_cutoffs, 'predict')

        # Run the model, saving predictions to a place
        # that the dashboard knows about
        print('making prediciton')
        pred.make_prediction(config, feature_table)


    return 0


