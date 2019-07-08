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

    if config['download']:
        # Download the data from database
        [df_meter, df_location, df_occupant,
         df_volume, df_charge, df_cutoffs] = fetch_data(config)

        # Store in .csv files
        save_tables(config['table_dir'], df_meter, df_location, df_occupant,
                    df_volume, df_charge, df_cutoffs)

    if config['prep_data']:
        # Read from .csv files
        [df_meter, df_location, df_occupant,
         df_volume, df_charge, df_cutoffs] = read_tables(config['table_dir'])

        xxxx

        # Store in .csv files
        save_tables(config['clean_dir'], df_meter, df_location, df_occupant,
                    df_volume_align_clean, df_charge_align_clean, df_cutoffs)

    if config['prep_features']:
        # Read from .csv files
        [df_meter, df_location, df_occupant,
         df_volume, df_charge, df_cutoffs] = read_tables(config['clean_dir'])

        xxxx

        # Store in .csv files
        xxxx

    if config['train_models']:
        # Read from .csv files
        xxxx

        # Store in .csv files
        xxxx

    return xxx


