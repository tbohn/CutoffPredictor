# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import h2o
from . import prep_features as prfe
from . import train_models as trmo

# Create a feature table and make a prediction, given a specified model
def make_prediction(config, df_meter, df_location, df_occupant, df_volume,
                    df_charge, df_cutoffs):

    mode = 'predict'
    ref_date_train = config['TRAINING']['REF_DATE']
    today_str = config['PREDICTION']['REF_DATE']
    option_cut_prior = config['PREDICTION']['FEATURES_CUT_PRIOR']
    option_metadata = config['PREDICTION']['FEATURES_METADATA']
    option_anom = config['PREDICTION']['FEATURES_ANOM']
    opt_str_train = '{:s}.{:s}.{:s}.{:s}.{:s}'.format('train', ref_date_train,
                                                      option_cut_prior,
                                                      option_metadata,
                                                      option_anom)
    opt_str_pred = '{:s}.{:s}.{:s}.{:s}.{:s}'.format(mode, today_str,
                                                     option_cut_prior,
                                                     option_metadata,
                                                     option_anom)

    # Read the relevant details of the best model from the best_model_info_file
    best_model_info_file = config['PATHS']['BEST_MODEL_INFO_FILE']
    df_best_model_info = pd.read_csv(best_model_info_file)
    nSamples = df_best_model_info['nSamples'].values[0]
    model_type = df_best_model_info['model_type'].values[0]
    instance_str = '{:s}.N{:02d}.{:s}'.format(opt_str_pred, nSamples,
                                              model_type)

    # To Do: put this logic and data into a separate function and/or file
    # Build feature list
    feature_list_basic = ['f_late', 'f_zero_vol']
    feature_list_anom = ['f_anom3_vol', 'f_manom3_vol']
    feature_list_metadata = ['cust_type_code', 'municipality', 'meter_size']
    feature_list_cut_prior = ['cut_prior']
    categoricals = feature_list_metadata.copy()
    categoricals.append(feature_list_cut_prior[0])
    feature_list = feature_list_basic.copy()
    if option_anom == 'anom':
        feature_list.append(feature_list_anom[0])
    elif option_anom == 'manom':
        feature_list.append(feature_list_anom[1])
    if option_metadata = 'with_meta':
        feature_list.extend(feature_list_metadata)
    if option_cut_prior = 'with_cut_prior':
        feature_list.append(feature_list_cut_prior[0])

    # Prepare feature table for the model prediction
    print('preparing feature table')
    feature_table = prfe.prep_features(config, df_meter, df_location,
                                       df_occupant, df_volume, df_charge,
                                       df_cutoffs, mode)

    # Read saved model
    model_save_dir = config['PATHS']['MODEL_SAVE_DIR']
    model_path_file_best = model_save_dir + '/model.' + opt_str_train + \
        '.best.path.txt'
    with open(model_path_file_best, 'r') as f:
        model_file = f.read()
    print('loading', model_file)
    model = h2o.loadModel(model_file)

    # Make prediction
    [probabilities, tmp, tmp, tmp, tmp, tmp, tmp, tmp] = \
        trmo.apply_model(model, feature_table, feature_list,
                         label, label_val_cut, categoricals,
                         score=False)
    df_prob = pd.DataFrame(data={'p_cutoff':probabilities})

    # Save the feature table as 'best' for consistency
    # (prep_features() already saved it, but we want to copy it to a simpler
    # filename for use by the dashboard)
    feature_dir = config['PATHS']['FEATURE_TABLE_DIR_PRED']
    outfile = feature_dir + '/feature_table.' + opt_str_pred + '.best.csv'
    feature_table.to_csv(outfile)
 
    # Save prediction
    prediction_dir = config['PATHS']['PREDICTIONS_DIR_PRED']
    prob_file1 = prediction_dir + '/probabilities.' + instance_str + '.csv'
    prob_file2 = prediction_dir + '/probabilities.' + opt_str_pred + '.best.csv'
    for prob_file in [prob_file1, prob_file2]:
        print('writing to', prob_file)
        df_prob.to_csv(prob_file)

