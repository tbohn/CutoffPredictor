# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pickle
from . import prep_features as prfe
from . import train_models as trmo

# Create a feature table and make a prediction, given a specified model
def make_prediction(config, df_meter, df_location, df_occupant, df_volume,
                    df_charge, df_cutoffs):

    today_str = config['PREDICTION']['REF_DAY']
    combo_type = config['PREDICTION']['COMBO_TYPE']

    # Read the relevant details of the best model from the best_model_info_file
    best_model_info_file = config['PATHS']['BEST_MODEL_INFO_FILE']
    df_best_model_info = pd.read_csv(best_model_info_file)
    ref_day_str = \
        df_best_model_info.loc[df_best_model_info['feature_combo_type'] == \
        combo_type, 'ref_day'].values[0]
    model_type = \
        df_best_model_info.loc[df_best_model_info['feature_combo_type'] == \
        combo_type, 'model_type'].values[0]
    N_sample = \
        df_best_model_info.loc[df_best_model_info['feature_combo_type'] == \
        combo_type, 'N_sample'].values[0]
    r = df_best_model_info.loc[df_best_model_info['feature_combo_type'] == \
        combo_type, 'realization'].values[0]
    mode = 'predict'
    feature_list = {
        'with_cut_prior': [
            'late_frac','zero_frac_vol','cut_prior','nCutPrior',
            'cust_type_code','municipality','meter_size','skew_vol',
            'n_anom3_vol','max_anom_vol',
            'n_anom3_local_vol','max_anom_local_vol',
            'n_anom3_vol_log','max_anom_vol_log',
            'n_anom3_local_vol_log','max_anom_local_vol_log',
        ],
        'omit_cut_prior': [
            'late_frac','zero_frac_vol',
            'cust_type_code','municipality','meter_size','skew_vol',
            'n_anom3_vol','max_anom_vol',
            'n_anom3_local_vol','max_anom_local_vol',
            'n_anom3_vol_log','max_anom_vol_log',
            'n_anom3_local_vol_log','max_anom_local_vol_log',
        ],
    }

    # Prepare feature table for the model prediction
    print('preparing feature table')
    feature_table = prfe.prep_features(config, df_meter, df_location,
                                       df_occupant, df_volume, df_charge,
                                       df_cutoffs, mode)

    # Read saved model
    model_save_dir = config['PATHS']['MODEL_SAVE_DIR']
    ref_day_train = config['TRAINING']['REF_DAY']
    model_file = model_save_dir + '/model.{:s}.{:s}.{:s}.best.sav' \
        .format('train', ref_day_train, combo_type)
    print('loading', model_file)
    model = pickle.load(open(model_file, 'rb'))

    # Make prediction
    [predictions,
     probabilities,
     tmp, tmp, tmp, tmp, tmp] = \
        trmo.apply_fitted_model(model, feature_table, feature_list[combo_type])
    df_pred = pd.DataFrame(data=predictions, columns=['cutoff'])
    df_prob = pd.DataFrame(data=probabilities, columns=['p_nocut', 'p_cutoff'])
            

    # Save the feature table as 'best' for consistency
    # (prep_features() already saved it, but we want to copy it to a simpler
    # filename for use by the dashboard)
    feature_dir = config['PATHS']['FEATURE_TABLE_DIR_PRED']
    outfile = feature_dir + '/feature_table.{:s}.{:s}.{:s}.best.csv' \
        .format(mode, today_str, combo_type)
 
    # Save prediction
    prediction_dir = config['PATHS']['PREDICTIONS_DIR_PRED']
    pred_file1 = prediction_dir + '/predictions.' + \
        '{:s}.{:s}.{:s}.best.csv'.format(mode, today_str, combo_type)
    pred_file2 = prediction_dir + '/predictions.' + \
        '{:s}.{:s}.{:s}.N{:02d}.{:s}.csv'.format(model_type, mode, today_str,
                                                 N_sample, combo_type)
    for pred_file in [pred_file1, pred_file2]:
        print('writing to', pred_file)
        df_pred.to_csv(pred_file)
    prob_file1 = prediction_dir + '/probabilities.' + \
        '{:s}.{:s}.{:s}.best.csv'.format(mode, today_str, combo_type)
    prob_file2 = prediction_dir + '/probabilities.' + \
        '{:s}.{:s}.{:s}.N{:02d}.{:s}.csv'.format(model_type, mode, today_str,
                                                 N_sample, combo_type)
    for prob_file in [prob_file1, prob_file2]:
        print('writing to', prob_file)
        df_prob.to_csv(prob_file)

