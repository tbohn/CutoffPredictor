# -*- coding: utf-8 -*-

# Make a prediction from the supplied feature table, given a specified model
def make_prediction(config, feature_table):

    today_str = config['PREDICTION']['REF_DAY']
    combo_type = config['PREDICTION']['COMBO_TYPE']

    # Read the relevant details of the best model from the best_model_info_file
    best_model_info_file = config['TRAINING']['BEST_MODEL_INFO_FILE']
    df_best_model_info = pd.read_csv(best_model_info_file)
    ref_day_str = df_best_model_info.loc['feature_combo_type' == combo_type, 'ref_day']
    model_type = df_best_model_info.loc['feature_combo_type' == combo_type, 'model_type']
    N_sample = df_best_model_info.loc['feature_combo_type' == combo_type, 'N_sample']
    r = df_best_model_info.loc['feature_combo_type' == combo_type, 'realization']
    oversample = df_best_model_info.loc['feature_combo_type' == combo_type, 'oversample']
    if oversample:
        oversample_str = '.os'
    else:
        oversample_str = ''

    # Make Prediction for predict customers as of today
    mode = 'predict'
    feature_dir = config['PATHS']['FEATURE_TABLE_DIR_PRED']
    prediction_dir = config['PATHS']['PREDICTIONS_DIR_PRED']
#NOTE build this from best_model_info?
#NOTE or just have it titled 'best'?
    infile = feature_dir + '/feature_table.{:s}.{:s}.best.csv'.format(today_str, combo_type)
    print('reading', infile)
    feature_table = pd.read_csv(infile)
            
    # Read saved model
#NOTE build this from best_model_info?
#NOTE or just have it titled 'best'?
    model_save_dir = config['PATHS']['SAVED_MODELS_DIR']
    model_file = model_save_dir + '/model.{:s}.{:s}.N{:02d}.{:s}.{:s}.r{:d}{:s}.sav'.format(model_type, today_str, N_sample, 'train', combo_type, r, oversample_str)
    print('loading', model_file)
    model = pickle.load(open(model_file, 'rb'))

    # Make prediction
    [predictions,
     probabilities,
     tmp, tmp, tmp, tmp, tmp] = apply_fitted_model(model,
                                                   feature_table,
                                                   feature_list[combo_type])
            
    # Save prediction
    pred_file = prediction_dir + '/predictions.' + '{:s}.{:s}.best.csv'.format(today_str, combo_type)
    print('writing to', pred_file)
    df_pred = pd.DataFrame(data=predictions, columns=['cutoff'])
    df_pred.to_csv(pred_file)
            
    # Save prediction
    prob_file = prediction_dir + '/probabilities.' + '{:s}.{:s}.best.csv'.format(today_str, combo_type)
    print('writing to', prob_file)
    df_prob = pd.DataFrame(data=probabilities, columns=['p_nocut', 'p_cutoff'])
    df_prob.to_csv(prob_file)

