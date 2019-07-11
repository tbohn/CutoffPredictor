# -*- coding: utf-8 -*-
# Make Prediction for predict customers as of today
mode = 'predict'
feature_dir = '../data/feature_tables.predict'
prediction_dir = '../data/predictions.predict'
oversample_str = '.os'
for c in range(nCTypes):
    model_type = model_types[mmax_ar[c]]
    N_sample = N_samples[nmax_ar[c]]
    r = rmax_ar[c]
    print('prediction', combo_types[c])
    infile = feature_dir + '/feature_table.{:s}.{:s}.best.csv'.format(today_str, combo_types[c])
    print('reading', infile)
    feature_table = pd.read_csv(infile)
            
    # Read saved model
    model_file = model_save_dir + '/model.{:s}.{:s}.N{:02d}.{:s}.{:s}.r{:d}{:s}.sav'.format(model_type, today_str,
                                                                                             N_sample, 'train',
                                                                                             combo_types[c],
                                                                                             r, oversample_str)
    print('loading', model_file)
    model = pickle.load(open(model_file, 'rb'))

    # Make prediction
    [predictions, probabilities, tmp, tmp, tmp, tmp, tmp] = apply_fitted_model(model, feature_table,
                                                                               features[combo_types[c]])
            
    # Save prediction
    pred_file = prediction_dir + '/predictions.' + '{:s}.{:s}.best.csv'.format(today_str, combo_types[c])
    print('writing to', pred_file)
    df_pred = pd.DataFrame(data=predictions, columns=['cutoff'])
    df_pred.to_csv(pred_file)
            
    # Save prediction
    prob_file = prediction_dir + '/probabilities.' + '{:s}.{:s}.best.csv'.format(today_str, combo_types[c])
    print('writing to', prob_file)
    df_prob = pd.DataFrame(data=probabilities, columns=['p_nocut', 'p_cutoff'])
    df_prob.to_csv(prob_file)

