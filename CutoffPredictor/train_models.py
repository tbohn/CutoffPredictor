# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import datetime as dt
from sklearn.model_selection import train_test_split, cross_val_score
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import pickle
#from itertools import combinations
from shutil import copyfile

def prep_train_test_data(feature_table, feature_list, label, oversample=True):
    
    # Divide feature_table into cut and nocut subsets, then divide these into train/test subsets separately
    # to ensure that cut records appear in similar proportions in the train/test subsets
    df = {}
    df['cut'] = feature_table.loc[feature_table['cutoff'] == 1].copy()
    df['nocut'] = feature_table.loc[feature_table['cutoff'] == 0].copy()
    
    features_dict = {}
    labels_dict = {}
    features_train = {}
    labels_train = {}
    features_test = {}
    labels_test = {}
    for subset in df.keys():
        
        # Separate features from labels
        features_dict[subset] = df[subset][feature_list]
        labels_dict[subset] = df[subset][label]

        # Randomly select train:test ratio from data
        [features_train[subset], features_test[subset],
         labels_train[subset], labels_test[subset]] = train_test_split(features_dict[subset], labels_dict[subset], 
                                                                       test_size=0.2, random_state=0)

    # Recombine the training and testing portions of the cut and nocut subsets
    features_train_tot = features_train['cut'].copy()
    features_train_tot = features_train_tot.append(features_train['nocut'])
    labels_train_tot = labels_train['cut'].copy()
    labels_train_tot = labels_train_tot.append(labels_train['nocut'])
    features_test_tot = features_test['cut'].copy()
    features_test_tot = features_test_tot.append(features_test['nocut'])
    labels_test_tot = labels_test['cut'].copy()
    labels_test_tot = labels_test_tot.append(labels_test['nocut'])
        
    # Oversample the minority class to balance the data set
    if oversample:
        oversampler=SMOTE(random_state=0)
        os_features, os_labels = oversampler.fit_sample(features_train_tot, labels_train_tot)
        features_train_out = pd.DataFrame(os_features, columns=feature_list)
        labels_train_out = pd.DataFrame(os_labels, columns=[label])
    else:
        features_train_out = features_train_tot
        labels_train_out = pd.DataFrame(labels_train_tot, columns=[label])
    features_test_out = features_test_tot
    labels_test_out = pd.DataFrame(labels_test_tot, columns=[label])

#    print(labels_train_out.info())
#    print(labels_test_out.info())
    return features_train_out, labels_train_out, features_test_out, labels_test_out

def apply_and_score(model, features, labels):
    
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, probabilities[:,1].flatten())
    auc_roc = auc(false_positive_rate, true_positive_rate)
    
    return auc_roc

def train_and_test_model_try1(model_type, feature_table, feature_list, label):

    # Instantiate model
    if model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=10, random_state=0)
    elif model_type == 'logistic_regression':
        model = LogisticRegression(solver='lbfgs', random_state=0)

    # Set up inputs for model
    features = feature_table[feature_list]
    labels = feature_table[label]

    # Cross-validate model
    scores = cross_val_score(model, features, labels, scoring=apply_and_score, cv=5)
    auc_roc = np.mean(scores)

    # Train model
    model.fit(features, labels)
    
    # Get model predictions
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)

    # Other performance metric
    conf_mat = confusion_matrix(labels, predictions)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, probabilities[:,1].flatten())
    
    return model, predictions, probabilities, auc_roc, conf_mat, false_positive_rate, true_positive_rate, thresholds

def train_and_test_model(model_type, feature_table, feature_list, label):

    # Instantiate model
    if model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, random_state=0)
    elif model_type == 'logistic_regression':
        model = LogisticRegression(solver='lbfgs', random_state=0, penalty='l2')

    # Set up inputs for model
    features = feature_table[feature_list]
    labels = feature_table[label]

    # Cross-validate model
    scores = cross_val_score(model, features, labels, scoring=apply_and_score, cv=5)
    auc_roc = np.mean(scores)

    # Train model
    model.fit(features, labels)
    
    # Get model predictions
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)

    # Other performance metric
    conf_mat = confusion_matrix(labels, predictions)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, probabilities[:,1].flatten())
    
    return model, predictions, probabilities, auc_roc, conf_mat, false_positive_rate, true_positive_rate, thresholds

def apply_fitted_model(model, feature_table, feature_list, score=False, labels=None):
    
    # Set up inputs for model
    features = feature_table[feature_list]

    # Apply model
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)

    # Evaluate performance
    if score:
        conf_mat = confusion_matrix(labels, predictions)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, probabilities[:,1].flatten())
        auc_roc = auc(false_positive_rate, true_positive_rate)
    else:
        conf_mat = None
        false_positive_rate = None
        true_positive_rate = None
        thresholds = None
        auc_roc = None

    return [predictions, probabilities, conf_mat,
            false_positive_rate, true_positive_rate, thresholds, auc_roc]


# Find best-performing model, based on the average performance score across all realizations
def train_and_compare_models(config):

    # Train/Test various models

    ref_day = config['TRAINING']['REF_DAY']
    model_type_list_str = config['TRAINING']['MODEL_TYPES']
    model_types = model_type_list_str.split(',')
    combo_type_list_str = config['TRAINING']['COMBO_TYPES']
    combo_types = combo_type_list_str.split(',')
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
    categoricals = ['cust_type_code','municipality']
    N_sample_list_str = config['TRAINING']['N_SAMPLE_LIST']
    N_sample_list = N_sample_list_str.split(',')
    nN_samples = len(N_sample_list)
    nRealizations = config['TRAINING']['N_REALIZATIONS']
    feature_dir = config['PATHS']['FEATURE_TABLE_DIR_TRAIN']
    model_save_dir = config['PATHS']['MODEL_SAVE_DIR']
    model_perf_dir = config['PATHS']['MODEL_PERF_DIR']
    features_train = {}
    labels_train = {}
    features_test = {}
    labels_test = {}
    nCTypes = len(combo_types)
    nMTypes = len(model_types)
    auroc_array = np.zeros([nCTypes,nMTypes,nN_samples,nRealizations])
    mode = 'train'
    first = 1
    n = 0
    for N_sample in N_samples:
        features_train[N_sample] = {}
        labels_train[N_sample] = {}
        features_test[N_sample] = {}
        labels_test[N_sample] = {}
        r = 0
        for r in range(nRealizations):
        
            # Read feature table
            infile = feature_dir + '/feature_table.{:s}.N{:02d}.{:s}.r{:d}.csv'.format(ref_day, N_sample, mode, r)
            print('reading', infile)
            feature_table = pd.read_csv(infile)

            # Prepare train and test datasets
            oversample = True
            oversample_str = '.os'
            feature_list_all = feature_list=['with_cut_prior']
            [features_train[N_sample][r],
             labels_train[N_sample][r],
             features_test[N_sample][r],
             labels_test[N_sample][r]] = prep_train_test_data(feature_table,
                                                              feature_list_all,
                                                              'cutoff',
                                                              oversample)
            features_train[N_sample][r]['cutoff'] = labels_train[N_sample][r]['cutoff']
            features_test[N_sample][r]['cutoff'] = labels_test[N_sample][r]['cutoff']

            for c in range(nCTypes):
            
                m = 0
                for model_type in model_types:
                    features_to_use = feature_list[combo_types[c]].copy()
                    if model_type == 'logistic_regression':
                        for feat in categoricals:
                            features_to_use.remove(feat)

                    # Train and test model
                    [model,
                     predictions,
                     probabilities,
                     auc_roc,
                     conf_mat,
                     false_positive_rate,
                     true_positive_rate, 
                     thresholds] = train_and_test_model(model_type,
                                                        features_train[N_sample][r],
                                                        features_to_use,
                                                        'cutoff')
                    print(N_sample,r,combo_types[c],model_type,auc_roc)
            

                    # Write performance metrics to a file
                    perf_file = model_perf_dir + '/stats.cross_val.{:s}.{:s}.N{:02d}.{:s}.f{:s}.r{:d}{:s}.txt'.format(model_type, ref_day, N_sample, mode, combo_types[c], r, oversample_str)
                    if first:
                        code = 'w'
                    else:
                        code = 'a'
                    f = open(perf_file, code)

                    if first:
                        tmpstr = 'model_type,combo_type,N_sample,r,AUROC\n'
                        f.write(tmpstr)
                    tmpstr = '{:s},{:s},{:02d},{:d},{:.6f}\n'.format(model_type,combo_types[c],N_samples[n],r,auc_roc)
                    f.write(tmpstr)
                    f.close()

                    if model_type == 'random_forest':
                        # Write feature importances to a file
                        importances = model.feature_importances_
                        imp_file = model_perf_dir + '/importances.{:s}.{:s}.N{:02d}.{:s}.f{:s}.r{:d}{:s}.txt'.format(model_type, ref_day, N_sample, mode, combo_types[c], r, oversample_str)
                        f = open(perf_file, code)
                        tmpstr = ','.join(str(importances)) + '\n'
                        f.write(tmpstr)
                        f.close()
                
                    # Store AUC ROC
                    auroc_array[c,m,n,r] = auc_roc
            
                    first = 0

                    m += 1
            r += 1
        n += 1

        
    # Two sets of scores - one with prior cutoffs included as features, one without
    max_auroc_ar = np.empty(2)
    mmax_ar = np.empty(2, dtype=np.int)
    nmax_ar = np.empty(2, dtype=np.int)
    rmax_ar = np.empty(2, dtype=np.int)

    # Average auroc scores across realizations
    auroc_mean = np.mean(auroc_array, axis=3)

    # Find maximum average scores
    for c in range(nCTypes):
        max_auroc_ar[c] = np.max(auroc_mean[c])
        idx = auroc_mean[c].flatten().argmax()
        mmax_ar[c] = int(idx / nN_samples) 
        nmax_ar[c] = idx - mmax_ar[c] * nN_samples
        print(combo_types[c],model_types[mmax_ar[c]],N_samples[nmax_ar[c]])

#    # Print out metrics of all models just for reference
#    for c in range(nCTypes):
#        for m in range(nMTypes):
#            for n in range(nN_samples):
#                print(combo_types[c],model_types[m],N_samples[n],
#                      auroc_mean[c,m,n])

    # Show the best models
    for c in range(nCTypes):
        print(combo_types[c], 'best model:', 'alg:',
              model_types[mmax_ar[c]], 'N:', N_samples[nmax_ar[c]],
              'ROC AUC over train:', max_auroc_ar[c])

    # For the model with best perf averaged over 3 realizations, find the realization for which perf was best
    for c in range(nCTypes):
        rmax_ar[c] = np.argmax(auroc_array[c,mmax_ar[c],nmax_ar[c],:])


    best_model_data = {}
    for col in ['ref_day', 'feature_combo_type', 'model_type', 'N_sample',
                'realization', 'oversample']:
        best_model_data[col] = np.empty([nCTypes])

    conf_mat_list = []
    fpr_list = []
    tpr_list = []
    thresh_list = []
    auroc_list = []
    for c in range(nCTypes):

        mode = 'train'

        # Best model details
        combo_type = combo_types[c]
        model_type = model_types[mmax_ar[c]]
        N_sample = N_samples[nmax_ar[c]]
        r = rmax_ar[c]
        features_to_use = feature_list[combo_type].copy()
#        print(features_to_use)
        if model_type == 'logistic_regression':
            for feat in categoricals:
                features_to_use.remove(feat)

        # Save the relevant details of the best model
        best_model_info['ref_day'][c] = ref_day
        best_model_info['feature_combo_type'][c] = combo_type
        best_model_info['model_type'][c] = model_type
        best_model_info['N_sample'][c] = N_sample
        best_model_info['realization'][c] = r
        best_model_info['oversample'][c] = oversample

        # Re-fit best model to training set and save the model
        [model,
         predictions,
         probabilities,
         auc_roc,
         conf_mat,
         false_positive_rate,
         true_positive_rate,
         thresholds] = train_and_test_model(model_type,
                                            features_train[N_sample][r],
                                            features_to_use,
                                            'cutoff')
        print(N_sample,r,model_type,combo_type,auc_roc)

        # Save model
        model_file = model_save_dir + '/model.{:s}.{:s}.N{:02d}.{:s}.{:s}.r{:d}{:s}.sav'.format(model_type, ref_day, N_sample, mode, combo_type, r, oversample_str)
        pickle.dump(model, open(model_file, 'wb'))
        model_file_best = model_save_dir + '/model.{:s}.best.sav'.format(combo_type)
        copyfile(model_file, model_file_best)
    

        # Now validate best model on test set
        [predictions,
         probabilities,
         conf_mat,
         false_positive_rate,
         true_positive_rate,
         thresholds,
         auc_roc] = apply_fitted_model(model,
                                       features_test[N_sample][r],
                                       features_to_use,
                                       score=True,
                                       labels=labels_test[N_sample][r])
        print('Perf on test data: combo_type:', combo_type,
              'r', r, 'auroc:', auc_roc)
        conf_mat_list.append(conf_mat)
        fpr_list.append(false_positive_rate)
        tpr_list.append(true_positive_rate)
        thresh_list.append(thresholds)
        auroc_list.append(np.round(auc_roc,3))
        print(conf_mat)
        print(thresholds)
        print(false_positive_rate)
        print(true_positive_rate)
    
        if model_type == 'random_forest':
            # Write feature importances to a file
            importances = model.feature_importances_
            print('importances',importances)
            imp_file = model_perf_dir + '/importances.{:s}.{:s}.N{:02d}.{:s}.{:s}.r{:d}{:s}.txt'.format(model_type, ref_day, N_sample, mode, combo_type, r, oversample_str)
            f = open(perf_file, 'w')
            tmpstr = ','.join(str(importances)) + '\n'
            f.write(tmpstr)
            f.close()
        
        # Copy feature tables with best N_sample to 'best'
        feature_dir = '../data/feature_tables.{:s}'.format(mode)
        r_str_list = ['.r0','.r1','.r2']
        for r_str in r_str_list:
            infile = feature_dir + '/feature_table.{:s}.N{:02d}.{:s}{:s}.csv'.format(ref_day, N_sample, mode, r_str)
            outfile = feature_dir + '/feature_table.{:s}.{:s}{:s}.best.csv'.format(ref_day, combo_type, r_str)
            copyfile(infile, outfile)

    # Save the relevant details of the best model
    best_model_info_file = config['TRAINING']['BEST_MODEL_INFO_FILE']
    df_best_model_info = pd.DataFrame(data=best_model_info)
    df_best_model_info.to_csv(best_model_info_file)

    return 0
