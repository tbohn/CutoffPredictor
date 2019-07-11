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


# Train/Test various models
import pickle
from itertools import combinations
from shutil import copyfile

today_str = '2019-01-01'
today = pd.Timestamp(today_str)
model_types = ['random_forest','logistic_regression']
N_samples = [4,6,8,10,12,15,18,21,24]
combo_types = ['with_cut_prior','omit_cut_prior']
features = {
    'with_cut_prior': [
        'late_frac','zero_frac_vol','cut_prior','nCutPrior',
        'cust_type_code','municipality','meter_size','skew_vol',
        'n_anom3_vol','max_anom_vol','n_anom3_local_vol','max_anom_local_vol',
        'n_anom3_vol_log','max_anom_vol_log','n_anom3_local_vol_log','max_anom_local_vol_log',
    ],
    'omit_cut_prior': [
        'late_frac','zero_frac_vol',
        'cust_type_code','municipality','meter_size','skew_vol',
        'n_anom3_vol','max_anom_vol','n_anom3_local_vol','max_anom_local_vol',
        'n_anom3_vol_log','max_anom_vol_log','n_anom3_local_vol_log','max_anom_local_vol_log',
    ],
}
categoricals = ['cust_type_code','municipality']
nN_samples = len(N_samples)
nRealizations = 3
feature_dir = '../data/feature_tables.train'
model_save_dir = '../data/saved_models'
model_perf_dir = '../data/model_perf'
prediction_dir = '../data/predictions.train'
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
        infile = feature_dir + '/feature_table.{:s}.N{:02d}.{:s}.r{:d}.csv'.format(today_str, N_sample, mode, r)
        print('reading', infile)
        feature_table = pd.read_csv(infile)

        # Prepare train and test datasets
        oversample = True
        oversample_str = '.os'
        [features_train[N_sample][r],
         labels_train[N_sample][r],
         features_test[N_sample][r],
         labels_test[N_sample][r]] = prep_train_test_data(feature_table, features[combo_types[0]], 'cutoff', oversample)
        features_train[N_sample][r]['cutoff'] = labels_train[N_sample][r]['cutoff']
        features_test[N_sample][r]['cutoff'] = labels_test[N_sample][r]['cutoff']

        for c in range(nCTypes):
            
            m = 0
            for model_type in model_types:
                features_to_use = features[combo_types[c]].copy()
                if model_type == 'logistic_regression':
                    for feat in categoricals:
                        features_to_use.remove(feat)

                # Train and test model
                [model, predictions, probabilities,
                 auc_roc, conf_mat, false_positive_rate,
                 true_positive_rate, thresholds] = train_and_test_model(model_type, features_train[N_sample][r],
                                                                        features_to_use, 'cutoff')
                print(N_sample,r,combo_types[c],model_type,auc_roc)
            

                # Write performance metrics to a file
                perf_file = model_perf_dir + '/stats.cross_val.{:s}.{:s}.N{:02d}.{:s}.f{:s}.r{:d}{:s}.txt'.format(model_type, today_str,
                                                                                                                  N_sample, mode, combo_types[c],
                                                                                                                  r, oversample_str)
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
                    imp_file = model_perf_dir + '/importances.{:s}.{:s}.N{:02d}.{:s}.f{:s}.r{:d}{:s}.txt'.format(model_type, today_str,
                                                                                                                 N_sample, mode, combo_types[c],
                                                                                                                 r, oversample_str)
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

        

# Find best-performing model, based on the average performance score across all realizations
# Average auroc scores across realizations
max_auroc_ar = np.empty(2)
mmax_ar = np.empty(2, dtype=np.int)
nmax_ar = np.empty(2, dtype=np.int)
rmax_ar = np.empty(2, dtype=np.int)
auroc_mean = np.mean(auroc_array, axis=3)
print(auroc_mean)
max_auroc_ar[0] = np.max(auroc_mean[0])
idx = auroc_mean[0].flatten().argmax()
mmax_ar[0] = int(idx / nN_samples) 
nmax_ar[0] = idx - mmax_ar[0] * nN_samples
print('with_cutoffs',model_types[mmax_ar[0]],N_samples[nmax_ar[0]])
max_auroc_ar[1] = np.max(auroc_mean[1])
idx = auroc_mean[1].flatten().argmax()
mmax_ar[1] = int(idx / nN_samples) 
nmax_ar[1] = idx - mmax_ar[1] * nN_samples
print('without_cutoffs',model_types[mmax_ar[1]],N_samples[nmax_ar[1]])
# Print out metrics of all models just for reference
for c in range(nCTypes):
    for m in range(nMTypes):
        for n in range(nN_samples):
            print(combo_types[c],model_types[m],N_samples[n],auroc_mean[c,m,n])
# Show the best models
print('best of models with cutoff features:','alg:',model_types[mmax_ar[0]],'N:',N_samples[nmax_ar[0]],
      'ROC AUC over train:',max_auroc_ar[0])
print('best of models without cutoff features:','alg:',model_types[mmax_ar[1]],'N:',N_samples[nmax_ar[1]],
      'ROC AUC over train:',max_auroc_ar[1])

# For the model with best perf averaged over 3 realizations, find the realization for which perf was best
for c in range(nCTypes):
    rmax_ar[c] = np.argmax(auroc_array[c,mmax_ar[c],nmax_ar[c],:])


feature_dir = '../data/feature_tables.current'
prediction_dir = '../data/predictions.current'
conf_mat_list = []
fpr_list = []
tpr_list = []
thresh_list = []
auroc_list = []
for c in range(nCTypes):

    # Re-fit model to training set and save the model
    mode = 'train'
    model_type = model_types[mmax_ar[c]]
    N_sample = N_samples[nmax_ar[c]]
    r = rmax_ar[c]
    features_to_use = features[combo_types[c]].copy()
#    print(features_to_use)
    if model_type == 'logistic_regression':
        for feat in categoricals:
            features_to_use.remove(feat)
    [model, predictions, probabilities,
     auc_roc, conf_mat, false_positive_rate,
     true_positive_rate, thresholds] = train_and_test_model(model_type, features_train[N_sample][r],
                                                            features_to_use, 'cutoff')
    print(N_sample,r,model_type,combo_types[c],auc_roc)
    model_file = model_save_dir + '/model.{:s}.{:s}.N{:02d}.{:s}.{:s}.r{:d}{:s}.sav'.format(model_type, today_str,
                                                                                            N_sample, mode, combo_types[c],
                                                                                            r, oversample_str)
    pickle.dump(model, open(model_file, 'wb'))
    
    # Now validate best model on test set
    [predictions, probabilities, conf_mat,
     false_positive_rate, true_positive_rate,
     thresholds, auc_roc] = apply_fitted_model(model, features_test[N_sample][r], features_to_use,
                                               score=True, labels=labels_test[N_sample][r])
    print('Perf on test data: combo_type:', combo_types[c], 'r', r, 'auroc:', auc_roc)
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
        imp_file = model_perf_dir + '/importances.{:s}.{:s}.N{:02d}.{:s}.{:s}.r{:d}{:s}.txt'.format(model_type, today_str,
                                                                                                    N_sample, mode, combo_types[c],
                                                                                                    r, oversample_str)
        f = open(perf_file, 'w')
        tmpstr = ','.join(str(importances)) + '\n'
        f.write(tmpstr)
        f.close()
        
    # Copy feature tables with best N_sample to 'best'
    for mode in ['train','current']:
        feature_dir = '../data/feature_tables.{:s}'.format(mode)
        if mode == 'train':
            r_str_list = ['.r0','.r1','.r2']
        else:
            r_str_list = ['']
        for r_str in r_str_list:
            infile = feature_dir + '/feature_table.{:s}.N{:02d}.{:s}{:s}.csv'.format(today_str, N_sample, mode, r_str)
            outfile = feature_dir + '/feature_table.{:s}.{:s}{:s}.best.csv'.format(today_str, combo_types[c], r_str)
            copyfile(infile, outfile)

    fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(6,3))
    ax.plot(fpr_list[c], tpr_list[c])
    ax.plot([0,1],[0,1],'k--',linewidth=0.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.text(0.05, 0.92, 'AUC: ' + str(auroc_list[c]))
    plt.tight_layout()
    fig.show()
    filename = 'roc.{:s}.{:s}.best.png'.format(today_str, combo_types[c])
    fig.savefig(filename)

