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
import h2o
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator

def prep_train_test_data(feature_table, feature_list, categorical_features,
                         label, label_val_nocut, label_val_cut, rebalance=None):
 
    # H2O doesn't require dummy variables to be supplied for categorical
    # features; it does this internally in generalized linear models and
    # treats categoricals correctly in random forests.
    # Otherwise, we would need to do the following:
#    for col in categorical_features:
#        tmp = pd.get_dummies(df[col], drop_first=True)
#        feature_table = pd.concat([feature_table, tmp])

    # H2O doesn't require us to standardize the variables externally;
    # it does it internally.
    # Otherwise, insert standardization code here for all continuous variables

    # Divide feature_table into cut and nocut subsets, then divide these into
    # train/test subsets separately to ensure that cut records appear in
    # similar proportions in the train/test subsets
    df = {}
    df['nocut'] = feature_table.loc[feature_table[label] == label_val_nocut].copy()
    df['cut'] = feature_table.loc[feature_table[label] == label_val_cut].copy()

    # We additionally want to ensure that both the first cutoffs of all
    # occupants and subsequent cutoffs (for those who have multiple cutoffs)
    # are also split proportionally into train/test subsets; this helps ensure
    # that train and test subsets contain similar proportions of first and
    # subsequent cutoffs (in case these behave differently).
    # First, select the first cutoff (chronologically) from
    # each occupant; this will have the max segment value for that occupant
    # due to our clipping the windows in reverse chronological order.
    df_tmp = df_cut.groupby('occupant_id').agg({'segment':'max'})
        .reset_index().rename(columns={'segment':'segmax'})
    df_cut = df_cut.merge(df_tmp, on='occupant_id')
    df['cut0'] = df_cut.loc[df_cut['segment'] == df_cut['segmax']].copy()
    # All other cutoffs are subsequent to the first ones
    df['cut1'] = df_cut.loc[df_cut['segment'] != df_cut['segmax']].copy()
    
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

        if len(df[subset].index) > 1:

            # Randomly select train:test ratio from data
            [features_train[subset], features_test[subset],
             labels_train[subset], labels_test[subset]] = \
                train_test_split(features_dict[subset], labels_dict[subset], 
                                 test_size=0.2, random_state=0)

        elif len(df[subset].index) == 1:

            # Put the single record in the train subset;
            # test subset will be empty dataframe
            features_train[subset] = features_dict[subset].copy()
            labels_train[subset] = labels_dict[subset].copy()
            features_test[subset] = pd.DataFrame(columns=feature_list)
            labels_test[subset] = pd.DataFrame(columns=[label])

        else:

            # Everything becomes an empty dataframe
            features_train[subset] = pd.DataFrame(columns=feature_list)
            labels_train[subset] = pd.DataFrame(columns=[label])
            features_test[subset] = pd.DataFrame(columns=feature_list)
            labels_test[subset] = pd.DataFrame(columns=[label])

    # Recombine the training and testing portions of the cut and nocut subsets
    if len(features_train['cut0'].index) > 0:
        features_train_tot = features_train['cut0'].copy()
        labels_train_tot = labels_train['cut0'].copy()
    else:
        print('ERROR: no cutoffs found in training set')
    if len(features_test['cut0'].index) > 0:
        features_test_tot = features_test['cut0'].copy()
        labels_test_tot = labels_test['cut0'].copy()
    else:
        print('ERROR: no cutoffs found in test set')
    if len(features_train['cut1'].index) > 0:
        features_train_tot = features_train_tot.append(features_train['cut1'],
                                                       ignore_index=True)
        labels_train_tot = labels_train_tot.append(labels_train['cut1'],
                                                   ignore_index=True)
    if len(features_test['cut1'].index) > 0:
        features_test_tot = features_test_tot.append(features_test['cut1'],
                                                     ignore_index=True)
        labels_test_tot = labels_test_tot.append(labels_test['cut1'],
                                                 ignore_index=True)
    nCut = len(features_train_tot.index)
    features_train_tot = features_train_tot.append(features_train['nocut'],
                                                   ignore_index=True)
    nTot = len(features_train_tot.index)
    nNoCut = nTot - nCut
    labels_train_tot = labels_train_tot.append(labels_train['nocut'],
                                               ignore_index=True)
    features_test_tot = features_test_tot.append(features_test['nocut'],
                                                 ignore_index=True)
    labels_test_tot = labels_test_tot.append(labels_test['nocut'],
                                             ignore_index=True)

    # Rebalance the labels in the training subset
    if rebalance == 'oversample':
        # Oversample the minority class
        oversampler=SMOTE(random_state=0)
        os_features, os_labels = oversampler.fit_sample(features_train_tot,
                                                        labels_train_tot)
        features_train_out = pd.DataFrame(os_features, columns=feature_list)
        labels_train_out = pd.DataFrame(os_labels, columns=[label])
    elif rebalance == 'weights':
        # Create vector of weights based on numbers of records
        features_train_out = features_train_tot.copy()
        labels_train_out = labels_train_tot.copy()
        weights = labels_train_out.map({label_val_cut:(0.5 * nTot / nCut),
                                        label_val_nocut:(0.5 * nTot / nNoCut)})
        labels_train_out = pd.DataFrame({label:labels_train_out,
                                         'weights':weights})
    else:
        # No rebalancing
        features_train_out = features_train_tot
        labels_train_out = labels_train_tot

    # Test subset is unchanged
    features_test_out = features_test_tot
    labels_test_out = labels_test_tot

    return [features_train_out, labels_train_out, features_test_out,
           labels_test_out]


def set_up_H2OFrame(df, feature_list, categoricals=None, label=None):

    # Cast categorical column values as strings
    # prepend 'cat' as necessary to ensure they are interpreted as strings
    if categoricals != None:
        categoricals = [x for x in categoricals if x in feature_list]
        for col in categoricals:
            if df[col].dtype == np.float:
                df[col] = 'cat' + df[col].astype(str)

    # Denote categorical columns as such via h2o's asfactor() method
    df_H2O = h2o.H2OFrame(python_obj=df)
    if categoricals != None:
        for col in categoricals:
            df_H2O[col] = df_H2O[col].asfactor()
    if label != None:
        df_H2O[label] = df_H2O[label].asfactor()

    return df_H2O


def apply_model(model, feature_table, feature_list, label, label_val_cut,
                categoricals=None, score=False):

    # Create H2OFrame
    df = set_up_H2OFrame(feature_table, feature_list, categoricals, label)

    # Get model predictions
    tmp = model.predict(test_data=df)
    colname = 'p' + str(label_val_cut)
    probabilities = np.asarray(tmp[colname].as_data_frame()).flatten()

    # Get model performance metrics
    if score and label != None:
        perf = model.model_performance(df)
        auc_roc = perf.auc()
        # To Do: figure out how to get h2o to return the FPR, TPR,
        # and threshold values for all possible threshold values
        # so that we can easily plot the ROC curve and label the
        # thresholds of each point. For now, just set thresholds list
        # to None and use h2o's default thresholds for FPR and TPR
        thresholds = None
        FPR = perf.fpr()
        TPR = perf.tpr()
        F1 = perf.F1()
        conf_mat = perf.confusion_matrix(metrics=['f1'])
        R2 = perf.r2()
    else:
        auc_roc = None
        thresholds = None
        FPR = None
        TPR = None
        F1 = None
        conf_mat = None
        R2 = None

    return [probabilities, auc_roc, thresholds, FPR, TPR, F1, conf_mat, R2]

    
def train_and_test_model(model_type, feature_table_train, feature_table_test,
                         feature_list, label, label_val_cut, categoricals=None,
                         wtcol=None, regularization=False, grid_search=False):

    # Set up inputs for model
    df_train = set_up_H2OFrame(feature_table_train, feature_list,
                               categoricals, label)
    df_test = set_up_H2OFrame(feature_table_test, feature_list,
                              categoricals, label)

    # Cross-validation parameters
    nfolds = 5
    fold_asgmnt = 'Stratified'

    # Instantiate model
    if model_type == 'random_forest':
        if grid_search == False:
            model = H2ORandomForestEstimator(ntrees=100, max_depth=20,
                                             training_frame=df_train,
                                             validation_frame=df_test,
                                             nfolds=nfolds,
                                             fold_assignment=fold_asgmnt,
                                             seed=1)
        # To Do: figure out why h2o's grid_search didn't work
#        else:
#            model_grid = H2OGridSearch(model=H2ORandomForestEstimator,
#                                       grid_id='model_grid',
#                                       hyper_params={
#                                           'max_depth':list(range(3,21))
#                                       })
    elif model_type == 'logistic_regression':
        if regularization == True:
            model = H2OGeneralizedLinearEstimator(family='Binomial',
                                                  link='Logit',
                                                  lambda_search=True,
                                                  training_frame=df_train,
                                                  validation_frame=df_test,
                                                  nfolds=nfolds,
                                                  fold_assignment=fold_asgmnt,
                                                  seed=1)
        else:
            model = H2OGeneralizedLinearEstimator(family='Binomial',
                                                  link='Logit',
                                                  lambda_=0,
                                                  solver='IRLSM',
                                                  compute_p_values=True,
                                                  training_frame=df_train,
                                                  validation_frame=df_test,
                                                  nfolds=nfolds,
                                                  fold_assignment=fold_asgmnt,
                                                  seed=1)

    # Train and cross-validate model
    # Here's where we do the grid search over max_depth for random forest
    if model_type == 'random_forest' and grid_search == True:
#        model_grid.train(x=feature_list, y=label, training_frame=df_train,
#                         weights_column=wtcol, ntrees=100,
#                         validation_frame=df_test, nfolds=nfolds,
#                         fold_assignment=fold_asgmnt, seed=1)
#        # Get the best-performing model
#        model_gridperf = model_grid.get_grid(sort_by='auc', decreasing=True)
#        model = model_gridperf.models[0]
        max_depth_list = list(range(3,21))
        nMaxDepth = len(max_depth_list)
        models = []
        aucs = np.zeros([nMaxDepth])
        for i in range(nMaxDepth):
            model = H2ORandomForestEstimator(ntrees=100,
                                             max_depth=max_depth_list[i],
                                             training_frame=df_train,
                                             validation_frame=df_test,
                                             nfolds=nfolds,
                                             fold_assignment=fold_asgmnt,
                                             seed=1)
            model.train(x=feature_list, y=label, training_frame=df_train,
                        weights_column=wtcol)
            models.append(model)
            aucs[i] = model.auc(xval=True)
        imax = np.argmax(aucs)
        model = models[imax]
    else:
        model.train(x=feature_list, y=label, training_frame=df_train,
                    weights_column=wtcol)

    # Model parameters
    if model_type == 'logistic_regression':
        coefs = model._model_json['output']['coefficients_table'].as_data_frame()
    elif model_type == 'random_forest':
        coefs = model.varimp(use_pandas=True)

    # Get model reults and performance metrics for the training
    # and test subsets, and also for cross-validation
    probabilities = {}
    auc_roc = {}
    FPR = {}
    TPR = {}
    F1 = {}
    conf_mat = {}
    R2 = {}
    for dataset in ['train', 'xval', 'test']:
        if dataset in ['train', 'test']:
            if dataset == 'train':
                feature_table = feature_table_train
            elif dataset == 'test':
                feature_table = feature_table_test

            [
                probabilities[dataset],
                auc_roc[dataset],
                thresholds,
                FPR[dataset],
                TPR[dataset],
                F1[dataset],
                conf_mat[dataset],
                R2[dataset]
            ] = apply_model(model, feature_table, feature_list,
                            label, label_val_cut, categoricals,
                            score=True)
        else:
            auc_roc[dataset] = model.auc(xval=True)

    return [model, coefs, probabilities, auc_roc, thresholds, FPR, TPR, F1,
            conf_mat, R2]


# Find best-performing model, based on the average performance score
# across all realizations
def train_and_compare_models(config):

    # Initialize an h2o socket
    h2o.init()

NOTE: have user select combo stuff in config file

    # Train/Test various models

    mode = 'train'
    ref_day = config['TRAINING']['REF_DAY']
    model_types = config['TRAINING']['MODEL_TYPES']
    combo_types = config['TRAINING']['COMBO_TYPES']
    feature_list = {
        'with_cut_prior': [
            'f_late','f_zero_vol','cut_prior','nCutPrior',
            'cust_type_code','municipality','meter_size','skew_vol',
            'f_anom3_vol','max_anom_vol',
            'f_anom3_local_vol','max_anom_local_vol',
            'f_anom3_vol_log','max_anom_vol_log',
            'f_anom3_local_vol_log','max_anom_local_vol_log',
        ],
        'omit_cut_prior': [
            'f_late','f_zero_vol',
            'cust_type_code','municipality','meter_size','skew_vol',
            'f_anom3_vol','max_anom_vol',
            'f_anom3_local_vol','max_anom_local_vol',
            'f_anom3_vol_log','max_anom_vol_log',
            'f_anom3_local_vol_log','max_anom_local_vol_log',
        ],
    }
    categoricals = ['cust_type_code','municipality']
    nSamples_list = config['TRAINING']['N_SAMPLE_LIST']
    nSamples_list = list(map(int, nSamples_list))
    nnSampless = len(nSamples_list)
    feature_dir = config['PATHS']['FEATURE_TABLE_DIR_TRAIN']
    model_save_dir = config['PATHS']['MODEL_SAVE_DIR']
    model_perf_dir = config['PATHS']['MODEL_PERF_DIR']
    perf_file = model_perf_dir + '/stats.cross_val.{:s}.{:s}.txt' \
        .format(mode, ref_day)
    features_train = {}
    labels_train = {}
    features_test = {}
    labels_test = {}
    nCTypes = len(combo_types)
    nMTypes = len(model_types)
    auroc_array = np.zeros([nCTypes,nMTypes,nSamples,nRealizations])
    first = 1
    n = 0
    for nSamples in nSamples_list:
        features_train[nSamples] = {}
        labels_train[nSamples] = {}
        features_test[nSamples] = {}
        labels_test[nSamples] = {}
        r = 0
        for r in range(nRealizations):
        
            # Read feature table
            infile = feature_dir + \
                '/feature_table.{:s}.{:s}.N{:02d}.r{:d}.csv' \
                .format(mode, ref_day, nSamples, r)
            print('....reading', infile)
            feature_table = pd.read_csv(infile)

           # Prepare train and test datasets
            oversample = True
            feature_list_all = feature_list['with_cut_prior']
            [features_train[nSamples][r],
             labels_train[nSamples][r],
             features_test[nSamples][r],
             labels_test[nSamples][r]] = prep_train_test_data(feature_table,
                                                              feature_list_all,
                                                              'cutoff',
                                                              oversample)
            features_train[nSamples][r]['cutoff'] = \
                labels_train[nSamples][r]['cutoff']
            features_test[nSamples][r]['cutoff'] = \
                labels_test[nSamples][r]['cutoff']

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
                     thresholds] = \
                         train_and_test_model(model_type,
                                              features_train[nSamples][r],
                                              features_to_use,
                                              'cutoff')
                    print('....nSamples', nSamples, 'realization', r,
                          combo_types[c], model_type, 'AUC_ROC', auc_roc)
            

                    # Write performance metrics to a file
                    if first:
                        code = 'w'
                    else:
                        code = 'a'
                    f = open(perf_file, code)

                    if first:
                        tmpstr = 'model_type,mode,ref_day,nSamples,r,' + \
                                 'combo_type,AUROC\n'
                        f.write(tmpstr)
                    tmpstr = '{:s},{:s},{:s},{:02d},{:d},{:s},{:.6f}\n' \
                        .format(model_type, mode, ref_day, nSamples_list[n], r,
                                combo_types[c], auc_roc)
                    f.write(tmpstr)
                    f.close()

                    if model_type == 'random_forest':
                        # Write feature importances to a file
                        importances = model.feature_importances_
                        imp_file = model_perf_dir + '/importances.' + \
                            '{:s}.{:s}.{:s}.N{:02d}.r{:d}.{:s}.txt' \
                            .format(model_type, mode, ref_day, nSamples, r,
                                    combo_types[c])
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

        
    # Two sets of scores - one with prior cutoffs included as features,
    # one without
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
        mmax_ar[c] = int(idx / nnSampless) 
        nmax_ar[c] = idx - mmax_ar[c] * nnSampless
        print(combo_types[c],model_types[mmax_ar[c]],nSamples_list[nmax_ar[c]])

#    # Print out metrics of all models just for reference
#    for c in range(nCTypes):
#        for m in range(nMTypes):
#            for n in range(nnSampless):
#                print(combo_types[c],model_types[m],nSamples_list[n],
#                      auroc_mean[c,m,n])

    # Show the best models
    for c in range(nCTypes):
        print(combo_types[c], 'best model:', 'alg:',
              model_types[mmax_ar[c]], 'N:', nSamples_list[nmax_ar[c]],
              'ROC AUC over train:', max_auroc_ar[c])

    # For the model with best perf averaged over 3 realizations,
    # find the realization for which perf was best
    for c in range(nCTypes):
        rmax_ar[c] = np.argmax(auroc_array[c,mmax_ar[c],nmax_ar[c],:])


    best_model_info = {}
    best_model_info['ref_day'] = np.empty([nCTypes], dtype='|U256')
    best_model_info['feature_combo_type'] = np.empty([nCTypes], dtype='|U256')
    best_model_info['model_type'] = np.empty([nCTypes], dtype='|U256')
    best_model_info['nSamples'] = np.empty([nCTypes], dtype=np.int)
    best_model_info['realization'] = np.empty([nCTypes], dtype=np.int)
    best_model_info['oversample'] = np.empty([nCTypes], dtype=np.bool)

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
        nSamples = nSamples_list[nmax_ar[c]]
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
        best_model_info['nSamples'][c] = nSamples
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
                                            features_train[nSamples][r],
                                            features_to_use,
                                            'cutoff')
#        print('nSamples', nSamples, 'realization', r, model_type,
#              combo_type, 'AUX_ROC', auc_roc)

        # Save model
        model_file = model_save_dir + '/model.' + \
            '{:s}.{:s}.{:s}.N{:02d}.r{:d}.{:s}.sav' \
            .format(model_type, mode, ref_day, nSamples, r, combo_type)
        pickle.dump(model, open(model_file, 'wb'))
        model_file_best = model_save_dir + \
            '/model.{:s}.{:s}.{:s}.best.sav'.format(mode, ref_day, combo_type)
        copyfile(model_file, model_file_best)
    

        # Now validate best model on test set
        [predictions,
         probabilities,
         conf_mat,
         false_positive_rate,
         true_positive_rate,
         thresholds,
         auc_roc] = apply_fitted_model(model,
                                       features_test[nSamples][r],
                                       features_to_use,
                                       score=True,
                                       labels=labels_test[nSamples][r])
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
            imp_file = model_perf_dir + '/importances.' + \
                '{:s}.{:s}.{:s}.N{:02d}.r{:d}.{:s}.txt' \
                .format(model_type, mode, ref_day, nSamples, r, combo_type)
            f = open(perf_file, 'w')
            tmpstr = ','.join(str(importances)) + '\n'
            f.write(tmpstr)
            f.close()
        
        # Copy feature tables with best nSamples to 'best'
        feature_dir = config['PATHS']['FEATURE_TABLE_DIR_TRAIN']
        infile = feature_dir + '/feature_table.' + \
            '{:s}.{:s}.N{:02d}.r{:d}.csv'.format(mode, ref_day, nSamples, r)
        outfile = feature_dir + '/feature_table.' + \
            '{:s}.{:s}.{:s}.best.csv'.format(mode, ref_day, combo_type)
        copyfile(infile, outfile)

    # Save the relevant details of the best model
    best_model_info_file = config['PATHS']['BEST_MODEL_INFO_FILE']
    df_best_model_info = pd.DataFrame(data=best_model_info)
    df_best_model_info.to_csv(best_model_info_file)

    return 0
