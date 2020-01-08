# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import datetime as dt
from shutil import copyfile
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
    df['nocut'] = feature_table.loc[feature_table[label] == label_val_nocut] \
        .copy()
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
        coefs = model._model_json['output']['coefficients_table'] \
            .as_data_frame()
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


# Find best-performing model, based on the cross-validation AUC score
# Also selects the optimum window length
def train_and_compare_models(config):

    # Initialize an h2o socket
    h2o.init()

    # Train/Test various models

    # Directories
    feature_dir = config['PATHS']['FEATURE_TABLE_DIR_TRAIN']
    model_save_dir = config['PATHS']['MODEL_SAVE_DIR']
    model_perf_dir = config['PATHS']['MODEL_PERF_DIR']

    # Option settings
    mode = 'train'
    ref_day = config['TRAINING']['REF_DAY']
    model_types = config['TRAINING']['MODEL_TYPES']
    nMTypes = len(model_types)
    nSamples_list = config['TRAINING']['N_SAMPLE_LIST']
    nSamples_list = list(map(int, nSamples_list))
    nNSamples = len(nSamples_list)

    # To Do: this should ideally be moved to some separate config-type function
    # Types of features
    feature_list_basic = ['f_late', 'f_zero_vol']
    feature_list_anom = ['f_anom3_vol', 'f_manom3_vol']
    feature_list_metadata = ['cust_type_code', 'municipality', 'meter_size']
    feature_list_cut_prior = ['cut_prior']
    categoricals = feature_list_metadata.copy()
    categoricals.append(feature_list_cut_prior[0])
    # Valid feature combo options
    anom_types_options = ['anom', 'manom']
    metadata_options = ['with_meta', 'no_meta']
    cut_prior_options = ['with_cut_prior', 'no_cut_prior']

NOTE: have user select combo stuff in config file

    # User-selected feature combo options
    option_cut_prior = config['TRAINING']['FEATURE_OPTIONS']['CUT_PRIOR']
    option_metadata = config['TRAINING']['FEATURE_OPTIONS']['METADATA']
    option_anom = config['TRAINING']['FEATURE_OPTIONS']['ANOM']

    # Build feature list
    feature_list = feature_list_basic.copy()
    if option_anom == 'manom':
        feature_list.append(feature_list_anom[1])
    else:
        feature_list.append(feature_list_anom[0])
    if option_metadata = 'with_meta':
        feature_list.extend(feature_list_metadata)
    if option_cut_prior = 'with_cut_prior':
        feature_list.append(feature_list_cut_prior[0])
    feature_and_label_list = feature_list.copy()
    feature_and_label_list.append(label)
    feature_label_wt_list = feature_and_label_list.copy()
    feature_label_wt_list.append(wtcol)

NOTE: should regularization be automatically set based on presence of categoricals, or should it be user-selected option? Why would only categoricals need regularization?

    # Label
    label = 'cutoff_strict'
    label_val_nocut = 0
    label_val_cut = 1

    # Rebalancing info
    rebalance = 'weights'

    # Weights column
    wtcol = 'weights'

    # Arrays of performance metrics
    auroc_array = np.zeros([nNSamples,nMTypes])

    # Loop over nSamples and model type to find best combination
    n = 0
    for nSamples in nSamples_list:

        # Read feature table
        infile = feature_dir + '/feature_table.{:s}.{:s}.N{:02d}.csv' \
            .format(mode, ref_day, nSamples)
        print('....reading', infile)
        feature_table = pd.read_csv(infile)

        # Prepare train and test datasets
        [features_train, labels_train,
         features_test, labels_test] = prep_train_test_data(feature_table,
                                                            feature_list,
                                                            categoricals,
                                                            label,
                                                            label_val_nocut,
                                                            label_val_cut,
                                                            rebalance)
        feature_table_train = pd.concat([features_train, labels_train], axis=1)
        feature_table_test = pd.concat([features_test, labels_test], axis=1)

        m = 0
        for model_type in model_types:
            if model_type == 'random_forest':
                grid_search = True
            else:
                grid_search = False

            instance_str = '{:s}.{:s}.N{:02d}.{:s}'.format(mode,
                                                           ref_date_str,
                                                           nSamples,
                                                           model_type)

            # Train and test model
            [model, coefs,
             probabilities,
             auc_roc,
             thresholds,
             FPR, TPR, F1, 
             conf_mat, R2] = \
                 train_and_test_model(model_type,
                                      feature_table_train[feature_label_wt_list],
                                      feature_table_test[feature_and_label_list],
                                      feature_list, label, label_val_cut,
                                      categoricals, wtcol,
                                      regularization, grid_search)
            print('....nSamples', nSamples, model_type, 'AUC_ROC[xval]',
                  auc_roc['xval'])

            # Save model
            model_path = h2o.save_model(model=model, path=model_save_dir,
                                        force=True)
            model_path_file = model_save_dir + '/model.' + instance_str + \
                '.path.txt'
            with open(model_path_file, 'w') as f:
                f.write(model_path)

            # Write performance metrics to a file
            perf_file = model_perf_dir + '/stats.' + instance_str + '.csv'
            f = open(perf_file, 'w')
            tmpstr = 'model_type,nSamples,' + \
                     'AUROC_train,FPR_train,TPR_train,F1_train,R2_train,' + \
                     'AUROC_xval,' + \
                     'AUROC_test,FPR_test,TPR_test,F1_test,R2_test\n'
            f.write(tmpstr)
            tmpstr1 = '{:s},{:d},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},' \
                .format(model_type, nSamples, auc_roc['train'],
                        FPR['train'], TPR['train'], F1['train'], R2['train'])
            tmpstr2 = '{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f}\n' \
                .format(auc_roc['xval'], auc_roc['test'],
                        FPR['test'], TPR['test'], F1['test'], R2['test'])
            f.write(tmpstr1 + tmpstr2)
            f.close()

            # Store auroc in memory for finding optimum model
            auroc_array[n, m] = auc_roc['xval']

            # Write feature importances or coefficients to a file
            imp_file = model_perf_dir + '/coefs.' + instance_str + '.csv'
            coefs.to_csv(imp_file)

            m += 1
        n += 1

    # Find indices of maximum auc_roc
    imax_flat = np.argmax(auroc_array)
    mmax = imax_flat % 2
    nmax = int(imax_flat / 2)

#    # Print out metrics of all models just for reference
#    for n in range(nNSamples):
#        for m in range(nMTypes):
#            print(nSamples_list[n],model_types[m],auroc_array[n,m])

    print('best model:', model_types[mmax], 'nSamples:', nSamples_list[nmax],
              'ROC AUC cross validation over train:', auroc_array[n, m])

NOTE: save best model info
NOTE: copy best saved model to best model file
NOTE: copy best model saved perf stats to best model perf stats file
    # Save the relevant details of the best model
    best_model_info = {}
    nSamples = nSamples_list[nmax]
    model_type = model_types[mmax]
    best_model_info['ref_day'] = ref_day
    best_model_info['rebalance'] = rebalance
    best_model_info['nSamples'] = nSamples_list[nmax]
    best_model_info['model_type'] = model_types[mmax]
    best_model_info['option_cut_prior'] = option_cut_prior
    best_model_info['option_metadata'] = option_metadata
    best_model_info['option_anom'] = option_anom
    best_model_info['feature_list'] = feature_list

        model_file_best = model_save_dir + \
            '/model.{:s}.{:s}.{:s}.best.sav'.format(mode, ref_day, combo_type)
        copyfile(model_file, model_file_best)
    

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
