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
    df_cut = df['cut'].copy()
    df.pop('cut', None)

    # We additionally want to ensure that both the first cutoffs of all
    # occupants and subsequent cutoffs (for those who have multiple cutoffs)
    # are also split proportionally into train/test subsets; this helps ensure
    # that train and test subsets contain similar proportions of first and
    # subsequent cutoffs (in case these behave differently).
    # First, select the first cutoff (chronologically) from
    # each occupant; this will have the max segment value for that occupant
    # due to our clipping the windows in reverse chronological order.
    df_tmp = df_cut.groupby('occupant_id').agg({'segment':'max'}) \
        .reset_index().rename(columns={'segment':'segmax'})
    df_cut = df_cut.merge(df_tmp, on='occupant_id')
    df['cut0'] = df_cut.loc[df_cut['segment'] == df_cut['segmax']].copy()
    # All other cutoffs are subsequent to the first ones
    df['cut1'] = df_cut.loc[df_cut['segment'] != df_cut['segmax']].copy()

    # Free up memory
    df_tmp = []
    df_cut = []
 
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

    # Free up memory
    features_dict = {}
    labels_dict = {}

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

    # Free up memory
    features_train = {}
    labels_train = {}
    features_test = {}
    labels_test = {}

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
        auc = perf.auc()
        logloss = perf.logloss()
        R2 = perf.r2()
    else:
        auc = None
        logloss = None
        R2 = None

    return [probabilities, auc, logloss, R2]

 
def train_and_test_model(model_type, feature_table_train, feature_table_test,
                         feature_list, label, label_val_cut, categoricals=None,
                         wtcol=None, nfolds=5, fold_asgmnt='Stratified',
                         regularization=False, ntrees=100,
                         max_depth=20, max_depth_list=None):

    # Set up inputs for model
    df_train = set_up_H2OFrame(feature_table_train, feature_list,
                               categoricals, label)
    df_test = set_up_H2OFrame(feature_table_test, feature_list,
                              categoricals, label)

    # Instantiate model
    if model_type == 'random_forest':
        if max_depth_list == None:
            model = H2ORandomForestEstimator(ntrees=ntrees, max_depth=max_depth,
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
#                                           'max_depth':max_depth_list
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
    if model_type == 'random_forest' and max_depth_list != None:
#        model_grid.train(x=feature_list, y=label, training_frame=df_train,
#                         weights_column=wtcol, ntrees=ntrees,
#                         validation_frame=df_test, nfolds=nfolds,
#                         fold_assignment=fold_asgmnt, seed=1)
#        # Get the best-performing model
#        model_gridperf = model_grid.get_grid(sort_by='auc', decreasing=True)
#        model = model_gridperf.models[0]
        nMaxDepth = len(max_depth_list)
        models = []
        aucs = np.zeros([nMaxDepth])
        for i in range(nMaxDepth):
            model = H2ORandomForestEstimator(ntrees=ntrees,
                                             max_depth=int(max_depth_list[i]),
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
    auc = {}
    logloss = {}
    R2 = {}
    for dataset in ['train', 'xval', 'test']:
        if dataset in ['train', 'test']:
            if dataset == 'train':
                feature_table = feature_table_train
            elif dataset == 'test':
                feature_table = feature_table_test

            [
                probabilities[dataset],
                auc[dataset],
                logloss[dataset],
                R2[dataset]
            ] = apply_model(model, feature_table, feature_list,
                            label, label_val_cut, categoricals,
                            score=True)
        else:
            auc[dataset] = model.auc(xval=True)

    return [model, coefs, probabilities, auc, logloss, R2]


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
    prediction_dir = config['PATHS']['PREDICTIONS_DIR_TRAIN']

    # Option settings
    mode = 'train'
    ref_date = config['TRAINING']['REF_DATE']
    option_cut_prior = config['TRAINING']['FEATURES_CUT_PRIOR']
    option_metadata = config['TRAINING']['FEATURES_METADATA']
    option_anom = config['TRAINING']['FEATURES_ANOM']
    model_types = config['TRAINING']['MODEL_TYPES']
    nMTypes = len(model_types)
    nSamples_list = config['TRAINING']['N_SAMPLE_LIST']
    nSamples_list = list(map(int, nSamples_list))
    nNSamples = len(nSamples_list)

    # Options string for filenames
    opt_str = '{:s}.{:s}.{:s}.{:s}.{:s}'.format(mode, ref_date,
                                                option_cut_prior,
                                                option_metadata, option_anom)

    # To Do: this should ideally be moved to some separate config-type function
    # Types of features
    feature_list_basic = ['f_late', 'f_zero_vol']
    feature_list_anom = ['f_anom3_vol', 'f_manom3_vol']
    feature_list_metadata = ['cust_type_code', 'municipality', 'meter_size']
    feature_list_cut_prior = ['cut_prior']
    categoricals = feature_list_metadata.copy()
    categoricals.append(feature_list_cut_prior[0])
    # Valid feature combo options
    anom_types_options = ['anom', 'manom', 'none']
    metadata_options = ['with_meta', 'no_meta']
    cut_prior_options = ['with_cut_prior', 'no_cut_prior']

    # Label
    label = 'cutoff_strict'
    label_val_nocut = 0
    label_val_cut = 1

    # Weights column
    wtcol = 'weights'

    # To Do: put this logic and data into a separate function and/or file
    # Build feature list
    feature_list = feature_list_basic.copy()
    if option_anom == 'anom':
        feature_list.append(feature_list_anom[0])
    elif option_anom == 'manom':
        feature_list.append(feature_list_anom[1])
    if option_metadata == 'with_meta':
        feature_list.extend(feature_list_metadata)
    if option_cut_prior == 'with_cut_prior':
        feature_list.append(feature_list_cut_prior[0])
    feature_and_label_list = feature_list.copy()
    feature_and_label_list.append(label)
    feature_label_wt_list = feature_and_label_list.copy()
    feature_label_wt_list.append(wtcol)

    # Rebalancing info
    rebalance = 'weights'

    # Regularization
    regularization = True

    # Arrays of performance metrics
    auc_array = np.zeros([nNSamples,nMTypes])

    # Loop over nSamples and model type to find best combination
    n = 0
    for nSamples in nSamples_list:

        # Read feature table
        infile = feature_dir + '/feature_table.{:s}.{:s}.N{:02d}.csv' \
            .format(mode, ref_date, nSamples)
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
                max_depth_list = config['TRAINING']['MAX_DEPTH_LIST']
            else:
                max_depth_list = None

            instance_str = '{:s}.N{:02d}.{:s}'.format(opt_str, nSamples,
                                                      model_type)

            # Train and test model
            [model, coefs,
             probabilities,
             auc, logloss, R2] = \
                 train_and_test_model(model_type,
                                      feature_table_train[feature_label_wt_list],
                                      feature_table_test[feature_and_label_list],
                                      feature_list, label, label_val_cut,
                                      categoricals=categoricals,
                                      wtcol=wtcol,
                                      regularization=regularization,
                                      max_depth_list=max_depth_list)

            print('....nSamples', nSamples, model_type, 'AUC_ROC[xval]',
                  auc['xval'])

            # Save model
            model_path = h2o.save_model(model=model, path=model_save_dir,
                                        force=True)
            model_path_file = model_save_dir + '/model.' + instance_str + \
                '.path.txt'
            with open(model_path_file, 'w') as f:
                f.write(model_path)

            # Save model predictions
            df_prob = pd.DataFrame(data={'p_cutoff':probabilities})
            prob_file = prediction_dir + '/probabilities.' + instance_str + '.csv'
            df_prob.to_csv(prob_file)

            # Write performance metrics to a file
            perf_file = model_perf_dir + '/stats.' + instance_str + '.csv'
            f = open(perf_file, 'w')
            tmpstr = 'AUC_train,LogLoss_train,R2_train,AUROC_xval,' + \
                     'AUROC_test,LogLoss_test,R2_test\n'
            f.write(tmpstr)
            print(auc['train'], logloss['train'], R2['train'])
            tmpstr1 = '{:.6f},{:.6f},{:.6f},' \
                .format(auc['train'], logloss['train'], R2['train'])
            tmpstr2 = '{:.6f},{:.6f},{:.6f},{:.6f}\n' \
                .format(auc['xval'], auc['test'], logloss['test'], R2['test'])
            f.write(tmpstr1 + tmpstr2)
            f.close()

            # Store auc in memory for finding optimum model
            auc_array[n, m] = auc['xval']

            # Write feature importances or coefficients to a file
            imp_file = model_perf_dir + '/coefs.' + instance_str + '.csv'
            coefs.to_csv(imp_file)

            m += 1
        n += 1

    # Find indices of maximum auc
    imax_flat = np.argmax(auc_array)
    mmax = imax_flat % 2
    nmax = int(imax_flat / 2)

    print('best model:', model_types[mmax], 'nSamples:', nSamples_list[nmax],
              'ROC AUC cross validation over train:', auc_array[nmax, mmax])

    # Save the relevant details of the best model
    best_model_info = {}
    nSamples = nSamples_list[nmax]
    model_type = model_types[mmax]
    instance_str = '{:s}.N{:02d}.{:s}'.format(opt_str, nSamples, model_type)
    best_model_info['ref_date'] = ref_date
    best_model_info['option_cut_prior'] = option_cut_prior
    best_model_info['option_metadata'] = option_metadata
    best_model_info['option_anom'] = option_anom
    best_model_info['feature_list'] = feature_list
    best_model_info['nSamples'] = nSamples
    best_model_info['model_type'] = model_type
    best_model_info['rebalance'] = rebalance
#    best_model_info_file = config['PATHS']['BEST_MODEL_INFO_FILE']
    best_model_info_file = model_save_dir + '/best_model_info.' + opt_str + \
        '.csv'
    df_best_model_info = pd.DataFrame(data=best_model_info)
    df_best_model_info.to_csv(best_model_info_file)

    # Copy best saved model to file named 'best'
    model_path_file = model_save_dir + '/model.' + instance_str + '.path.txt'
    model_path_file_best = model_save_dir + '/model.' + opt_str + \
        '.best.path.txt'
    copyfile(model_path_file, model_path_file_best)
 
    # Copy predictions of best model to file named 'best'
    prob_file = prediction_dir + '/probabilities.' + instance_str + '.csv'
    prob_file_best = prediction_dir + '/probabilities.' + opt_str + '.best.csv'
    copyfile(prob_file, prob_file_best)

    # Copy perf stats of best model to file named 'best'
    perf_file = model_perf_dir + '/stats.' + instance_str + '.csv'
    perf_file_best = model_perf_dir + '/stats.' + opt_str + '.best.csv'
    copyfile(perf_file, perf_file_best)

    # Copy feature tables with best nSamples to 'best'
    feature_dir = config['PATHS']['FEATURE_TABLE_DIR_TRAIN']
    infile = feature_dir + '/feature_table.{:s}.{:s}.N{:02d}.csv' \
        .format(mode, ref_date, nSamples)
    outfile = feature_dir + '/feature_table.' + opt_str + '.best.csv'
    copyfile(infile, outfile)

    return 0
