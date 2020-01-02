#!/usr/bin/env python3
"""Machine Learning module for ADNI capstone project.

This module contains functions for use with the ADNI dataset.
"""

if 'pd' not in globals():
    import pandas as pd

if 'np' not in globals():
    import numpy as np
    
if 'plt' not in globals():
    import matplotlib.pyplot as plt
    
if 'sns' not in globals():
    import seaborn as sns
    
if 'scipy.stats' not in globals():
    import scipy.stats
    
if 'StandardScaler' not in globals():
    from sklearn.preprocessing import StandardScaler, MinMaxScaler

if 'KNeighborsClassifier' not in globals():
    from sklearn.neighbors import KNeighborsClassifier
    
if 'SVC' not in globals():    
    from sklearn.svm import SVC
    
if 'train_test_split' not in globals():
    from sklearn.model_selection import train_test_split, GridSearchCV
    
if 'MultinomialNB' not in globals():
    from sklearn.naive_bayes import MultinomialNB

if 'confusion_matrix' not in globals():
    from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report

if 'RandomForestClassifier' not in globals():
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

if 'linear_model' not in globals():
    from sklearn import linear_model
    
if 'PCA' not in globals():
    from sklearn.decomposition import PCA


sns.set()


def get_delta_scaled(final_exam, neg_one=False):
    """Take the final_exam dataframe and return datasets.
    
    This function returns five numpy arrays: feature_names, X_delta_male, 
    X_delta_female, y_delta_male, and y_delta_female. The two X arrays hold
    the feature data. The two y arrays hold the diagnosis group labels.
    The feature_names array hold a list of the features. The neg_one
    parameter allows you to specify -1 for the negative class (for SVM)."""
    
    # map the diagnosis group and assign to dx_group
    nc_idx = final_exam[final_exam.DX == final_exam.DX_bl2].index
    cn_mci_idx = final_exam[(final_exam.DX == 'MCI') & (final_exam.DX_bl2 == 'CN')].index
    mci_ad_idx = final_exam[(final_exam.DX == 'AD') & (final_exam.DX_bl2 == 'MCI')].index
    cn_ad_idx = final_exam[(final_exam.DX == 'AD') & (final_exam.DX_bl2 == 'CN')].index

    if neg_one:
        labels = pd.concat([pd.DataFrame({'dx_group': -1}, index=nc_idx),
                            pd.DataFrame({'dx_group': -1}, index=cn_mci_idx),
                            pd.DataFrame({'dx_group': 1}, index=mci_ad_idx),
                            pd.DataFrame({'dx_group': 1}, index=cn_ad_idx)
                           ]).sort_index()
    else:
        labels = pd.concat([pd.DataFrame({'dx_group': 0}, index=nc_idx),
                            pd.DataFrame({'dx_group': 0}, index=cn_mci_idx),
                            pd.DataFrame({'dx_group': 1}, index=mci_ad_idx),
                            pd.DataFrame({'dx_group': 1}, index=cn_ad_idx)
                           ]).sort_index()
    
    # add to the dataframe and ensure every row has a label
    deltas_df = final_exam.loc[labels.index]
    deltas_df.loc[:,'dx_group'] = labels.dx_group 

    # convert gender to numeric column
    deltas_df = pd.get_dummies(deltas_df, drop_first=True, columns=['PTGENDER'])
    
    # extract the features for change in diagnosis
    X_delta = deltas_df.reindex(columns=['CDRSB_delta', 'ADAS11_delta', 'ADAS13_delta', 'MMSE_delta',
                                         'RAVLT_delta', 'Hippocampus_delta', 'Ventricles_delta',
                                         'WholeBrain_delta', 'Entorhinal_delta', 'MidTemp_delta',
                                         'PTGENDER_Male', 'AGE'])
      
    # store the feature names
    feature_names = np.array(['CDRSB_delta', 'ADAS11_delta', 'ADAS13_delta', 'MMSE_delta', 'RAVLT_delta',
                              'Hippocampus_delta', 'Ventricles_delta', 'WholeBrain_delta',
                              'Entorhinal_delta', 'MidTemp_delta', 'PTGENDER_Male', 'AGE'])
    
    # standardize the data
    scaler = StandardScaler()
    Xd = scaler.fit_transform(X_delta)
    
    # extract the labels
    yd = np.array(deltas_df.dx_group)
    
    # return the data
    return feature_names, Xd, yd

def plot_best_k(X_train, X_test, y_train, y_test, kmax=9):
    """This function will create a plot to help choose the best k for k-NN.
    
    Supply the training and test data to compare accuracy at different k values.
    Specifying a max k value is optional."""
    
    # Setup arrays to store train and test accuracies
    # view the plot to help pick the best k to use
    neighbors = np.arange(1, kmax)
    train_accuracy = np.empty(len(neighbors))
    test_accuracy = np.empty(len(neighbors))

    # Loop over different values of k
    for i, k in enumerate(neighbors):
        # Setup a k-NN Classifier with k neighbors: knn
        knn = KNeighborsClassifier(n_neighbors=k)

        # Fit the classifier to the training data
        knn.fit(X_train, y_train)
    
        #Compute accuracy on the training set
        train_accuracy[i] = knn.score(X_train, y_train)

        #Compute accuracy on the testing set
        test_accuracy[i] = knn.score(X_test, y_test)
    
    if kmax < 11:
        s = 2
    elif kmax < 21:
        s = 4
    elif kmax < 41:
        s = 5
    elif kmax < 101:
        s = 10
    else:
        s = 20
        
    # Generate plot
    _ = plt.title('k-NN: Varying Number of Neighbors')
    _ = plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
    _ = plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
    _ = plt.legend()
    _ = plt.xlabel('Number of Neighbors')
    _ = plt.ylabel('Accuracy')
    _ = plt.xticks(np.arange(0,kmax,s))
    plt.show()
    
def plot_f1_scores(k, s, r, b, l, n):
    """This function accepts six dictionaries containing classification reports.
    
    This function is designed to work specifically with the six dictionaries created 
    in the 5-Machine_Learning notebook, as the second dictionary is SVM, which
    uses classes of -1 and 1, whereas the other classes are 0 and 1."""
    
    # extract the data and store in a dataframe
    df = pd.DataFrame({'score': [k['0']['f1-score'], k['1']['f1-score'], s['-1']['f1-score'], s['1']['f1-score'],
                                 r['0']['f1-score'], r['1']['f1-score'], b['0']['f1-score'], b['1']['f1-score'],
                                 l['0']['f1-score'], l['1']['f1-score'], n['0']['f1-score'], n['1']['f1-score']],
                       'model': ['KNN', 'KNN', 'SVM', 'SVM', 'Random Forest', 'Random Forest',
                                 'AdaBoost', 'AdaBoost', 'Log Reg', 'Log Reg', 'Naive Bayes', 'Naive Bayes'],
                       'group': ['Non AD', 'AD', 'Non AD', 'AD', 'Non AD', 'AD', 'Non AD', 'AD',
                                 'Non AD', 'AD', 'Non AD', 'AD']})
    
    # create the plot
    ax = sns.barplot('model', 'score', hue='group', data=df)
    _ = plt.setp(ax.get_xticklabels(), rotation=25)
    _ = plt.title('F1 Scores for Each Model')
    _ = plt.ylabel('F1 Score')
    _ = plt.xlabel('Model')
    _ = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
    
def get_bl_data(final_exam, neg_one=False):
    """This function extracts the baseline data features for machine learning.
    
    Pass the final_exam dataframe, specify optional neg_one=True for SVM (sets)
    the non-Ad class as -1 vs 0. Returns features (X), labels (y), and 
    feature_names.
    """
    
    # map the diagnosis group and assign to dx_group
    non_ad_idx = final_exam[final_exam.DX != 'AD'].index
    ad_idx = final_exam[final_exam.DX == 'AD'].index
    
    if neg_one:
        labels = pd.concat([pd.DataFrame({'dx_group': -1}, index=non_ad_idx),
                            pd.DataFrame({'dx_group': 1}, index=ad_idx)
                           ]).sort_index()
    else:
        labels = pd.concat([pd.DataFrame({'dx_group': 0}, index=non_ad_idx),
                            pd.DataFrame({'dx_group': 1}, index=ad_idx)
                           ]).sort_index()
    
    # add to the dataframe and ensure every row has a label
    bl_df = final_exam.loc[labels.index]
    bl_df.loc[:,'dx_group'] = labels.dx_group 

    # convert gender to numeric column
    bl_df = pd.get_dummies(bl_df, drop_first=True, columns=['PTGENDER'])
    
    # extract the baseline features
    X_bl = bl_df.reindex(columns=['CDRSB_bl', 'ADAS11_bl', 'ADAS13_bl', 'MMSE_bl', 'RAVLT_immediate_bl', 
                              'Hippocampus_bl', 'Ventricles_bl', 'WholeBrain_bl', 'Entorhinal_bl', 
                              'MidTemp_bl', 'PTGENDER_Male', 'AGE'])
      
    # store the feature names
    feature_names = np.array(['CDRSB_bl', 'ADAS11_bl', 'ADAS13_bl', 'MMSE_bl', 'RAVLT_immediate_bl', 
                              'Hippocampus_bl', 'Ventricles_bl', 'WholeBrain_bl', 'Entorhinal_bl', 
                              'MidTemp_bl', 'PTGENDER_Male', 'AGE'])
    
    # standardize the data
    scaler = StandardScaler()
    Xd = scaler.fit_transform(X_bl)
    
    # extract the labels
    yd = np.array(bl_df.dx_group)
    
    # return the data
    return feature_names, Xd, yd

def run_clinical_models(final_exam, biomarkers):
    """This dataframe runs six machine learning models on only the clinical biomarkes.
    
    A dataframe containing summary information will be returned."""
    
    # map the diagnosis group and assign to dx_group
    nc_idx = final_exam[final_exam.DX == final_exam.DX_bl2].index
    cn_mci_idx = final_exam[(final_exam.DX == 'MCI') & (final_exam.DX_bl2 == 'CN')].index
    mci_ad_idx = final_exam[(final_exam.DX == 'AD') & (final_exam.DX_bl2 == 'MCI')].index
    cn_ad_idx = final_exam[(final_exam.DX == 'AD') & (final_exam.DX_bl2 == 'CN')].index
    
    labels = pd.concat([pd.DataFrame({'dx_group': 0}, index=nc_idx),
                            pd.DataFrame({'dx_group': 0}, index=cn_mci_idx),
                            pd.DataFrame({'dx_group': 1}, index=mci_ad_idx),
                            pd.DataFrame({'dx_group': 1}, index=cn_ad_idx)
                           ]).sort_index()
    
    # add to the dataframe and ensure every row has a label
    labeled_df = final_exam.loc[labels.index]
    labeled_df.loc[:,'dx_group'] = labels.dx_group

    # convert gender to numeric column
    labeled_df = pd.get_dummies(labeled_df, drop_first=True, columns=['PTGENDER'])
    
    if biomarkers == 'deltas':
        # extract the features for change in diagnosis
        X = labeled_df.reindex(columns=['CDRSB_delta', 'ADAS11_delta', 'ADAS13_delta', 'MMSE_delta',
                                         'RAVLT_delta', 'PTGENDER_Male', 'AGE'])
          
        # store the feature names
        feature_names = np.array(['CDRSB_delta', 'ADAS11_delta', 'ADAS13_delta', 'MMSE_delta', 'RAVLT_delta',
                              'PTGENDER_Male', 'AGE'])
    
    elif biomarkers == 'baseline':
        # extract the features for change in diagnosis
        X = labeled_df.reindex(columns=['CDRSB_bl', 'ADAS11_bl', 'ADAS13_bl', 'MMSE_bl',
                                         'RAVLT_immediate_bl', 'PTGENDER_Male', 'AGE'])
          
        # store the feature names
        feature_names = np.array(['CDRSB_bl', 'ADAS11_bl', 'ADAS13_bl', 'MMSE_bl',
                                         'RAVLT_immediate_bl', 'PTGENDER_Male', 'AGE'])
        
    # standardize the data
    scaler = StandardScaler()
    Xd = scaler.fit_transform(X)
    
    # extract the labels
    yd = np.array(labeled_df.dx_group)
        
    # split into training and test data
    Xd_train, Xd_test, yd_train, yd_test = train_test_split(Xd, yd, test_size=0.3, 
                                                    random_state=21, stratify=yd)
    
    # initialize dataframe to hold summary info for the models
    columns = ['model', 'hyper_params', 'train_acc', 'test_acc', 'auc', 'tp', 'fn', 'tn', 'fp',
              'precision', 'recall', 'fpr', 'neg_f1', 'AD_f1']
    df = pd.DataFrame(columns=columns)

    # knn model
    param_grid = {'n_neighbors': np.arange(1, 50)}
    knn = KNeighborsClassifier()
    knn_cv = GridSearchCV(knn, param_grid, cv=5)
    knn_cv.fit(Xd_train, yd_train)
    k = knn_cv.best_params_['n_neighbors']
    hp = 'k: {}'.format(k)
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(Xd_train, yd_train)
    y_pred = knn.predict(Xd_test)
    train_acc = knn.score(Xd_train, yd_train)
    test_acc = knn.score(Xd_test, yd_test)
    y_pred_prob = knn.predict_proba(Xd_test)[:,1]
    auc = roc_auc_score(yd_test, y_pred_prob)
    tn, fp, fn, tp = confusion_matrix(yd_test, y_pred).ravel()
    prec = tp / (tp + fp)
    recall = tp / (tp + fn)
    fpr = fp / (tn + fp)
    rep = classification_report(yd_test, y_pred, output_dict=True)
    knn_df = pd.DataFrame({'model': 'knn', 'hyper_params': hp, 'train_acc': train_acc, 'test_acc': test_acc,
               'auc': auc, 'tp': tp, 'fn': fn, 'tn': tn, 'fp': fp, 'precision': prec, 'recall': recall,
               'fpr': fpr, 'neg_f1': rep['0']['f1-score'], 'AD_f1': rep['1']['f1-score']}, index=[0])
    df = df.append(knn_df, ignore_index=True, sort=False)
    
    # SVM model
    # map the svm labels
    yd_train_svm = np.where(yd_train == 0, yd_train - 1, yd_train)
    yd_test_svm = np.where(yd_test == 0, yd_test - 1, yd_test)
    num_features = Xd_train.shape[1]
    param_grid = {'C': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 1.75], 
              'gamma': [(1/(num_features*Xd_train.var())), (1/num_features)]}
    svm = SVC(class_weight='balanced', probability=True)
    svm_cv = GridSearchCV(svm, param_grid, cv=5)
    svm_cv.fit(Xd_train, yd_train_svm)
    C = svm_cv.best_params_['C']
    gamma = svm_cv.best_params_['gamma']
    hp = 'C: {}'.format(C) + ', gamma: {:.4f}'.format(gamma)
    svm = SVC(C=C, gamma=gamma, class_weight='balanced',
         probability=True)
    svm.fit(Xd_train, yd_train_svm)
    y_pred = svm.predict(Xd_test)
    train_acc = svm.score(Xd_train, yd_train_svm)
    test_acc = svm.score(Xd_test, yd_test_svm)
    y_pred_prob = svm.predict_proba(Xd_test)[:,1]
    auc = roc_auc_score(yd_test_svm, y_pred_prob)
    tn, fp, fn, tp = confusion_matrix(yd_test_svm, y_pred).ravel()
    prec = tp / (tp + fp)
    recall = tp / (tp + fn)
    fpr = fp / (tn + fp)
    rep = classification_report(yd_test_svm, y_pred, output_dict=True)
    roc_auc_score(yd_test_svm, y_pred_prob)
    svm_df = pd.DataFrame({'model': 'svm', 'hyper_params': hp, 'train_acc': train_acc, 
                           'test_acc': test_acc, 'auc': auc, 'tp': tp, 'fn': fn, 'tn': tn, 'fp': fp, 'precision': prec,
                           'recall': recall, 'fpr': fpr, 'neg_f1': rep['-1']['f1-score'], 'AD_f1': rep['1']['f1-score']},
                         index=[1])
    df = df.append(svm_df, ignore_index=True, sort=False)
    
    # Random Forests Model
    trees = [101, 111, 121, 131, 141, 151, 161, 171, 181, 191, 201, 211, 221]
    max_f = [1, num_features, 'log2', 'sqrt']
    param_grid = {'n_estimators': trees, 'max_features': max_f}
    r_forest = RandomForestClassifier(class_weight='balanced', random_state=42)
    r_forest_cv = GridSearchCV(r_forest, param_grid, cv=5)
    r_forest_cv.fit(Xd_train, yd_train)
    n_est = r_forest_cv.best_params_['n_estimators']
    n_feat = r_forest_cv.best_params_['max_features']
    hp = 'trees: {}'.format(n_est) + ', max_feats: {}'.format(n_feat)
    rfc = RandomForestClassifier(n_estimators=n_est, max_features=n_feat, 
                             class_weight='balanced', random_state=42)
    rfc.fit(Xd_train, yd_train)
    y_pred = rfc.predict(Xd_test)
    train_acc = rfc.score(Xd_train, yd_train)
    test_acc = rfc.score(Xd_test, yd_test)
    y_pred_prob = rfc.predict_proba(Xd_test)[:,1]
    auc = roc_auc_score(yd_test, y_pred_prob)
    tn, fp, fn, tp = confusion_matrix(yd_test, y_pred).ravel()
    prec = tp / (tp + fp)
    recall = tp / (tp + fn)
    fpr = fp / (tn + fp)
    rep = classification_report(yd_test, y_pred, output_dict=True)
    rfc_df = pd.DataFrame({'model': 'RF', 'hyper_params': hp, 'train_acc': train_acc, 
                           'test_acc': test_acc, 'auc': auc, 'tp': tp, 'fn': fn, 'tn': tn, 'fp': fp, 
                           'precision': prec, 'recall': recall, 'fpr': fpr, 'neg_f1': rep['0']['f1-score'], 
                           'AD_f1': rep['1']['f1-score']}, index=[2])
    df = df.append(rfc_df, ignore_index=True, sort=False)
    
    # AdaBoost Classifier
    est = [31, 41, 51, 61, 71, 81, 91, 101]
    param_grid = {'n_estimators': est}
    boost = AdaBoostClassifier(random_state=42)
    boost_cv = GridSearchCV(boost, param_grid, cv=5)
    boost_cv.fit(Xd_train, yd_train)
    n_est = boost_cv.best_params_['n_estimators']
    hp = 'num_estimators: {}'.format(n_est)
    model = AdaBoostClassifier(n_estimators=n_est, random_state=0)
    model.fit(Xd_train, yd_train)
    y_pred = model.predict(Xd_test)
    train_acc = model.score(Xd_train, yd_train)
    test_acc = model.score(Xd_test, yd_test)
    y_pred_prob = model.predict_proba(Xd_test)[:,1]
    auc = roc_auc_score(yd_test, y_pred_prob)
    tn, fp, fn, tp = confusion_matrix(yd_test, y_pred).ravel()
    prec = tp / (tp + fp)
    recall = tp / (tp + fn)
    fpr = fp / (tn + fp)
    rep = classification_report(yd_test, y_pred, output_dict=True)
    boost_df = pd.DataFrame({'model': 'AdaBoost', 'hyper_params': hp, 'train_acc': train_acc, 
                           'test_acc': test_acc, 'auc': auc, 'tp': tp, 'fn': fn, 'tn': tn, 'fp': fp, 
                           'precision': prec, 'recall': recall, 'fpr': fpr, 'neg_f1': rep['0']['f1-score'], 
                           'AD_f1': rep['1']['f1-score']}, index=[3])
    df = df.append(boost_df, ignore_index=True, sort=False)
    
    # logistic regression
    logreg = linear_model.LogisticRegression(solver='lbfgs', class_weight='balanced', random_state=42)
    logreg.fit(Xd_train, yd_train)
    y_pred = logreg.predict(Xd_test)
    train_acc = logreg.score(Xd_train, yd_train)
    test_acc = logreg.score(Xd_test, yd_test)
    y_pred_prob = logreg.predict_proba(Xd_test)[:,1]
    auc = roc_auc_score(yd_test, y_pred_prob)
    tn, fp, fn, tp = confusion_matrix(yd_test, y_pred).ravel()
    prec = tp / (tp + fp)
    recall = tp / (tp + fn)
    fpr = fp / (tn + fp)
    rep = classification_report(yd_test, y_pred, output_dict=True)
    logreg_df = pd.DataFrame({'model': 'logreg', 'hyper_params': None, 'train_acc': train_acc, 
                           'test_acc': test_acc, 'auc': auc, 'tp': tp, 'fn': fn, 'tn': tn, 'fp': fp, 
                           'precision': prec, 'recall': recall, 'fpr': fpr, 'neg_f1': rep['0']['f1-score'], 
                           'AD_f1': rep['1']['f1-score']}, index=[4])
    df = df.append(logreg_df, ignore_index=True, sort=False)
    
    # Naive Bayes
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(Xd_train)
    model = MultinomialNB()
    model.fit(X_scaled, yd_train)
    y_pred = model.predict(Xd_test)
    train_acc = model.score(X_scaled, yd_train)
    test_acc = model.score(Xd_test, yd_test)
    y_pred_prob = model.predict_proba(Xd_test)[:,1]
    auc = roc_auc_score(yd_test, y_pred_prob)
    tn, fp, fn, tp = confusion_matrix(yd_test, y_pred).ravel()
    prec = tp / (tp + fp)
    recall = tp / (tp + fn)
    fpr = fp / (tn + fp)
    rep = classification_report(yd_test, y_pred, output_dict=True)
    nb_df = pd.DataFrame({'model': 'bayes', 'hyper_params': None, 'train_acc': train_acc, 
                           'test_acc': test_acc, 'auc': auc, 'tp': tp, 'fn': fn, 'tn': tn, 'fp': fp, 
                           'precision': prec, 'recall': recall, 'fpr': fpr, 'neg_f1': rep['0']['f1-score'], 
                           'AD_f1': rep['1']['f1-score']}, index=[5])
    df = df.append(nb_df, ignore_index=True, sort=False)
    
    # return the dataframe
    return df

def run_models(Xd_train, Xd_test, yd_train, yd_test):
    """This function runs all of the classification data supplied through the models.
    
    Supply the training and test data.
    """
    
    # initialize dataframe to hold summary info for the models
    columns = ['model', 'hyper_params', 'train_acc', 'test_acc', 'auc', 'tp', 'fn', 'tn', 'fp',
              'precision', 'recall', 'fpr', 'neg_f1', 'AD_f1']
    df = pd.DataFrame(columns=columns)

    # knn model
    param_grid = {'n_neighbors': np.arange(1, 50)}
    knn = KNeighborsClassifier()
    knn_cv = GridSearchCV(knn, param_grid, cv=5)
    knn_cv.fit(Xd_train, yd_train)
    k = knn_cv.best_params_['n_neighbors']
    hp = 'k: {}'.format(k)
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(Xd_train, yd_train)
    y_pred = knn.predict(Xd_test)
    train_acc = knn.score(Xd_train, yd_train)
    test_acc = knn.score(Xd_test, yd_test)
    y_pred_prob = knn.predict_proba(Xd_test)[:,1]
    auc = roc_auc_score(yd_test, y_pred_prob)
    tn, fp, fn, tp = confusion_matrix(yd_test, y_pred).ravel()
    prec = tp / (tp + fp)
    recall = tp / (tp + fn)
    fpr = fp / (tn + fp)
    rep = classification_report(yd_test, y_pred, output_dict=True)
    knn_df = pd.DataFrame({'model': 'knn', 'hyper_params': hp, 'train_acc': train_acc, 'test_acc': test_acc,
               'auc': auc, 'tp': tp, 'fn': fn, 'tn': tn, 'fp': fp, 'precision': prec, 'recall': recall,
               'fpr': fpr, 'neg_f1': rep['0']['f1-score'], 'AD_f1': rep['1']['f1-score']}, index=[0])
    df = df.append(knn_df, ignore_index=True, sort=False)
    
    # SVM model
    # map the svm labels
    yd_train_svm = np.where(yd_train == 0, yd_train - 1, yd_train)
    yd_test_svm = np.where(yd_test == 0, yd_test - 1, yd_test)
    num_features = Xd_train.shape[1]
    param_grid = {'C': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5], 
              'gamma': [(1/(num_features*Xd_train.var())), (1/num_features)]}
    svm = SVC(class_weight='balanced', probability=True)
    svm_cv = GridSearchCV(svm, param_grid, cv=5)
    svm_cv.fit(Xd_train, yd_train_svm)
    C = svm_cv.best_params_['C']
    gamma = svm_cv.best_params_['gamma']
    hp = 'C: {}'.format(C) + ', gamma: {:.4f}'.format(gamma)
    svm = SVC(C=C, gamma=gamma, class_weight='balanced',
         probability=True)
    svm.fit(Xd_train, yd_train_svm)
    y_pred = svm.predict(Xd_test)
    train_acc = svm.score(Xd_train, yd_train_svm)
    test_acc = svm.score(Xd_test, yd_test_svm)
    y_pred_prob = svm.predict_proba(Xd_test)[:,1]
    auc = roc_auc_score(yd_test_svm, y_pred_prob)
    tn, fp, fn, tp = confusion_matrix(yd_test_svm, y_pred).ravel()
    prec = tp / (tp + fp)
    recall = tp / (tp + fn)
    fpr = fp / (tn + fp)
    rep = classification_report(yd_test_svm, y_pred, output_dict=True)
    roc_auc_score(yd_test_svm, y_pred_prob)
    svm_df = pd.DataFrame({'model': 'svm', 'hyper_params': hp, 'train_acc': train_acc, 
                           'test_acc': test_acc, 'auc': auc, 'tp': tp, 'fn': fn, 'tn': tn, 'fp': fp, 'precision': prec,
                           'recall': recall, 'fpr': fpr, 'neg_f1': rep['-1']['f1-score'], 'AD_f1': rep['1']['f1-score']},
                         index=[1])
    df = df.append(svm_df, ignore_index=True, sort=False)
    
    # Random Forests Model
    trees = [101, 111, 121, 131, 141, 151, 161, 171, 181, 191]
    max_f = [1, num_features, 'log2', 'sqrt']
    param_grid = {'n_estimators': trees, 'max_features': max_f}
    r_forest = RandomForestClassifier(class_weight='balanced', random_state=42)
    r_forest_cv = GridSearchCV(r_forest, param_grid, cv=5)
    r_forest_cv.fit(Xd_train, yd_train)
    n_est = r_forest_cv.best_params_['n_estimators']
    n_feat = r_forest_cv.best_params_['max_features']
    hp = 'trees: {}'.format(n_est) + ', max_feats: {}'.format(n_feat)
    rfc = RandomForestClassifier(n_estimators=n_est, max_features=n_feat, 
                             class_weight='balanced', random_state=42)
    rfc.fit(Xd_train, yd_train)
    y_pred = rfc.predict(Xd_test)
    train_acc = rfc.score(Xd_train, yd_train)
    test_acc = rfc.score(Xd_test, yd_test)
    y_pred_prob = rfc.predict_proba(Xd_test)[:,1]
    auc = roc_auc_score(yd_test, y_pred_prob)
    tn, fp, fn, tp = confusion_matrix(yd_test, y_pred).ravel()
    prec = tp / (tp + fp)
    recall = tp / (tp + fn)
    fpr = fp / (tn + fp)
    rep = classification_report(yd_test, y_pred, output_dict=True)
    rfc_df = pd.DataFrame({'model': 'RF', 'hyper_params': hp, 'train_acc': train_acc, 
                           'test_acc': test_acc, 'auc': auc, 'tp': tp, 'fn': fn, 'tn': tn, 'fp': fp, 
                           'precision': prec, 'recall': recall, 'fpr': fpr, 'neg_f1': rep['0']['f1-score'], 
                           'AD_f1': rep['1']['f1-score']}, index=[2])
    df = df.append(rfc_df, ignore_index=True, sort=False)
    
    # AdaBoost Classifier
    est = [31, 41, 51, 61, 71, 81, 91, 101]
    param_grid = {'n_estimators': est}
    boost = AdaBoostClassifier(random_state=42)
    boost_cv = GridSearchCV(boost, param_grid, cv=5)
    boost_cv.fit(Xd_train, yd_train)
    n_est = boost_cv.best_params_['n_estimators']
    hp = 'num_estimators: {}'.format(n_est)
    model = AdaBoostClassifier(n_estimators=n_est, random_state=0)
    model.fit(Xd_train, yd_train)
    y_pred = model.predict(Xd_test)
    train_acc = model.score(Xd_train, yd_train)
    test_acc = model.score(Xd_test, yd_test)
    y_pred_prob = model.predict_proba(Xd_test)[:,1]
    auc = roc_auc_score(yd_test, y_pred_prob)
    tn, fp, fn, tp = confusion_matrix(yd_test, y_pred).ravel()
    prec = tp / (tp + fp)
    recall = tp / (tp + fn)
    fpr = fp / (tn + fp)
    rep = classification_report(yd_test, y_pred, output_dict=True)
    boost_df = pd.DataFrame({'model': 'AdaBoost', 'hyper_params': hp, 'train_acc': train_acc, 
                           'test_acc': test_acc, 'auc': auc, 'tp': tp, 'fn': fn, 'tn': tn, 'fp': fp, 
                           'precision': prec, 'recall': recall, 'fpr': fpr, 'neg_f1': rep['0']['f1-score'], 
                           'AD_f1': rep['1']['f1-score']}, index=[3])
    df = df.append(boost_df, ignore_index=True, sort=False)
    
    # logistic regression
    logreg = linear_model.LogisticRegression(solver='lbfgs', class_weight='balanced', random_state=42)
    logreg.fit(Xd_train, yd_train)
    y_pred = logreg.predict(Xd_test)
    train_acc = logreg.score(Xd_train, yd_train)
    test_acc = logreg.score(Xd_test, yd_test)
    y_pred_prob = logreg.predict_proba(Xd_test)[:,1]
    auc = roc_auc_score(yd_test, y_pred_prob)
    tn, fp, fn, tp = confusion_matrix(yd_test, y_pred).ravel()
    prec = tp / (tp + fp)
    recall = tp / (tp + fn)
    fpr = fp / (tn + fp)
    rep = classification_report(yd_test, y_pred, output_dict=True)
    logreg_df = pd.DataFrame({'model': 'logreg', 'hyper_params': None, 'train_acc': train_acc, 
                           'test_acc': test_acc, 'auc': auc, 'tp': tp, 'fn': fn, 'tn': tn, 'fp': fp, 
                           'precision': prec, 'recall': recall, 'fpr': fpr, 'neg_f1': rep['0']['f1-score'], 
                           'AD_f1': rep['1']['f1-score']}, index=[4])
    df = df.append(logreg_df, ignore_index=True, sort=False)
    
    # Naive Bayes
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(Xd_train)
    model = MultinomialNB()
    model.fit(X_scaled, yd_train)
    y_pred = model.predict(Xd_test)
    train_acc = model.score(X_scaled, yd_train)
    test_acc = model.score(Xd_test, yd_test)
    y_pred_prob = model.predict_proba(Xd_test)[:,1]
    auc = roc_auc_score(yd_test, y_pred_prob)
    tn, fp, fn, tp = confusion_matrix(yd_test, y_pred).ravel()
    prec = tp / (tp + fp)
    recall = tp / (tp + fn)
    fpr = fp / (tn + fp)
    rep = classification_report(yd_test, y_pred, output_dict=True)
    nb_df = pd.DataFrame({'model': 'bayes', 'hyper_params': None, 'train_acc': train_acc, 
                           'test_acc': test_acc, 'auc': auc, 'tp': tp, 'fn': fn, 'tn': tn, 'fp': fp, 
                           'precision': prec, 'recall': recall, 'fpr': fpr, 'neg_f1': rep['0']['f1-score'], 
                           'AD_f1': rep['1']['f1-score']}, index=[5])
    df = df.append(nb_df, ignore_index=True, sort=False)
    
    # return the dataframe
    return df

def plot_dr_fpr(df):
    """This function accepts a dataframe and plots the detection rates and false positive rates.
    
    This is designed to work with the dataframe returned by the run_models() function, and
    column names must include 'model', 'recall', and 'fpr' for this function to work.
    """
    
    # plot the detection and false positive rates
    scores = df.reindex(columns=['model', 'recall', 'fpr'])
    if scores.loc[0,'recall'] < 1:
        scores.loc[:,'fpr'] = scores.loc[:,'fpr'].apply(lambda x: x * 100)
        scores.loc[:, 'recall'] = scores.loc[:, 'recall'].apply(lambda x: x * 100)
    scores.columns = ['model', 'Detection Rate', 'False Positive Rate']
    scores_melt = pd.melt(frame=scores, id_vars='model', value_vars=['Detection Rate', 'False Positive Rate'],
                          var_name='group', value_name='rate')
    ax = sns.barplot('model', 'rate', hue='group', data=scores_melt, palette='muted')
    _ = plt.setp(ax.get_xticklabels(), rotation=25)
    _ = plt.title('Scores for Each Model')
    _ = plt.ylabel('Rates (%)')
    _ = plt.xlabel('Model')
    _ = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
def run_deltas_ensemble(Xd_train, Xd_test, yd_train, yd_test, feature_names):
    """This function creates and returns information for an ensemble machine learning model.
    
    This model is designed specifically for this analysis and includes full feature SVM,
    logistic regression, and reduced feature logistic regression from feature selection.
    """
    
    # create -1, 1 labels for SVM
    ysvm_train = np.where(yd_train == 0, yd_train - 1, yd_train)
    ysvm_test = np.where(yd_test == 0, yd_test - 1, yd_test)
    
    # create the SVM model  
    num_features = Xd_train.shape[1]
    param_grid = {'C': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5], 
                  'gamma': [(1/(num_features*Xd_train.var())), (1/num_features)]}
    svm = SVC(class_weight='balanced', probability=True)
    svm_cv = GridSearchCV(svm, param_grid, cv=5)
    svm_cv.fit(Xd_train, ysvm_train)
    C = svm_cv.best_params_['C']
    gamma = svm_cv.best_params_['gamma']
    svm = SVC(C=C, gamma=gamma, class_weight='balanced', probability=True)
    svm.fit(Xd_train, ysvm_train)
    svm_pred_scaled = svm.predict(Xd_test)
    svm_pred = np.where(svm_pred_scaled == -1, svm_pred_scaled + 1, svm_pred_scaled)
    svm_prob = svm.predict_proba(Xd_test)[:,1]

    # Logistic regression (full feature)
    logreg = linear_model.LogisticRegression(solver='lbfgs', class_weight='balanced', random_state=42)
    logreg.fit(Xd_train, yd_train)
    logreg_pred = logreg.predict(Xd_test)
    logreg_prob = logreg.predict_proba(Xd_test)[:,1]

    # reduced logistic regression
    mask = (feature_names != 'ADAS13_delta') & (feature_names != 'PTGENDER_Male')
    Xtrain_reduced = Xd_train[:,mask]
    Xtest_reduced = Xd_test[:,mask]
    red_logreg = linear_model.LogisticRegression(solver='lbfgs', class_weight='balanced', random_state=42)
    red_logreg.fit(Xtrain_reduced, yd_train)
    red_logreg_pred = red_logreg.predict(Xtest_reduced)
    red_logreg_prob = red_logreg.predict_proba(Xtest_reduced)[:,1]
    
    # create a dataframe and count the votes
    pred = pd.DataFrame({'svm': svm_pred, 'lr_ff': logreg_pred, 'lr_red': red_logreg_pred})
    pred.loc[:,'total'] = pred.svm + pred.lr_ff + pred.lr_red
    mapper = {0: 0, 1: 0, 2: 1, 3: 1}
    y_pred = pred.total.map(mapper)
    
    # print the results
    print(confusion_matrix(yd_test, y_pred))
    tn, fp, fn, tp = confusion_matrix(yd_test, y_pred).ravel()
    dr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    print('True Negatives: {}'.format(tn))
    print('False Positives: {}'.format(fp))
    print('False Negatives: {}'.format(fn))
    print('True Positives: {}'.format(tp))
    print('Detection Rate: {}'.format(dr))
    print('False Positive Rate: {}'.format(fpr))
    
def run_bl_ensemble(Xd_train, Xd_test, yd_train, yd_test, feature_names):
    """This function creates and returns information for an ensemble machine learning model.
    
    This model is designed specifically for this analysis and includes full feature SVM and 
    logistic regression, 9 component pca for SVM and logistic regression, and logistic regression
    omitting ADAS13_bl and PTGENDER_Male.
    """
    
    # create -1, 1 labels for SVM
    ysvm_train = np.where(yd_train == 0, yd_train - 1, yd_train)
    ysvm_test = np.where(yd_test == 0, yd_test - 1, yd_test)
    
    # create the SVM model  
    num_features = Xd_train.shape[1]
    param_grid = {'C': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5], 
                  'gamma': [(1/(num_features*Xd_train.var())), (1/num_features)]}
    svm = SVC(class_weight='balanced', probability=True)
    svm_cv = GridSearchCV(svm, param_grid, cv=5)
    svm_cv.fit(Xd_train, ysvm_train)
    C = svm_cv.best_params_['C']
    gamma = svm_cv.best_params_['gamma']
    svm = SVC(C=C, gamma=gamma, class_weight='balanced', probability=True)
    svm.fit(Xd_train, ysvm_train)
    svm_pred_scaled = svm.predict(Xd_test)
    svm_pred = np.where(svm_pred_scaled == -1, svm_pred_scaled + 1, svm_pred_scaled)
    svm_prob = svm.predict_proba(Xd_test)[:,1]

    # Logistic regression (full feature)
    logreg = linear_model.LogisticRegression(solver='lbfgs', class_weight='balanced', random_state=42)
    logreg.fit(Xd_train, yd_train)
    logreg_pred = logreg.predict(Xd_test)
    logreg_prob = logreg.predict_proba(Xd_test)[:,1]

    # reduced logistic regression
    mask = (feature_names != 'ADAS13_bl') & (feature_names != 'PTGENDER_Male')
    Xtrain_reduced = Xd_train[:,mask]
    Xtest_reduced = Xd_test[:,mask]
    red_logreg = linear_model.LogisticRegression(solver='lbfgs', class_weight='balanced', random_state=42)
    red_logreg.fit(Xtrain_reduced, yd_train)
    red_logreg_pred = red_logreg.predict(Xtest_reduced)
    red_logreg_prob = red_logreg.predict_proba(Xtest_reduced)[:,1]
    
    # create PCA model with 9 components
    pca = PCA(n_components=9)
    pca.fit(Xd_train)
    Xpca_train = pca.transform(Xd_train)
    Xpca_test = pca.transform(Xd_test)
    
    # SVM PCA
    num_features = Xpca_train.shape[1]
    param_grid = {'C': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5], 
                  'gamma': [(1/(num_features*Xpca_train.var())), (1/num_features)]}
    svm_pca = SVC(class_weight='balanced', probability=True)
    svm_pca_cv = GridSearchCV(svm_pca, param_grid, cv=5)
    svm_pca_cv.fit(Xpca_train, ysvm_train)
    C_pca = svm_pca_cv.best_params_['C']
    gamma_pca = svm_pca_cv.best_params_['gamma']
    svm_pca = SVC(C=C_pca, gamma=gamma_pca, class_weight='balanced', probability=True)
    svm_pca.fit(Xpca_train, ysvm_train)
    svm_pca_pred_scaled = svm_pca.predict(Xpca_test)
    svm_pca_pred = np.where(svm_pca_pred_scaled == -1, svm_pca_pred_scaled + 1, svm_pca_pred_scaled)
    svm_pca_prob = svm_pca.predict_proba(Xpca_test)[:,1]
    
    # logreg PCA
    logreg_pca = linear_model.LogisticRegression(solver='lbfgs', class_weight='balanced', random_state=42)
    logreg_pca.fit(Xpca_train, yd_train)
    logreg_pca_pred = logreg_pca.predict(Xpca_test)
    logreg_pca_prob = logreg_pca.predict_proba(Xpca_test)[:,1]
    
    # create a dataframe and count the votes
    pred = pd.DataFrame({'svm_ff': svm_pred, 'lr_ff': logreg_pred, 'lr_red': red_logreg_pred,
                        'svm_pca': svm_pca_pred, 'lr_pca': logreg_pca_pred})
    pred.loc[:,'total'] = pred.svm_ff + pred.lr_ff + pred.lr_red + pred.svm_pca + pred.lr_pca
    mapper = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}
    y_pred = pred.total.map(mapper)
    
    # print the results
    print(confusion_matrix(yd_test, y_pred))
    tn, fp, fn, tp = confusion_matrix(yd_test, y_pred).ravel()
    dr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    print('True Negatives: {}'.format(tn))
    print('False Positives: {}'.format(fp))
    print('False Negatives: {}'.format(fn))
    print('True Positives: {}'.format(tp))
    print('Detection Rate: {}'.format(dr))
    print('False Positive Rate: {}'.format(fpr))