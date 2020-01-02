#!/usr/bin/env python3
"""Exploratory Data Analysis Module for Alzheimer's Capstone 1.

This module contains functions used for Data Storytelling.
The functions within wrangle and format data for visualizations.
Inputs for these functions can be obtained using adnidatawrangling module.
Required modules for all functions to run: pandas, numpy, matplotlib.pyplot,
seaborn all using standard namespaces.
"""

if 'pd' not in globals():
    import pandas as pd

if 'np' not in globals():
    import numpy as np
    
if 'plt' not in globals():
    import matplotlib.pyplot as plt
    
if 'sns' not in globals():
    import seaborn as sns
    
sns.set()

def get_bl_data(adni_comp, clin_data, scan_data):
    """This function extracts the data from the baseline visit only for each patient.
    
    Supply the three dataframes adni_comp, clin_data, and scan_data as input.
    """
    
    # extract the baseline data only
    adni_bl = adni_comp[adni_comp.EXAMDATE == adni_comp.EXAMDATE_bl]
    clin_bl = clin_data[clin_data.EXAMDATE == clin_data.EXAMDATE_bl]
    scan_bl = scan_data[scan_data.EXAMDATE == scan_data.EXAMDATE_bl]
    
    # return the three dataframes
    return adni_bl, clin_bl, scan_bl

def plot_gender_counts(adni_bl):
    """This function plots gender counts from the complete data.
    
    Supply adni_bl as input to ensure only 1 row per patient.
    """
    
    # construct and show the plot
    adni_bl['PTGENDER'].value_counts().plot(kind='bar', rot=0)
    plt.xlabel('Gender')
    plt.ylabel('Number of Patients')
    plt.title('Number of Patients of Each Gender')
    plt.show()

def plot_bl_diag(adni_bl):
    """This function plots baseline data.
    
    Supply adni_bl as the argument and unpack the x, y variables for plotting.
    """
    
    x = ['CN', 'SMC', 'EMCI', 'LMCI', 'AD']
    y = [adni_bl.DX_bl.value_counts()['CN'], adni_bl.DX_bl.value_counts()['SMC'],
         adni_bl.DX_bl.value_counts()['EMCI'], adni_bl.DX_bl.value_counts()['LMCI'],
         adni_bl.DX_bl.value_counts()['AD']]
    
    # construct and show the plot
    sns.barplot(x=x, y=y)
    plt.xlabel('Diagnosis')
    plt.ylabel('Number of Patients')
    plt.title('Baseline Diagnosis')
    plt.show()

def plot_final_diag(adni_comp):
    """This function extracts the final diagnosis counts for plotting.
    
    Supply adni_comp as input and extract the x, y data for plotting.
    """
    
    # filter the most advanced diagnosis for each patient
    dx_final = adni_comp.groupby('RID')['DX'].max()
    
    # get the frequency of occurrence
    dx_counts = dx_final.value_counts()
    
    # extract the data and assign to x, y
    x = ['CN', 'MCI', 'AD']
    y = [dx_counts['CN'], dx_counts['MCI'], dx_counts['AD']]
    
    # construct and show the plot
    sns.barplot(x=x, y=y)
    plt.xlabel('Diagnosis')
    plt.ylabel('Number of Patients')
    plt.title('Final Diagnosis')
    plt.show()

def plot_bl_final_diag(adni_comp):
    """This function extracts baseline and final diagnosis data and plots it.
    
    Provide the adni_comp df as input and extract 'diag' df for plotting.
    Use the following call to plot:
    sns.barplot(x='diagnosis', y='counts', hue='visit', data=diag)
    Add appropriate titles, labels as needed.
    """
    
    # filter the most advanced diagnosis for each patient
    dx_final = adni_comp.groupby('RID')['DX'].max()
    
    # get the frequency of occurrence
    dx_counts = dx_final.value_counts()
    
    # extract the basline diagnosis counts, using bl2
    DX_bl2 = adni_comp.groupby('RID')['DX_bl2'].min()
    DX_bl2_counts = DX_bl2.value_counts()
    
    
    # join baseline and final diagnosis data for comparison
    counts = [DX_bl2_counts[1], DX_bl2_counts[0], DX_bl2_counts[2], dx_counts[2], dx_counts[1], dx_counts[0]]
    dx_comp = {'visit': ['Baseline', 'Baseline', 'Baseline', 'Final', 'Final', 'Final'], 
               'diagnosis': ['CN', 'MCI', 'AD', 'CN', 'MCI', 'AD'],
               'counts': counts}
    
    # create dataframe using dx_comp
    diag = pd.DataFrame(dx_comp)
    
    sns.barplot(x='diagnosis', y='counts', hue='visit', data=diag)
    plt.xlabel('Diagnosis')
    plt.ylabel('Number of Patients')
    plt.title('Diagnosis Counts: Baseline vs. Final')
    plt.legend(loc='upper left')
    plt.show()

def get_final_exam(adni_comp):
    """This function returns only the last or final exam for each patient.
    
    Supply the adni_comp dataframe as an argument and extract the final_exam dataframe.
    """
    
    # reset the index and add examdate to the index
    adni_comp.reset_index(inplace=True)
    
    # create a list of patient ID's
    RID_list = adni_comp.RID.unique()
    
    # set the index back
    adni_comp.set_index([adni_comp.RID, adni_comp.EXAMDATE], inplace=True)
    adni_comp.drop('RID', axis='columns', inplace=True)
    
    # create a list of the last exam date for each patient
    last_exam = adni_comp.groupby('RID')['EXAMDATE'].max()
    
    # intialize empty dictionary
    data_dict = {}
    
    # loop through the dictionary, use the RID and last exam date as the index to extract each row
    for i in range(len(RID_list)):
        data_dict[RID_list[i]] = adni_comp.loc[(RID_list[i], str(last_exam[RID_list[i]]))]
    
    # create the dataframe and transpose columns/rows
    final_exam = pd.DataFrame(data_dict)
    final_exam = final_exam.transpose()
    
    return final_exam

def get_dx_groups(adni_comp):
    """This function separates the data into diagnosis groups.
    
    Each dataframe that is returned follows a group of patients based on their 
    progression during the study. cn_cn means a group that was 'CN' at baseline
    and 'CN' at final exam, while mci_ad means a group that was 'MCI' at baseline
    and 'AD' at final exam.
    
    Returns a tuple containing the following items:
    cn_cn, cn_mci, cn_ad, mci_cn, mci_mci, mci_ad, ad_cn, ad_mci, ad_ad
    """
    
    # filter the data
    cn_cn = adni_comp[(adni_comp.DX == 'CN') & (adni_comp.DX_bl2 == 'CN')]
    cn_mci = adni_comp[(adni_comp.DX == 'MCI') & (adni_comp.DX_bl2 == 'CN')]
    cn_ad = adni_comp[(adni_comp.DX == 'AD') & (adni_comp.DX_bl2 == 'CN')]
    mci_cn = adni_comp[(adni_comp.DX == 'CN') & (adni_comp.DX_bl2 == 'MCI')]
    mci_mci = adni_comp[(adni_comp.DX == 'MCI') & (adni_comp.DX_bl2 == 'MCI')]
    mci_ad = adni_comp[(adni_comp.DX == 'AD') & (adni_comp.DX_bl2 == 'MCI')]
    ad_cn = adni_comp[(adni_comp.DX == 'CN') & (adni_comp.DX_bl2 == 'AD')]
    ad_mci = adni_comp[(adni_comp.DX == 'MCI') & (adni_comp.DX_bl2 == 'AD')]
    ad_ad = adni_comp[(adni_comp.DX == 'AD') & (adni_comp.DX_bl2 == 'AD')]
    
    return (cn_cn, cn_mci, cn_ad, mci_cn, mci_mci, mci_ad, ad_cn, ad_mci, ad_ad)

def get_final_dx_groups(final_exam):
    """This function separates the final exam data into diagnosis groups.
    
    Each dataframe that is returned follows a group of patients based on their 
    progression during the study. cn_cn means a group that was 'CN' at baseline
    and 'CN' at final exam, while mci_ad means a group that was 'MCI' at baseline
    and 'AD' at final exam. This function only contains data for the final exam
    
    Returns a tuple containing the following items:
    cn_cn_f, cn_mci_f, cn_ad_f, mci_cn_f, mci_mci_f, mci_ad_f, ad_cn_f, ad_mci_f, ad_ad_f
    """
    
    # filter the data
    cn_cn_f = final_exam[(final_exam.DX == 'CN') & (final_exam.DX_bl2 == 'CN')]
    cn_mci_f = final_exam[(final_exam.DX == 'MCI') & (final_exam.DX_bl2 == 'CN')]
    cn_ad_f = final_exam[(final_exam.DX == 'AD') & (final_exam.DX_bl2 == 'CN')]
    mci_cn_f = final_exam[(final_exam.DX == 'CN') & (final_exam.DX_bl2 == 'MCI')]
    mci_mci_f = final_exam[(final_exam.DX == 'MCI') & (final_exam.DX_bl2 == 'MCI')]
    mci_ad_f = final_exam[(final_exam.DX == 'AD') & (final_exam.DX_bl2 == 'MCI')]
    ad_cn_f = final_exam[(final_exam.DX == 'CN') & (final_exam.DX_bl2 == 'AD')]
    ad_mci_f = final_exam[(final_exam.DX == 'MCI') & (final_exam.DX_bl2 == 'AD')]
    ad_ad_f = final_exam[(final_exam.DX == 'AD') & (final_exam.DX_bl2 == 'AD')]
    
    return (cn_cn_f, cn_mci_f, cn_ad_f, mci_cn_f, mci_mci_f, mci_ad_f, ad_cn_f, ad_mci_f, ad_ad_f)

def plot_dx_change(final_exam):
    """This function creates a stacked bar chart showing the different diagnosis groups
    
    Supply the final_exam dataframe as input.
    """
    
    # filter the data
    cn_cn_f = final_exam[(final_exam.DX == 'CN') & (final_exam.DX_bl2 == 'CN')]
    cn_mci_f = final_exam[(final_exam.DX == 'MCI') & (final_exam.DX_bl2 == 'CN')]
    cn_ad_f = final_exam[(final_exam.DX == 'AD') & (final_exam.DX_bl2 == 'CN')]
    mci_cn_f = final_exam[(final_exam.DX == 'CN') & (final_exam.DX_bl2 == 'MCI')]
    mci_mci_f = final_exam[(final_exam.DX == 'MCI') & (final_exam.DX_bl2 == 'MCI')]
    mci_ad_f = final_exam[(final_exam.DX == 'AD') & (final_exam.DX_bl2 == 'MCI')]
    ad_cn_f = final_exam[(final_exam.DX == 'CN') & (final_exam.DX_bl2 == 'AD')]
    ad_mci_f = final_exam[(final_exam.DX == 'MCI') & (final_exam.DX_bl2 == 'AD')]
    ad_ad_f = final_exam[(final_exam.DX == 'AD') & (final_exam.DX_bl2 == 'AD')]
    
    # create a dictionary and convert to a dataframe
    diag_dict = {'CN': [cn_cn_f.shape[0], cn_mci_f.shape[0], cn_ad_f.shape[0]], 
                 'MCI': [mci_cn_f.shape[0], mci_mci_f.shape[0], mci_ad_f.shape[0]],
                 'AD': [ad_cn_f.shape[0], ad_mci_f.shape[0], ad_ad_f.shape[0]]}
    diag_df = pd.DataFrame(diag_dict)
    
    # extract the bar heights
    ad_final = [diag_df.loc[:,'CN'].sum(), diag_df.loc[:,'MCI'].sum(), diag_df.loc[:,'AD'].sum()]
    mci_final = [diag_df.loc[0:1, 'CN'].sum(), diag_df.loc[0:1, 'MCI'].sum(), diag_df.loc[0:1, 'AD'].sum()]
    cn_final = [diag_df.loc[0, 'CN'], diag_df.loc[0, 'MCI'], diag_df.loc[0, 'AD']]
    
    # contruct and show the plot
    sns.barplot(x=['CN', 'MCI', 'AD'], y=ad_final, color='indianred', label='AD')
    sns.barplot(x=['CN', 'MCI', 'AD'], y=mci_final, color='limegreen', label='MCI')
    sns.barplot(x=['CN', 'MCI', 'AD'], y=cn_final, color='royalblue', label='CN')
    plt.xlabel('Baseline Diagnosis')
    plt.ylabel('Number of Patients')
    plt.title('Final Diagnosis from Baseline Diagnosis')
    plt.legend(title='Final Diagnosis')
    plt.show()

def calc_deltas(final_exam):
    """This function calculates changes in each biomarker and adds that column to
    the final exam dataframe.
    
    Supply the final_exam dataframe, and the columns are added in place.
    """
    
    final_exam.loc[:, 'CDRSB_delta'] = final_exam.CDRSB - final_exam.CDRSB_bl
    final_exam.loc[:, 'ADAS11_delta'] = final_exam.ADAS11 - final_exam.ADAS11_bl
    final_exam.loc[:, 'ADAS13_delta'] = final_exam.ADAS13 - final_exam.ADAS13_bl
    final_exam.loc[:, 'MMSE_delta'] = final_exam.MMSE - final_exam.MMSE_bl
    final_exam.loc[:, 'RAVLT_delta'] = final_exam.RAVLT_immediate - final_exam.RAVLT_immediate_bl
    final_exam.loc[:, 'Hippocampus_delta'] = final_exam.Hippocampus - final_exam.Hippocampus_bl
    final_exam.loc[:, 'Ventricles_delta'] = final_exam.Ventricles - final_exam.Ventricles_bl
    final_exam.loc[:, 'WholeBrain_delta'] = final_exam.WholeBrain - final_exam.WholeBrain_bl
    final_exam.loc[:, 'Entorhinal_delta'] = final_exam.Entorhinal - final_exam.Entorhinal_bl
    final_exam.loc[:, 'MidTemp_delta'] = final_exam.MidTemp - final_exam.MidTemp_bl

def setup_dist_plots(final_exam):
    """This function creates parameters needed by the plot_dist() and plot_hist() functions.
    
    This must be called prior to the plot_hist() or plot_dist() functions. The final_exam dataframe is used.
    Unpack dist_groups and dist_bins, which serve as default arguments for plot_hist() and plot_dist().
    """
    
    # isolate patients with no diagnosis change
    no_change = final_exam[final_exam['DX'] == final_exam['DX_bl2']]
    
    # isolate patients who progressed from 'CN' to 'AD'
    cn_mci = final_exam[(final_exam['DX'] == 'MCI') & (final_exam['DX_bl2'] == 'CN')]
    
    # isolate patients who progressed from 'MCI' to 'AD'
    mci_ad = final_exam[(final_exam['DX'] == 'AD') & (final_exam['DX_bl2'] == 'MCI')]
    
    # isolate patients who progressed from 'CN' to 'AD'
    cn_ad = final_exam[(final_exam['DX'] == 'AD') & (final_exam['DX_bl2'] == 'CN')]
    
    # calculate the number of observations to use in calcuting number of bins for histograms
    no_change_bins = int(np.sqrt(no_change.shape[0]))
    cn_mci_bins = int(np.sqrt(cn_mci.shape[0]))
    mci_ad_bins = int(np.sqrt(mci_ad.shape[0]))
    cn_ad_bins = int(np.sqrt(cn_ad.shape[0]))
    
    # pack the data
    dist_groups = (no_change, cn_mci, mci_ad, cn_ad)
    labels = ('No Change', 'CN to MCI', 'MCI to AD', 'CN to AD')
    dist_bins = (no_change_bins, cn_mci_bins, mci_ad_bins, cn_ad_bins)
    
    return dist_groups, labels, dist_bins
    
def plot_dist(column, dist_groups, labels):
    """This function accepts a column name and creates a sns.distplot() of that column from the
    final_exam dataframe.
    
    The default grouping is the grouping from setup_dist_plots().
    """
   
    # loop through groups to build the plot
    for i in range(len(labels)):
        sns.distplot(dist_groups[i][column].values, hist=False, label=labels[i])

    # format labels and legend
    
    plt.ylabel('Kernel Density Estimate')
    if column.find('_delta') != -1:
        plt.xlabel(column[:-6] + ' Change')
        plt.title(column[:-6] + ' Change by Change in Diagnosis')
    elif column.find('_bl') != -1:
        plt.xlabel('Baseline ' + column[:-3])
        plt.title('Baseline ' + column[:-3] + '\nby Change in Diagnosis')
    else:
        plt.xlabel(column)
        plt.title(column + ' by Change in Diagnosis')
    plt.legend(loc='best')
    plt.show()
    
def plot_hist(column, dist_groups, labels, bins):
    """Supply a column name to create histogram plots similar to the distplots.
    
    Defaults come from the setup_dist_plot function. This function is recommended
    for plotting the '_delta' columns, but will work for other columns as well.
    """
    
    # set the color sequence (some extras added just in case)
    colors = ['blue', 'yellow', 'green', 'red', 'orange', 'purple']
    
    # set the alpha sequence
    alphas = [0.6, 0.5, 0.4, 0.2, 0.2, 0.2]
    
    # loop through groups to build the plot
    for i in range(len(labels)):
        plt.hist(dist_groups[i][column].values, bins=bins[i], density=True,
                alpha=alphas[i], color=colors[i], label=labels[i])
        
    if column.find('_delta') != -1:
        plt.xlabel(column[:-6] + ' Change')
        plt.title('Change in ' + column[:-6] + ': Baseline to Final\nby Change in Diagnosis')
    elif column.find('_bl') != -1:
        plt.xlabel('Baseline ' + column[:-3])
        plt.title('Baseline ' + column[:-3] + '\nby Change in Diagnosis')
    else:
        plt.xlabel(column + ' Values')
        plt.title(column + ' Distribution by Final Diagnosis')
    plt.ylabel('Probability Density')
    plt.legend(loc='best')
    plt.show()
    
def plot_histdist(column, dist_groups, labels, bins):
    """This function will create side beside plots of the histograms and distributions
    similar to the plot_hist and plot_dist functions above.
    
    Supply column name, dist_groups, labels, and bins to produce side by side plots.
    """
    
    # set the color sequence (some extras added just in case)
    colors = ['blue', 'yellow', 'green', 'red', 'orange', 'purple']
    
    # set the alpha sequence
    alphas = [0.6, 0.5, 0.4, 0.2, 0.2, 0.2]
    
    # set the subplot
    plt.rcParams["figure.figsize"] = (14,4)
    plt.subplot(1, 2, 1)
    
    # loop through groups to build the plot
    for i in range(len(labels)):
        plt.hist(dist_groups[i][column].values, bins=bins[i], density=True,
                alpha=alphas[i], color=colors[i], label=labels[i])
        
    if column.find('_delta') != -1:
        plt.xlabel(column[:-6] + ' Change')
        plt.title('Change in ' + column[:-6] + ': Baseline to Final\nby Change in Diagnosis')
    elif column.find('_bl') != -1:
        plt.xlabel('Baseline ' + column[:-3])
        plt.title('Baseline ' + column[:-3] + '\nby Change in Diagnosis')
    else:
        plt.xlabel(column + ' Values')
        plt.title(column + ' Distribution by Final Diagnosis')
    plt.ylabel('Probability Density')
    plt.legend(loc='best')
    
    # change the subplot
    plt.subplot(1, 2, 2)
    
    # loop through groups to build the plot
    for i in range(len(labels)):
        sns.distplot(dist_groups[i][column].values, hist=False, label=labels[i])

    # format labels and legend
    plt.ylabel('Kernel Density Estimate')
    if column.find('_delta') != -1:
        plt.xlabel(column[:-6] + ' Change')
        plt.title(column[:-6] + ' Change by Change in Diagnosis')
    elif column.find('_bl') != -1:
        plt.xlabel('Baseline ' + column[:-3])
        plt.title('Baseline ' + column[:-3] + '\nby Change in Diagnosis')
    else:
        plt.xlabel(column)
        plt.title(column + ' by Change in Diagnosis')
    plt.legend(loc='best')
    
    plt.tight_layout()
    plt.subplots_adjust(top=1.2, wspace=0.2)
    
    # show the plot
    plt.show()
    
def setup_indicators(final_exam):
    """This function sets up the data needed to run plot_bl_indicators function.
    
    Store the results of this functions in setup_indicators variable.
    """
    
    ad_f = final_exam[(final_exam.DX == 'AD') & (final_exam.PTGENDER == 'Female')]
    ad_m = final_exam[(final_exam.DX == 'AD') & (final_exam.PTGENDER == 'Male')]
    mci_f = final_exam[(final_exam.DX == 'MCI') & (final_exam.PTGENDER == 'Female')]
    mci_m = final_exam[(final_exam.DX == 'MCI') & (final_exam.PTGENDER == 'Male')]
    cn_f = final_exam[(final_exam.DX == 'CN') & (final_exam.PTGENDER == 'Female')]
    cn_m = final_exam[(final_exam.DX == 'CN') & (final_exam.PTGENDER == 'Male')]
    
    return (cn_f, cn_m, mci_f, mci_m, ad_f, ad_m)

def plot_indicators(column1, si, column2=''):
    """This function accepts two column names, in particular baseline columns.
    
    Default data is supplied as input from the setup_indicators function.
    The function accepts baseline '_bl' or non-baseline data.
    This function will produce 4 plots if two columns are supplied and 2 plots if only 
    one is supplied.
    """
    
    # unpack the setup_indicators
    cn_f = si[0]
    cn_m = si[1]
    mci_f = si[2]
    mci_m = si[3]
    ad_f = si[4]
    ad_m = si[5]
    
    # do bin calculations
    cn_f_bins = int(np.sqrt(cn_f.shape[0]))
    mci_f_bins = int(np.sqrt(mci_f.shape[0]))
    ad_f_bins = int(np.sqrt(ad_f.shape[0]))
    cn_m_bins = int(np.sqrt(cn_m.shape[0]))
    mci_m_bins = int(np.sqrt(mci_m.shape[0]))
    ad_m_bins = int(np.sqrt(ad_m.shape[0]))
    
    # configure the plots
    # setup 2 x 2 if two columns provided
    # setup 1 x 2 if only one column provided
    if column2 == '':
        plt.rcParams["figure.figsize"] = (14,5)
    else:
        plt.rcParams["figure.figsize"] = (14,8)
    
    if column2 != '': 
        plt.subplot(2, 2, 1)
    else:
        plt.subplot(1, 2, 1)
    plt.hist(cn_f[column1], bins=cn_f_bins, density=True, alpha=0.3, color='blue', label='CN')
    plt.hist(mci_f[column1], bins=mci_f_bins, density=True, alpha=0.3, color='green', label='MCI')
    plt.hist(ad_f[column1], bins=ad_f_bins, density=True, alpha=0.3, color='red', label='AD')
    if column1.find('_bl') != -1:
        plt.xlabel('Baseline ' + column1[:-3])
        plt.title('Female Baseline ' + column1[:-3] + ' Score\nSeparated by Final Diagnosis')
    else:
        plt.xlabel(column1)
        plt.title('Female ' + column1 + ' Score\nSeparated by Final Diagnosis')
    plt.ylabel('Probability Density')
    plt.legend(title='Final Diagnosis', loc='best')
    
    if column2 != '':
        plt.subplot(2, 2, 2)
        plt.hist(cn_f[column2], bins=cn_f_bins, density=True, alpha=0.3, color='blue', label='CN')
        plt.hist(mci_f[column2], bins=mci_f_bins, density=True, alpha=0.3, color='green', label='MCI')
        plt.hist(ad_f[column2], bins=ad_f_bins, density=True, alpha=0.3, color='red', label='AD')
        if column2.find('_bl') != -1:
            plt.xlabel('Baseline ' + column2[:-3])
            plt.title('Female Baseline ' + column2[:-3] + ' Score\nSeparated by Final Diagnosis')
        else:
            plt.xlabel(column2)
            plt.title('Female ' + column2 + ' Score\nSeparated by Final Diagnosis')
        plt.ylabel('Probability Density')
        plt.legend(title='Final Diagnosis', loc='best')

    if column2 != '':
        plt.subplot(2, 2, 3)
    else:
        plt.subplot(1, 2, 2)
    plt.hist(cn_m[column1], bins=cn_m_bins, density=True, alpha=0.3, color='blue', label='CN')
    plt.hist(mci_m[column1], bins=mci_m_bins, density=True, alpha=0.3, color='green', label='MCI')
    plt.hist(ad_m[column1], bins=ad_m_bins, density=True, alpha=0.3, color='red', label='AD')
    if column1.find('_bl') != -1:
        plt.xlabel('Baseline ' + column1[:-3])
        plt.title('Male Baseline ' + column1[:-3] + ' Score\nSeparated by Final Diagnosis')
    else:
        plt.xlabel(column1)
        plt.title('Male ' + column1 + ' Score\nSeparated by Final Diagnosis')
    plt.ylabel('Probability Density')
    plt.legend(title='Final Diagnosis', loc='best')

    if column2 != '':
        plt.subplot(2, 2, 4)
        plt.hist(cn_m[column2], bins=cn_m_bins, density=True, alpha=0.3, color='blue', label='CN')
        plt.hist(mci_m[column2], bins=mci_m_bins, density=True, alpha=0.3, color='green', label='MCI')
        plt.hist(ad_m[column2], bins=ad_m_bins, density=True, alpha=0.3, color='red', label='AD')
        if column2.find('_bl') != -1:
            plt.xlabel('Baseline ' + column2[:-3])
            plt.title('Male Baseline ' + column2[:-3] + ' Score\nSeparated by Final Diagnosis')
        else:
            plt.xlabel(column2)
            plt.title('Male ' + column2 + ' Score\nSeparated by Final Diagnosis')
        plt.ylabel('Probability Density')
        plt.legend(title='Final Diagnosis', loc='best')

    plt.tight_layout()
    plt.subplots_adjust(top=1.2)
    
    plt.show()
    
def summarize_bl_thresholds(final_exam, column, gender, threshold):
    """This function provides some summary information about the number of patients with 
    certain threshold values were diagnosed with Alzheimer's disease.
    
    Provide the final_exam dataframe, baseline column, patient gender, and threshold value to check.
    """
    
    # ensure to search the correct side of the threshold values        
    if column in ['Hippocampus_bl', 'Hippocampus', 'MidTemp', 'MidTemp_bl', 'WholeBrain', 'WholeBrain_bl',
                 'Entorhinal', 'Entorhinal_bl']:
        
        # calculate the number of patients at threshold that ended up with AD
        end_ad = final_exam[(final_exam.DX == 'AD') & (final_exam.PTGENDER == gender) 
                            & (final_exam[column] <= threshold)].shape[0]
        
        # calcualte the number of patients below the threshold that already had AD diagnosis
        had_ad = final_exam[(final_exam[column] <= threshold) & (final_exam.PTGENDER == gender) 
                            & (final_exam.DX_bl2 == 'AD')].shape[0]
        
        # calculate the number of patients below threshold that didn't already have AD
        not_ad = final_exam[(final_exam[column] <= threshold) & (final_exam.PTGENDER == gender) 
                            & (final_exam.DX_bl2 != 'AD')].shape[0]
    
    else:
        
        # calculate the number of patients at threshold that ended up with AD
        end_ad = final_exam[(final_exam.DX == 'AD') & (final_exam.PTGENDER == gender) 
                            & (final_exam[column] >= threshold)].shape[0]
        
        # calcualte the number of patients above the threshold that already had AD diagnosis
        had_ad = final_exam[(final_exam[column] >= threshold) & (final_exam.PTGENDER == gender) 
                            & (final_exam.DX_bl2 == 'AD')].shape[0]
        
        # calculate the number of patients above threshold that didn't already have AD
        not_ad = final_exam[(final_exam[column] >= threshold) & (final_exam.PTGENDER == gender) 
                            & (final_exam.DX_bl2 != 'AD')].shape[0]
        
    # calculate the numbers and percentages of patients that show predictive power
    got_ad = end_ad - had_ad
        
    # calculate the percentage that didn't have AD that were later diagnosed AD
    per_ad = round((got_ad / not_ad * 100), 2)
    
    str1 = ' patients had baseline ' + column + ' values exceeding the threshold of '
    str2 = '% of patients that didn\'t have AD yet but had '
        
    print(str(got_ad) + ' of ' + str(not_ad) + str1 + str(threshold) + ' \nand ended with AD.')
    print()
    print(str(per_ad) + str2 + column + ' \nexceeding threshold value of ' + str(threshold) + ' ended with AD.')
        