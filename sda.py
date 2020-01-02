#!/usr/bin/env python3
"""Statistical data analysis module for Alzheimer's Capstone 1.

This module contains functions used to extract data for and to perform
statistical analysis on the ADNI Alzheimer's Disease dataset for my
Capstone project. Inputs for these functions can be obtained using 
adnidatawrangling module, and some additional wrangling from the eda module.
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
    
if 'scipy.stats' not in globals():
    import scipy.stats
    
sns.set()

def get_deltadx_groups(df):
    """This function uses the supplied dataframe to divide the data by diagnosis change.
    
    The groups returned by this function are no_change, cn_mci, mci_ad, and cn_ad.
    """
    
    # isolate patients with no diagnosis change
    no_change = df[df['DX'] == df['DX_bl2']]
    
    # isolate patients who progressed from 'CN' to 'AD'
    cn_mci = df[(df['DX'] == 'MCI') & (df['DX_bl2'] == 'CN')]
    
    # isolate patients who progressed from 'MCI' to 'AD'
    mci_ad = df[(df['DX'] == 'AD') & (df['DX_bl2'] == 'MCI')]
    
    # isolate patients who progressed from 'CN' to 'AD'
    cn_ad = df[(df['DX'] == 'AD') & (df['DX_bl2'] == 'CN')]
    
    return no_change, cn_mci, mci_ad, cn_ad

def divide_genders(df):
    """This function divides the supplied dataframe by the 'PTGENDER' column.
    
    Returns two dataframes: males, females.
    """
    
    males = df[df.PTGENDER == 'Male']
    females = df[df.PTGENDER == 'Female']
    
    return males, females

def test_gender_effect(df, biomarker, size):
    """This function returns the p value for the test that males/females have the 
    
    same distribution for the change in the supplied biomarker. The test will be 
    performed over 'size' permutations. A significant p value means that males/females 
    should be separated for further analysis of change in in the supplied biomarker. 
    A high p value means males/females can be considered together.
    """
    
    # create a combined array for the biomarker
    c_arr = np.array(df[biomarker])
    
    # divide the data by gender
    fe_males = df[df.PTGENDER == 'Male']
    fe_females = df[df.PTGENDER == 'Female']

    # get counts of the number of males and females
    num_males = df.PTGENDER.value_counts()['Male']
    
    # calculate the observed mean difference
    obs_mean_diff = np.mean(fe_males[biomarker]) - np.mean(fe_females[biomarker])
    
    # initialize empty numpy array
    perm_mean_diffs = np.empty(size)
    
    # run the permutations calculating means each time
    for i in range(size):
        r_arr = np.random.permutation(c_arr)
        null_arr1 = r_arr[:num_males]
        null_arr2 = r_arr[num_males:]
        perm_mean_diffs[i] = np.mean(null_arr1) - np.mean(null_arr2)
    
    # uncomment to quickly view the distribution
    _ = plt.hist(perm_mean_diffs, density=True, color='blue', 
                 label='Permutation Mean Diffs\nUnder the Null Hypothesis')
    _ = plt.axvline(obs_mean_diff, color='C1', label='Observed Mean Difference')
    _ = plt.title('Probability Distribution for Mean Differences\nBetween Genders for ' + biomarker)
    _ = plt.xlabel('Mean Difference Between Males/Females')
    _ = plt.ylabel('Probability Density')
    _ = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    
    # calculate and display p value
    if obs_mean_diff > np.mean(perm_mean_diffs):
        p = np.sum(perm_mean_diffs >= obs_mean_diff) / len(perm_mean_diffs)
    else:
        p = np.sum(perm_mean_diffs <= obs_mean_diff) / len(perm_mean_diffs)
    print('Distribution Test for Males/Females')
    print('Variable: ', biomarker)
    print('If p < 0.05, then split the data by gender')
    print('p-value: ', p)
    
def bs(df, biomarker, size):
    """This function generates and plots a bootstrap distribution.
    
    Supply the dataframe, biomarker, and number of samples to take for each distribution.
    This function returns the 95% confidence interval for the mean and plots the distribution.
    """
    
    # create the bootstrap distribution
    bs_df = pd.DataFrame({biomarker: [np.mean(np.random.choice(df[biomarker], 
                                                                size=len(df))) for i in range(size)]})
    
    # calculate and display the 95% confidence interval
    lower = bs_df[biomarker].quantile(0.025)
    upper = bs_df[biomarker].quantile(0.975)
    print('95% Confidence Interval: ', lower, ' to ', upper)
    
    # create and display histogram of the bootstrap distribution
    _ = bs_df[biomarker].hist(histtype='step')
    _ = plt.axvline(lower, color='C1', linewidth=1)
    _ = plt.axvline(upper, color='C1', linewidth=1)
    _ = plt.title('Bootstrap Estimate around the Mean for ' + biomarker + '\nNo Diagnosis Change')
    _ = plt.xlabel('Resampled Mean ' + biomarker)
    _ = plt.ylabel('Frequency')
    
    if abs(upper) > abs(lower):
        return upper
    else:
        return lower
    
def old_eval_bs(fe, biomarker, conf, gender='both'):
    """Old function. Code unsuccint. Created new version following DRY.
    
    Saving this old code because never delete anything that works.
    Calculate percentages of patients with a change in diagnosis that had
    a change in the biomarker larger than the threshold value identified from
    bootstrap analysis. You must supply the full final_exam dataframe, the biomarker 
    of interest, the confidence level to evaluate, and provide optional gender of male/female.
    """
    
    # isolate patients who progressed from 'CN' to 'AD'
    cn_mci = fe[(fe['DX'] == 'MCI') & (fe['DX_bl2'] == 'CN')]
    
    # isolate patients who progressed from 'MCI' to 'AD'
    mci_ad = fe[(fe['DX'] == 'AD') & (fe['DX_bl2'] == 'MCI')]
    
    # isolate patients who progressed from 'CN' to 'AD'
    cn_ad = fe[(fe['DX'] == 'AD') & (fe['DX_bl2'] == 'CN')]
    
    if gender == 'both':
        if conf > 0:
            end_CN = len(fe[(fe['DX'] == 'CN') & (fe[biomarker] > conf)]) / len(fe[fe[biomarker] > conf])
            end_MCI = len(fe[(fe['DX'] == 'MCI') & (fe[biomarker] > conf)]) / len(fe[fe[biomarker] > conf])
            end_AD = len(fe[(fe['DX'] == 'AD') & (fe[biomarker] > conf)]) / len(fe[fe[biomarker] > conf])     
            prog_CN_MCI = len(cn_mci[cn_mci[biomarker] > conf]) / len(cn_mci)
            prog_MCI_AD = len(mci_ad[mci_ad[biomarker] > conf]) / len(mci_ad)
            prog_CN_AD = len(cn_ad[cn_ad[biomarker] > conf]) / len(cn_ad)
        else:
            end_CN = len(fe[(fe['DX'] == 'CN') & (fe[biomarker] < conf)]) / len(fe[fe[biomarker] < conf])
            end_MCI = len(fe[(fe['DX'] == 'MCI') & (fe[biomarker] < conf)]) / len(fe[fe[biomarker] < conf])
            end_AD = len(fe[(fe['DX'] == 'AD') & (fe[biomarker] < conf)]) / len(fe[fe[biomarker] < conf])
            prog_CN_MCI = len(cn_mci[cn_mci[biomarker] < conf]) / len(cn_mci)
            prog_MCI_AD = len(mci_ad[mci_ad[biomarker] < conf]) / len(mci_ad)
            prog_CN_AD = len(cn_ad[cn_ad[biomarker] < conf]) / len(cn_ad)
    elif gender == 'males':
        m_cn_mci = cn_mci[cn_mci.PTGENDER == 'Male']
        m_mci_ad = mci_ad[mci_ad.PTGENDER == 'Male']
        m_cn_ad = cn_ad[cn_ad.PTGENDER == 'Male']
        m = fe[fe['PTGENDER'] == 'Male']
        if conf > 0:
            end_CN = len(m[(m['DX'] == 'CN') & (m[biomarker] > conf)]) / len(m[m[biomarker] > conf])
            end_MCI = len(m[(m['DX'] == 'MCI') & (m[biomarker] > conf)]) / len(m[m[biomarker] > conf])
            end_AD = len(m[(m['DX'] == 'AD') & (m[biomarker] > conf)]) / len(m[m[biomarker] > conf])     
            prog_CN_MCI = len(m_cn_mci[m_cn_mci[biomarker] > conf]) / len(m_cn_mci)
            prog_MCI_AD = len(m_mci_ad[m_mci_ad[biomarker] > conf]) / len(m_mci_ad)
            prog_CN_AD = len(m_cn_ad[m_cn_ad[biomarker] > conf]) / len(m_cn_ad)
        else:
            end_CN = len(m[(m['DX'] == 'CN') & (m[biomarker] < conf)]) / len(m[m[biomarker] < conf])
            end_MCI = len(m[(m['DX'] == 'MCI') & (m[biomarker] < conf)]) / len(m[m[biomarker] < conf])
            end_AD = len(m[(m['DX'] == 'AD') & (m[biomarker] < conf)]) / len(m[m[biomarker] < conf]) 
            prog_CN_MCI = len(m_cn_mci[m_cn_mci[biomarker] < conf]) / len(m_cn_mci)
            prog_MCI_AD = len(m_mci_ad[m_mci_ad[biomarker] < conf]) / len(m_mci_ad)
            prog_CN_AD = len(m_cn_ad[m_cn_ad[biomarker] < conf]) / len(m_cn_ad)
    else:
        f_cn_mci = cn_mci[cn_mci.PTGENDER == 'Female']
        f_mci_ad = mci_ad[mci_ad.PTGENDER == 'Female']
        f_cn_ad = cn_ad[cn_ad.PTGENDER == 'Female']
        f = fe[fe['PTGENDER'] == 'Female']
        if conf > 0:
            end_CN = len(f[(f['DX'] == 'CN') & (f[biomarker] > conf)]) / len(f[f[biomarker] > conf])
            end_MCI = len(f[(f['DX'] == 'MCI') & (f[biomarker] > conf)]) / len(f[f[biomarker] > conf])
            end_AD = len(f[(f['DX'] == 'AD') & (f[biomarker] > conf)]) / len(f[f[biomarker] > conf])
            prog_CN_MCI = len(f_cn_mci[f_cn_mci[biomarker] > conf]) / len(f_cn_mci)
            prog_MCI_AD = len(f_mci_ad[f_mci_ad[biomarker] > conf]) / len(f_mci_ad)
            prog_CN_AD = len(f_cn_ad[f_cn_ad[biomarker] > conf]) / len(f_cn_ad)
        else:
            end_CN = len(f[(f['DX'] == 'CN') & (f[biomarker] < conf)]) / len(f[f[biomarker] < conf])
            end_MCI = len(f[(f['DX'] == 'MCI') & (f[biomarker] < conf)]) / len(f[f[biomarker] < conf])
            end_AD = len(f[(f['DX'] == 'AD') & (f[biomarker] < conf)]) / len(f[f[biomarker] < conf])
            prog_CN_MCI = len(f_cn_mci[f_cn_mci[biomarker] < conf]) / len(f_cn_mci)
            prog_MCI_AD = len(f_mci_ad[f_mci_ad[biomarker] < conf]) / len(f_mci_ad)
            prog_CN_AD = len(f_cn_ad[f_cn_ad[biomarker] < conf]) / len(f_cn_ad)

    # print results
    print('Percent exceeding threshold that ended CN: ', round(end_CN*100,2), '%')
    print('Percent exceeding threshold that ended MCI: ', round(end_MCI*100,2), '%')
    print('Percent exceeding threshold that ended AD: ', round(end_AD*100,2), '%')
    print('Percent progressing CN to MCI exceeding threshold: ', round(prog_CN_MCI*100,2), '%')
    print('Percent Progressing MCI to AD exceeding threshold: ', round(prog_MCI_AD*100,2), '%')
    print('Percent Progressing CN to AD exceeding threshold: ', round(prog_CN_AD*100,2), '%')
    
def eval_bs(fe, biomarker, conf, res, cols, gender='both'):
    """Calculate percentages of patients with a change in diagnosis that had
    
    a change in the biomarker larger than the threshold value identified from
    bootstrap analysis. You must supply the full final_exam dataframe, the biomarker 
    of interest, the confidence level to evaluate, and provide optional gender of male/female.
    Also provide a results dataframe 'res' to store the results and its columns list 'cols'.
    """
    
    # isolate the patients who did not experience a change in diagnosis
    nc = fe[fe['DX'] == fe['DX_bl2']]
    
    # isolate patients who progressed from 'CN' to 'AD'
    cn_mci = fe[(fe['DX'] == 'MCI') & (fe['DX_bl2'] == 'CN')]
    
    # isolate patients who progressed from 'MCI' to 'AD'
    mci_ad = fe[(fe['DX'] == 'AD') & (fe['DX_bl2'] == 'MCI')]
    
    # isolate patients who progressed from 'CN' to 'AD'
    cn_ad = fe[(fe['DX'] == 'AD') & (fe['DX_bl2'] == 'CN')]
    
    if gender == 'both':
        df = fe
        df1 = nc
        df2 = cn_mci
        df3 = mci_ad
        df4 = cn_ad
    elif gender == 'males':
        df = fe[fe['PTGENDER'] == 'Male']
        df1 = nc[nc.PTGENDER == 'Male']
        df2 = cn_mci[cn_mci.PTGENDER == 'Male']
        df3 = mci_ad[mci_ad.PTGENDER == 'Male']
        df4 = cn_ad[cn_ad.PTGENDER == 'Male']
    else:
        df = fe[fe['PTGENDER'] == 'Female']
        df1 = nc[nc.PTGENDER == 'Female']
        df2 = cn_mci[cn_mci.PTGENDER == 'Female']
        df3 = mci_ad[mci_ad.PTGENDER == 'Female']
        df4 = cn_ad[cn_ad.PTGENDER == 'Female']
        
    # use correct comparison depending on biomarker increase vs. decrease    
    if conf > 0:
        end_CN = len(df[(df['DX'] == 'CN') & (df[biomarker] > conf)]) / len(df[df[biomarker] > conf])
        end_MCI = len(df[(df['DX'] == 'MCI') & (df[biomarker] > conf)]) / len(df[df[biomarker] > conf])
        end_AD = len(df[(df['DX'] == 'AD') & (df[biomarker] > conf)]) / len(df[df[biomarker] > conf])     
        no_prog = round((len(df1[df1[biomarker] > conf]) / len(df1))*100,2)
        prog_CN_MCI = round((len(df2[df2[biomarker] > conf]) / len(df2))*100,2)
        prog_MCI_AD = round((len(df3[df3[biomarker] > conf]) / len(df3))*100,2)
        prog_CN_AD = round((len(df4[df4[biomarker] > conf]) / len(df4))*100,2)
    else:
        end_CN = len(df[(df['DX'] == 'CN') & (df[biomarker] < conf)]) / len(df[df[biomarker] < conf])
        end_MCI = len(df[(df['DX'] == 'MCI') & (df[biomarker] < conf)]) / len(df[df[biomarker] < conf])
        end_AD = len(df[(df['DX'] == 'AD') & (df[biomarker] < conf)]) / len(df[df[biomarker] < conf])     
        no_prog = round((len(df1[df1[biomarker] < conf]) / len(df1))*100,2)
        prog_CN_MCI = round((len(df2[df2[biomarker] < conf]) / len(df2))*100,2)
        prog_MCI_AD = round((len(df3[df3[biomarker] < conf]) / len(df3))*100,2)
        prog_CN_AD = round((len(df4[df4[biomarker] < conf]) / len(df4))*100,2)

    # print results
    print('Threshold: ', conf)
    print('Percent exceeding threshold that ended CN: ', round(end_CN*100,2), '%')
    print('Percent exceeding threshold that ended MCI: ', round(end_MCI*100,2), '%')
    print('Percent exceeding threshold that ended AD: ', round(end_AD*100,2), '%')
    print('Percent with no diagnosis change exceeding threshold: ', no_prog, '%')
    print('Percent progressing CN to MCI exceeding threshold: ', prog_CN_MCI, '%')
    print('Percent Progressing MCI to AD exceeding threshold: ', prog_MCI_AD, '%')
    print('Percent Progressing CN to AD exceeding threshold: ', prog_CN_AD, '%')
    
    if gender == 'males':
        biomarker = str(biomarker) + '_m'
    elif gender == 'females':
        biomarker = str(biomarker) + '_f'
        
    temp = pd.DataFrame([[biomarker, conf, round(end_CN*100,2), 'Ended CN']], columns=cols)
    temp = temp.append(pd.DataFrame([[biomarker, conf, no_prog, 'No DX Change']], columns=cols))
    temp = temp.append(pd.DataFrame([[biomarker, conf, prog_CN_MCI, 'CN to MCI']], columns=cols))
    temp = temp.append(pd.DataFrame([[biomarker, conf, prog_MCI_AD, 'MCI to AD']], columns=cols))
    temp = temp.append(pd.DataFrame([[biomarker, conf, prog_CN_AD, 'CN to AD']], columns=cols))
    res = res.append(temp, ignore_index=True)
    return res

def summarize_clin_changes(changes):
    """Create plots to summarize the changes in biomarkers."""
    
    # separate the data for creating different plots
    clins = ['CDRSB_delta', 'ADAS11_delta_m', 'ADAS11_delta_f', 'ADAS13_delta_m', 
            'ADAS13_delta_f', 'MMSE_delta', 'RAVLT_delta']
    filt = changes.biomarker.isin(clins)
    c_clin = changes[filt]
    
    #exclude false positives
    fp = ['CN to MCI', 'MCI to AD', 'CN to AD']
    filt2 = c_clin.group.isin(fp)
    changes_clin = c_clin[filt2]
    
    # set the subplot
    #plt.rcParams["figure.figsize"] = (14,4)
    #plt.subplot(1, 2, 1)
    
    g = sns.catplot(x='biomarker', y='pct', hue='group', data=changes_clin, height=4.5, aspect=1.5,
                    kind='bar', palette='muted', legend=False)
    _ = g.despine(left=True)
    _ = g.set_xticklabels(rotation=60)
    _ = g.set_ylabels('Percent of Patients')
    _ = g.set_xlabels('Biomarker')
    _ = plt.title('Change Detection Rates for Clinical Biomarkers')
    _ = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    
def summarize_scan_changes(changes):    
    scans = ['Hippocampus_delta', 'Ventricles_delta_m', 'Ventricles_delta_f', 'WholeBrain_delta', 
            'Entorhinal_delta', 'MidTemp_delta_m', 'MidTemp_delta_f', ]
    filt = changes.biomarker.isin(scans)
    c_scan = changes[filt]
    
    #exclude false positives
    fp = ['CN to MCI', 'MCI to AD', 'CN to AD']
    filt2 = c_scan.group.isin(fp)
    changes_scan = c_scan[filt2]
    
    #plt.subplot(1, 2, 2)
    g = sns.catplot(x='biomarker', y='pct', hue='group', data=changes_scan, kind='bar', 
                    height=4.5, aspect=1.5, palette='muted', legend=False)
    _ = g.despine(left=True)
    _ = g.set_xticklabels(rotation=60)
    _ = g.set_ylabels('Percent of Patients')
    _ = g.set_xlabels('Biomarker')
    _ = plt.title('Change Detection Rates for Brain Scan Biomarkers')
    _ = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    
def summarize_clin_fps(changes):
    """Create plots to summarize the changes in biomarkers."""
    
    # separate the data for creating different plots
    clins = ['CDRSB_delta', 'ADAS11_delta_m', 'ADAS11_delta_f', 'ADAS13_delta_m', 
            'ADAS13_delta_f', 'MMSE_delta', 'RAVLT_delta']
    filt = changes.biomarker.isin(clins)
    c_clin = changes[filt]
    
    #include only false positives
    fp = ['Ended CN', 'No DX Change']
    filt2 = c_clin.group.isin(fp)
    changes_clin = c_clin[filt2]
    
    # set the subplot
    #plt.rcParams["figure.figsize"] = (14,4)
    #plt.subplot(1, 2, 1)
    
    g = sns.catplot(x='biomarker', y='pct', hue='group', data=changes_clin, height=4.5, aspect=1.5,
                    kind='bar', legend=False)
    _ = g.despine(left=True)
    _ = g.set_xticklabels(rotation=60)
    _ = g.set_ylabels('Percent of Patients')
    _ = g.set_xlabels('Biomarker')
    _ = plt.title('False Positive Rates for Clinical Biomarker Thresholds')
    _ = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    
def summarize_scan_fps(changes):    
    scans = ['Hippocampus_delta', 'Ventricles_delta_m', 'Ventricles_delta_f', 'WholeBrain_delta', 
            'Entorhinal_delta', 'MidTemp_delta_m', 'MidTemp_delta_f', ]
    filt = changes.biomarker.isin(scans)
    c_scan = changes[filt]
        
    #exclude false positives
    fp = ['Ended CN', 'No DX Change']
    filt2 = c_scan.group.isin(fp)
    changes_scan = c_scan[filt2]
    
    #plt.subplot(1, 2, 2)
    g = sns.catplot(x='biomarker', y='pct', hue='group', data=changes_scan, kind='bar', 
                    height=4.5, aspect=1.5, legend=False)
    _ = g.despine(left=True)
    _ = g.set_xticklabels(rotation=60)
    _ = g.set_ylabels('Percent of Patients')
    _ = g.set_xlabels('Biomarker')
    _ = plt.title('False Positive Rates for Brain Scan Biomarkers')
    _ = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        
def bl_perm_test(fe, biomarker, gender, size):
    """This function returns the p value for the test that patients that ended AD have the 
    
    same distribution as those that didn't in the supplied biomarker's baseline values. 
    The test will be performed over 'size' permutations. A significant p value means that patients
    that developed AD have a different distribution for their baseline values than those that didn't.
    """
    
    # divide data and use supplied gender
    if gender == 'males':
        df = fe[fe.PTGENDER == 'Male']
    else:
        df = fe[fe.PTGENDER == 'Female']
    
    # create a combined array for the biomarker
    c_arr = np.array(df[biomarker])
    
    # divide the data by final diagnosis
    ad = df[df.DX == 'AD']
    non_ad = df[df.DX != 'AD']

    # get counts of the number of patients that developed AD
    num_ad = df.DX.value_counts()['AD']
    
    # calculate the observed mean difference
    obs_mean_diff = np.mean(ad[biomarker]) - np.mean(non_ad[biomarker])
    
    # initialize empty numpy array
    perm_mean_diffs = np.empty(size)
    
    # run the permutations calculating means each time
    for i in range(size):
        r_arr = np.random.permutation(c_arr)
        null_arr1 = r_arr[:num_ad]
        null_arr2 = r_arr[num_ad:]
        perm_mean_diffs[i] = np.mean(null_arr1) - np.mean(null_arr2)
    
    # uncomment to quickly view the distribution
    _ = plt.hist(perm_mean_diffs, density=True, color='blue', 
                 label='Permutation Mean Diffs\nUnder the Null Hypothesis')
    _ = plt.axvline(obs_mean_diff, color='C1', label='Observed Difference')
    _ = plt.title('Probability Distribution for Mean Differences\nBetween AD/Non AD for ' + biomarker)
    _ = plt.xlabel('Mean Difference Between AD/Non AD')
    _ = plt.ylabel('Probability Density')
    _ = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    
    # calculate and display p value
    if obs_mean_diff > np.mean(perm_mean_diffs):
        p = np.sum(perm_mean_diffs >= obs_mean_diff) / len(perm_mean_diffs)
    else:
        p = np.sum(perm_mean_diffs <= obs_mean_diff) / len(perm_mean_diffs)
    print('Distribution Test for AD/Non AD')
    print('Variable: ', biomarker)
    print('If p < 0.05, then patients that ended AD had a different distribution for ', biomarker)
    print('p-value: ', p)
    
def bs_bl(fe, biomarker, size, gender):
    """This function generates and plots a bootstrap distribution.
    
    Supply the dataframe, biomarker, and number of samples to take for each distribution.
    This function returns the 95% confidence interval and plots the distribution.
    """
    
    # divide data and use supplied gender
    if gender == 'males':
        df = fe[fe.PTGENDER == 'Male']
    else:
        df = fe[fe.PTGENDER == 'Female']
        
    # divide the data by final diagnosis
    ad = df[df.DX == 'AD']
    non_ad = df[df.DX != 'AD']
    
    # create the bootstrap distribution for AD
    bs_ad = pd.DataFrame({biomarker: [np.mean(np.random.choice(ad[biomarker], 
                                                                size=len(ad))) for i in range(size)]})
    
    
    # create the bootstrap distribution for non AD
    bs_non = pd.DataFrame({biomarker: [np.mean(np.random.choice(non_ad[biomarker], 
                                                                size=len(non_ad))) for i in range(size)]})
    
    # calculate and display the 95% confidence intervals
    ad_lower = bs_ad[biomarker].quantile(0.025)
    ad_upper = bs_ad[biomarker].quantile(0.975)
    non_lower = bs_non[biomarker].quantile(0.025)
    non_upper = bs_non[biomarker].quantile(0.975)
    print('95% Confidence Interval for AD: ', ad_lower, ' to ', ad_upper)
    print('95% Confidence Interval for non AD: ', non_lower, ' to ', non_upper)
    
    # create and display histogram of the bootstrap distribution
    _ = bs_ad[biomarker].hist(histtype='stepfilled', color='blue', alpha=0.3, label='AD')
    _ = bs_non[biomarker].hist(histtype='stepfilled', color='orange', alpha=0.3, label='Non AD')
    #_ = plt.axvline(lower, color='C1', linewidth=1)
    #_ = plt.axvline(upper, color='C1', linewidth=1)
    _ = plt.axvline(bs_non[biomarker].quantile(0.75), color='C1', linewidth=1)
    _ = plt.axvline(bs_non[biomarker].quantile(0.95), color='red', linewidth=1)
    _ = plt.title('Bootstrap Means for ' + biomarker)
    _ = plt.xlabel('Resampled Means for ' + biomarker)
    _ = plt.ylabel('Frequency')
    _ = plt.legend(loc='best')
    
    # display further results     
    if abs(non_upper) > abs(non_lower):
        #return upper
        #print('25th percentile for AD bootstrap distribution: ', bs_ad[biomarker].quantile(0.25))
        #print('50th percentile for AD bootstrap distribution (mean): ', bs_ad[biomarker].quantile(0.5))
        #print('75th percentile for AD bootstrap distribution: ', bs_ad[biomarker].quantile(0.75))
        #print('95th percentile for AD bootstrap distribution: ', bs_ad[biomarker].quantile(0.95))
        
        #print('25th percentile for non AD bootstrap distribution: ', bs_non[biomarker].quantile(0.25))
        print('Mean for non AD bootstrap distribution: ', bs_non[biomarker].quantile(0.5))
        print('75th percentile for non AD bootstrap distribution: ', bs_non[biomarker].quantile(0.75))
        print('95th percentile for non AD bootstrap distribution: ', bs_non[biomarker].quantile(0.95))
        
        return bs_non[biomarker].quantile(0.75), bs_non[biomarker].quantile(0.95)
    
    else:
        #return lower
        #print('25th percentile for AD bootstrap distribution: ', bs_ad[biomarker].quantile(0.75))
        #print('50th percentile for AD bootstrap distribution (mean): ', bs_ad[biomarker].quantile(0.5))
        #print('75th percentile for AD bootstrap distribution: ', bs_ad[biomarker].quantile(0.25))
        #print('95th percentile for AD bootstrap distribution: ', bs_ad[biomarker].quantile(0.05))
        
        #print('25th percentile for non AD bootstrap distribution: ', bs_non[biomarker].quantile(0.75))
        print('Mean for non AD bootstrap distribution: ', bs_non[biomarker].quantile(0.5))
        print('75th percentile for non AD bootstrap distribution: ', bs_non[biomarker].quantile(0.25))
        print('95th percentile for non AD bootstrap distribution: ', bs_non[biomarker].quantile(0.05))
        
        return bs_non[biomarker].quantile(0.25), bs_non[biomarker].quantile(0.05)
        
def eval_bl(fe, biomarker, conf_75, conf_95, gender):
    """Calculate percentages of patients exceeding a baseline value that ended the study
    
    with AD vs. the percentage that didn't end the study with AD.
    """
    
    # divide the data by final diagnosis and gender
    if gender == 'males':
        ad = fe[(fe.DX == 'AD') & (fe.PTGENDER == 'Male')]
        non_ad = fe[(fe.DX != 'AD') & (fe.PTGENDER == 'Male')]
    else:
        ad = fe[(fe.DX == 'AD') & (fe.PTGENDER == 'Female')]
        non_ad = fe[(fe.DX != 'AD') & (fe.PTGENDER == 'Female')]
            
    # use correct comparison depending on biomarker increase vs. decrease    
    if conf_75 > 0:
        pct_non_75 = len(non_ad[non_ad[biomarker] >= conf_75]) / len(non_ad)
        pct_non_95 = len(non_ad[non_ad[biomarker] >= conf_95]) / len(non_ad)
        pct_ad_75 = len(ad[ad[biomarker] <= conf_75]) / len(ad)
        pct_ad_95 = len(ad[ad[biomarker] <= conf_95]) / len(ad)
        
    else:
        pct_non_75 = len(non_ad[non_ad[biomarker] <= conf_75]) / len(non_ad)
        pct_non_95 = len(non_ad[non_ad[biomarker] <= conf_95]) / len(non_ad)
        pct_ad_75 = len(ad[ad[biomarker] >= conf_75]) / len(ad)
        pct_ad_95 = len(ad[ad[biomarker] >= conf_95]) / len(ad)

    # print results
    print('Percent of patients without AD diagnosis exceeding lower bootstrap threshold: ',
          round(pct_non_75*100,2), '%')
    print('Percent of patients without AD diagnosis exceeding the higher bootstrap threshold: ', 
          round(pct_non_95*100,2), '%')
    print('Percent of AD patients below the lower bootstrap threshold: ', round(pct_ad_75*100,2), '%')
    print('Percent of AD patients below the higher bootstrap threshold: ', round(pct_ad_95*100,2), '%')
    
def bs_percentile(fe, biomarker, size, gender):
    """This function generates and plots a bootstrap distribution.
    
    Supply the dataframe, biomarker, and number of samples to take for each distribution.
    This function returns the 95% confidence interval and plots the distribution.
    """
    
    # divide data and use supplied gender
    if gender == 'males':
        df = fe[fe.PTGENDER == 'Male']
    else:
        df = fe[fe.PTGENDER == 'Female']
        
    # divide the data by final diagnosis
    ad = df[df.DX == 'AD']
    non_ad = df[df.DX != 'AD']
    
    if np.mean(ad[biomarker]) > np.mean(non_ad[biomarker]):            #min(ad[biomarker]) >= 0:
        # create the bootstrap distribution for AD
        bs_ad_25 = pd.DataFrame({biomarker: [np.percentile(np.random.choice(ad[biomarker], 
                                                                    size=len(ad)),25) for i in range(size)]})
    
        bs_ad_5 = pd.DataFrame({biomarker: [np.percentile(np.random.choice(ad[biomarker], 
                                                                    size=len(ad)),5) for i in range(size)]})
        
        bs_non_75 = pd.DataFrame({biomarker: [np.percentile(np.random.choice(non_ad[biomarker], 
                                                                    size=len(non_ad)),75) for i in range(size)]})
    
        bs_non_95 = pd.DataFrame({biomarker: [np.percentile(np.random.choice(non_ad[biomarker], 
                                                                    size=len(non_ad)),95) for i in range(size)]})
        
    else:
        # create the bootstrap distribution for AD
        bs_ad_25 = pd.DataFrame({biomarker: [np.percentile(np.random.choice(ad[biomarker], 
                                                                    size=len(ad)),75) for i in range(size)]})
    
        bs_ad_5 = pd.DataFrame({biomarker: [np.percentile(np.random.choice(ad[biomarker], 
                                                                    size=len(ad)),95) for i in range(size)]})
        
        bs_non_75 = pd.DataFrame({biomarker: [np.percentile(np.random.choice(non_ad[biomarker], 
                                                                    size=len(non_ad)),25) for i in range(size)]})
    
        bs_non_95 = pd.DataFrame({biomarker: [np.percentile(np.random.choice(non_ad[biomarker], 
                                                                    size=len(non_ad)),5) for i in range(size)]})
    
    
    # calculate and display the 95% confidence intervals
    #ad_lower = bs_ad[biomarker].quantile(0.025)
    #ad_upper = bs_ad[biomarker].quantile(0.975)
    #non_lower = bs_non[biomarker].quantile(0.025)
    #non_upper = bs_non[biomarker].quantile(0.975)
    #print('95% Confidence Interval for AD: ', ad_lower, ' to ', ad_upper)
    #print('95% Confidence Interval for non AD: ', non_lower, ' to ', non_upper)
    
    # create and display histogram of the bootstrap distribution
    _ = bs_ad_25[biomarker].hist(histtype='stepfilled', color='orange', alpha=0.3, label='AD_25')
    _ = bs_ad_5[biomarker].hist(histtype='stepfilled', color='red', alpha=0.3, label='AD_5')
    _ = bs_non_75[biomarker].hist(histtype='stepfilled', color='green', alpha=0.3, label='Non_75')
    _ = bs_non_95[biomarker].hist(histtype='stepfilled', color='blue', alpha=0.3, label='Non_95')
    
    #_ = plt.axvline(lower, color='C1', linewidth=1)
    #_ = plt.axvline(upper, color='C1', linewidth=1)
    _ = plt.axvline(np.mean(bs_non_75[biomarker]), color='C1', linewidth=1, label='Non AD 75th Pctle')
    _ = plt.axvline(np.mean(bs_ad_25[biomarker]), color='red', linewidth=1, label='AD 25th Pctle')
    if gender == 'males':
        _ = plt.title('Bootstrap Threshold Values for Males ' + biomarker)
    else:
        _ = plt.title('Bootstrap Threshold Values for Females ' + biomarker)
    _ = plt.xlabel('Resampled 75th and 95th Conf Intvl\'s for ' + biomarker)
    _ = plt.ylabel('Frequency')
    _ = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    
    # display further results     
    print('25% false positive threshold value: ', np.mean(bs_non_75[biomarker]))
    #print('Mean 95th percentile of bootstrap samples for non AD patients: ', np.mean(bs_non_95[biomarker]))
    print('75% detection threshold value: ', np.mean(bs_ad_25[biomarker]))
    #print('Mean 5th percentile of bootstrap samples for non AD patients: ', np.mean(bs_ad_5[biomarker]))
    
    return bs_non_75, bs_ad_25
    
def get_pctles(bs_non_75, bs_ad_25, fe, biomarker, gender, increase=True):
    """This function will take the percentile from one distribution and calculate
    
    The corresponding percentage associated with the other distribution. You must
    supply the biomarker and the two bootstrap distributions returned from the 
    bs_percentiles() function. The default assumes 'increase=True', meaning the 
    biomarker will increase as someone progresses to Alzheimer's. Set to false
    for biomarkers that decrease with progression to AD (like Hippocampus/MidTemp).
    """
    
    if gender == 'males':
        ad = fe[(fe.DX == 'AD') & (fe.PTGENDER == 'Male')]
        non = fe[(fe.DX != 'AD') & (fe.PTGENDER == 'Male')]
    else:
        ad = fe[(fe.DX == 'AD') & (fe.PTGENDER == 'Female')]
        non = fe[(fe.DX != 'AD') & (fe.PTGENDER == 'Female')]
        
    bs_ad = pd.DataFrame({'percentiles': 
                         [scipy.stats.percentileofscore(ad.sample(len(ad),replace=True)[biomarker],
                                                        np.mean(bs_non_75).values) for i in range(10000)]})
    
    bs_non = pd.DataFrame({'percentiles': 
                         [scipy.stats.percentileofscore(non.sample(len(non),replace=True)[biomarker],
                                                        np.mean(bs_ad_25).values) for i in range(10000)]})
    
    ad_score = round(np.mean(bs_ad.percentiles),2)
    non_score = round(np.mean(bs_non.percentiles),2)
    
    if increase:
        print('The detection rate for AD with 25% false positive is', round(100-ad_score,2), '%')
        print('The false positive rate for 75% AD detection is', round(100-non_score,2), '%')
        return round(100-ad_score,2), round(100-non_score,2)
    else:    
        print('The detection rate for AD with 25% false positive is', ad_score, '%')
        print('The false positive rate for 75% AD detection is', non_score, '%')
        return ad_score, non_score
          
def rebuild_changes_df():
    """This function is designed to rebuild the changes dataframe.
    
    The changes dataframe was generated one piece at a time while running other functions.
    This function recreates that dataframe using values from one version of that dataframe.
    """
    
    # repopulate list of biomarkers
    biomarker = ['CDRSB_delta', 'CDRSB_delta', 'CDRSB_delta', 'CDRSB_delta',
       'CDRSB_delta', 'ADAS11_delta_m', 'ADAS11_delta_m',
       'ADAS11_delta_m', 'ADAS11_delta_m', 'ADAS11_delta_m',
       'ADAS11_delta_f', 'ADAS11_delta_f', 'ADAS11_delta_f',
       'ADAS11_delta_f', 'ADAS11_delta_f', 'ADAS13_delta_m',
       'ADAS13_delta_m', 'ADAS13_delta_m', 'ADAS13_delta_m',
       'ADAS13_delta_m', 'ADAS13_delta_f', 'ADAS13_delta_f',
       'ADAS13_delta_f', 'ADAS13_delta_f', 'ADAS13_delta_f', 'MMSE_delta',
       'MMSE_delta', 'MMSE_delta', 'MMSE_delta', 'MMSE_delta',
       'RAVLT_delta', 'RAVLT_delta', 'RAVLT_delta', 'RAVLT_delta',
       'RAVLT_delta', 'Hippocampus_delta', 'Hippocampus_delta',
       'Hippocampus_delta', 'Hippocampus_delta', 'Hippocampus_delta',
       'Ventricles_delta_m', 'Ventricles_delta_m', 'Ventricles_delta_m',
       'Ventricles_delta_m', 'Ventricles_delta_m', 'Ventricles_delta_f',
       'Ventricles_delta_f', 'Ventricles_delta_f', 'Ventricles_delta_f',
       'Ventricles_delta_f', 'WholeBrain_delta', 'WholeBrain_delta',
       'WholeBrain_delta', 'WholeBrain_delta', 'WholeBrain_delta',
       'Entorhinal_delta', 'Entorhinal_delta', 'Entorhinal_delta',
       'Entorhinal_delta', 'Entorhinal_delta', 'MidTemp_delta_m',
       'MidTemp_delta_m', 'MidTemp_delta_m', 'MidTemp_delta_m',
       'MidTemp_delta_m', 'MidTemp_delta_f', 'MidTemp_delta_f',
       'MidTemp_delta_f', 'MidTemp_delta_f', 'MidTemp_delta_f']
    
    # repopulate data for the thresholds
    thresh = [ 6.03711790e-01,  6.03711790e-01,  6.03711790e-01,  6.03711790e-01,
        6.03711790e-01,  1.68623544e+00,  1.68623544e+00,  1.68623544e+00,
        1.68623544e+00,  1.68623544e+00,  2.07187739e+00,  2.07187739e+00,
        2.07187739e+00,  2.07187739e+00,  2.07187739e+00,  2.13385442e+00,
        2.13385442e+00,  2.13385442e+00,  2.13385442e+00,  2.13385442e+00,
        2.40918660e+00,  2.40918660e+00,  2.40918660e+00,  2.40918660e+00,
        2.40918660e+00, -1.00545852e+00, -1.00545852e+00, -1.00545852e+00,
       -1.00545852e+00, -1.00545852e+00, -1.93340611e+00, -1.93340611e+00,
       -1.93340611e+00, -1.93340611e+00, -1.93340611e+00, -2.81745497e+02,
       -2.81745497e+02, -2.81745497e+02, -2.81745497e+02, -2.81745497e+02,
        5.76426576e+03,  5.76426576e+03,  5.76426576e+03,  5.76426576e+03,
        5.76426576e+03,  4.68402189e+03,  4.68402189e+03,  4.68402189e+03,
        4.68402189e+03,  4.68402189e+03, -2.12204483e+04, -2.12204483e+04,
       -2.12204483e+04, -2.12204483e+04, -2.12204483e+04, -1.76039083e+02,
       -1.76039083e+02, -1.76039083e+02, -1.76039083e+02, -1.76039083e+02,
       -6.84145080e+02, -6.84145080e+02, -6.84145080e+02, -6.84145080e+02,
       -6.84145080e+02, -7.72849641e+02, -7.72849641e+02, -7.72849641e+02,
       -7.72849641e+02, -7.72849641e+02]
    
    # repopulate the percent data
    pct = [3.63, 24.34,  45.95,  95.45, 100.  ,  14.53,  37.55,  57.14,
        80.58, 100.  ,  14.35,  35.17,  50.  ,  84.93, 100.  ,  12.14,
        36.55,  52.38,  79.61, 100.  ,  19.33,  37.56,  62.5 ,  89.04,
       100.  ,  14.89,  30.79,  27.03,  83.52, 100.  ,  23.44,  48.36,
        59.46,  74.43, 100.  ,  26.22,  43.01,  59.46,  77.27,  80.  ,
        18.79,  36.14,  66.67,  78.64, 100.  ,  23.61,  33.73,  56.25,
        83.56,  66.67,  27.2 ,  38.86,  56.76,  73.3 ,  40.  ,  28.5 ,
        45.96,  40.54,  73.3 ,  40.  ,  22.22,  41.77,  66.67,  72.82,
        50.  ,  26.67,  39.23,  62.5 ,  82.19,  66.67]
    
    # repopulate the group data
    group = ['Ended CN', 'No DX Change', 'CN to MCI', 'MCI to AD', 'CN to AD',
       'Ended CN', 'No DX Change', 'CN to MCI', 'MCI to AD', 'CN to AD',
       'Ended CN', 'No DX Change', 'CN to MCI', 'MCI to AD', 'CN to AD',
       'Ended CN', 'No DX Change', 'CN to MCI', 'MCI to AD', 'CN to AD',
       'Ended CN', 'No DX Change', 'CN to MCI', 'MCI to AD', 'CN to AD',
       'Ended CN', 'No DX Change', 'CN to MCI', 'MCI to AD', 'CN to AD',
       'Ended CN', 'No DX Change', 'CN to MCI', 'MCI to AD', 'CN to AD',
       'Ended CN', 'No DX Change', 'CN to MCI', 'MCI to AD', 'CN to AD',
       'Ended CN', 'No DX Change', 'CN to MCI', 'MCI to AD', 'CN to AD',
       'Ended CN', 'No DX Change', 'CN to MCI', 'MCI to AD', 'CN to AD',
       'Ended CN', 'No DX Change', 'CN to MCI', 'MCI to AD', 'CN to AD',
       'Ended CN', 'No DX Change', 'CN to MCI', 'MCI to AD', 'CN to AD',
       'Ended CN', 'No DX Change', 'CN to MCI', 'MCI to AD', 'CN to AD',
       'Ended CN', 'No DX Change', 'CN to MCI', 'MCI to AD', 'CN to AD']
    
    # rebuild the dataframe
    changes = pd.DataFrame({'biomarker': biomarker, 'thresh': thresh, 'pct': pct, 'group': group})
    
    return changes

def rebuild_bl():
    """This function is designed to rebuild the summary of baseline biomarkers.
    
    This was dataframe was generated from running all of the bs_percentiles and 
    get_pctiles functions. The data are reloaded here so the dataframe can be 
    recreated without having to run every single function.
    """
    
    # populate the biomarker list
    biomarker = ['CDRSB_bl_m', 'CDRSB_bl_f', 'ADAS11_bl_m', 'ADAS11_bl_f',
       'ADAS13_bl_m', 'ADAS13_bl_f', 'MMSE_bl_m', 'MMSE_bl_f',
       'RAVLT_immediate_bl_m', 'RAVLT_immediate_bl_f', 'Hippocampus_bl_m',
       'Hippocampus_bl_f', 'Ventricles_bl_m', 'Ventricles_bl_f',
       'WholeBrain_bl_m', 'WholeBrain_bl_f', 'Entorhinal_bl_m',
       'Entorhinal_bl_f', 'MidTemp_bl_m', 'MidTemp_bl_f', 'CDRSB_bl_m',
       'CDRSB_bl_f', 'ADAS11_bl_m', 'ADAS11_bl_f', 'ADAS13_bl_m',
       'ADAS13_bl_f', 'MMSE_bl_m', 'MMSE_bl_f', 'RAVLT_immediate_bl_m',
       'RAVLT_immediate_bl_f', 'Hippocampus_bl_m', 'Hippocampus_bl_f',
       'Ventricles_bl_m', 'Ventricles_bl_f', 'WholeBrain_bl_m',
       'WholeBrain_bl_f', 'Entorhinal_bl_m', 'Entorhinal_bl_f',
       'MidTemp_bl_m', 'MidTemp_bl_f']
    
    # populate the kind of rate
    rate_kind = ['Detection Rate @25% FP', 'Detection Rate @25% FP',
            'Detection Rate @25% FP', 'Detection Rate @25% FP',
            'Detection Rate @25% FP', 'Detection Rate @25% FP',
            'Detection Rate @25% FP', 'Detection Rate @25% FP',
            'Detection Rate @25% FP', 'Detection Rate @25% FP',
            'Detection Rate @25% FP', 'Detection Rate @25% FP',
            'Detection Rate @25% FP', 'Detection Rate @25% FP',
            'Detection Rate @25% FP', 'Detection Rate @25% FP',
            'Detection Rate @25% FP', 'Detection Rate @25% FP',
            'Detection Rate @25% FP', 'Detection Rate @25% FP',
            'False Positive Rate @75% DR', 'False Positive Rate @75% DR',
            'False Positive Rate @75% DR', 'False Positive Rate @75% DR',
            'False Positive Rate @75% DR', 'False Positive Rate @75% DR',
            'False Positive Rate @75% DR', 'False Positive Rate @75% DR',
            'False Positive Rate @75% DR', 'False Positive Rate @75% DR',
            'False Positive Rate @75% DR', 'False Positive Rate @75% DR',
            'False Positive Rate @75% DR', 'False Positive Rate @75% DR',
            'False Positive Rate @75% DR', 'False Positive Rate @75% DR',
            'False Positive Rate @75% DR', 'False Positive Rate @75% DR',
            'False Positive Rate @75% DR', 'False Positive Rate @75% DR']
    
    # populate the thresholds
    th = [1.47796250e+00, 1.03582500e+00, 1.04565672e+01, 8.77764000e+00,
       1.67372023e+01, 1.37884600e+01, 2.73662500e+01, 2.79999500e+01,
       2.93589500e+01, 3.72926000e+01, 6.67301140e+03, 6.39104055e+03,
       5.07976370e+04, 3.79437399e+04, 1.02558200e+06, 9.28774859e+05,
       3.47441757e+03, 3.15239910e+03, 1.91798401e+04, 1.76171079e+04,
       1.54165000e+00, 1.90445000e+00, 1.13094967e+01, 1.15772740e+01,
       1.91089450e+01, 2.00123600e+01, 2.70373750e+01, 2.61414000e+01,
       2.88948000e+01, 3.22810000e+01, 6.97061595e+03, 6.24035630e+03,
       3.34590551e+04, 2.43875284e+04, 1.12003375e+06, 9.79476420e+05,
       3.65669015e+03, 3.14421010e+03, 2.06908618e+04, 1.80247469e+04]   
    
    # populate the rates
    rate = [83.21, 85.93, 81.62, 90.47, 85.69, 92.96, 79.62, 84.68, 77.54,
       91.1 , 65.8 , 80.27, 45.39, 40.16, 47.51, 56.02, 67.3 , 75.82,
       56.7 , 70.04, 15.44, 11.32, 20.17,  9.97, 14.58,  7.82, 25.77,
        9.98, 22.4 , 13.74, 35.22, 21.3 , 56.07, 52.58, 66.59, 45.83,
       34.28, 24.84, 44.84, 30.45]
    
    # build the dataframe
    bl = pd.DataFrame({'biomarker': biomarker, 'threshold': th, 'rate_kind': rate_kind, 'rate': rate})
    
    # return the dataframe
    return bl

def summarize_clin_bl(bl):
    """This function creates the summary plot for the clinical baseline thresholds."""
    
    # select the cinical exams
    clin = ['CDRSB_bl_m', 'CDRSB_bl_f', 'ADAS11_bl_m', 'ADAS11_bl_f', 'ADAS13_bl_m', 'ADAS13_bl_f' 
            'MMSE_bl_m', 'MMSE_bl_f', 'RAVLT_immediate_bl_m', 'RAVLT_immediate_bl_f']
    filt = bl.biomarker.isin(clin)
    bl_clin = bl[filt]
    
    # build the plot
    g = sns.catplot(x='biomarker', y='rate', hue='rate_kind', data=bl_clin, height=4.5, aspect=1.5,
                    kind='bar', legend=False)
    _ = g.despine(left=True)
    _ = g.set_xticklabels(rotation=60)
    _ = g.set_ylabels('Percent of Patients')
    _ = g.set_xlabels('Biomarker')
    _ = plt.title('Detection and False Positive Rates for Baseline Biomarkers')
    _ = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    
def summarize_scans_bl(bl):
    """This function creates the summary plot for the brain scan baseline thresholds."""
    
    # select the brain scans
    scans = ['Hippocampus_bl_m', 'Hippocampus_bl_f', 'Ventricles_bl_m', 'Ventricles_bl_f', 'WholeBrain_bl_m',
             'WholeBrain_bl_f', 'Entorhinal_bl_m', 'Entorhinal_bl_f', 'MidTemp_bl_m', 'MidTemp_bl_f']
    filt = bl.biomarker.isin(scans)
    bl_scans = bl[filt]
    
    # build the plot
    g = sns.catplot(x='biomarker', y='rate', hue='rate_kind', data=bl_scans, height=4.5, aspect=1.5,
                    kind='bar', legend=False)
    _ = g.despine(left=True)
    _ = g.set_xticklabels(rotation=60)
    _ = g.set_ylabels('Percent of Patients')
    _ = g.set_xlabels('Biomarker')
    _ = plt.title('Detection and False Positive Rates for Baseline Biomarkers')
    _ = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)