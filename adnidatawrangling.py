#!/usr/bin/env python3
"""Data Wrangling Module for Alzheimer's Capstone 1.

This module returns three dataframe: adni_comp, clin_data, and scan_data.
No filename is needed as input. Uses/designed for 'ADNIMERGE.csv' only.

This module is not designed to run with other .csv files.
Suggested namespace is aw.
"""

def wrangle_adni():
    """This function returns three dataframes.

    Unpack the dataframes when calling the function.
    """

    # ensure pandas availability for the function
    if 'pd' not in globals():
        import pandas as pd

    # read in the data to a pandas dataframe
    adni_full = pd.read_csv('ADNIMERGE.csv', dtype='object')

    # set the logical orders for the two diagnoses
    DX = ['CN', 'MCI', 'AD']
    DX_bl = ['CN', 'SMC', 'EMCI', 'LMCI', 'AD']

    # initialize empty dataframe
    adni = pd.DataFrame()

    # convert datatypes to categorical, datetime, and int
    adni.loc[:, 'PTGENDER'] = pd.Categorical(adni_full.PTGENDER)
    adni.loc[:, 'DX'] = pd.Categorical(adni_full.DX, ordered=True, categories=DX)
    adni.loc[:, 'DX_bl'] = pd.Categorical(adni_full.DX_bl, ordered=True, categories=DX_bl)
    adni.loc[:, 'EXAMDATE'] = pd.to_datetime(adni_full['EXAMDATE'])
    adni.loc[:, 'EXAMDATE_bl'] = pd.to_datetime(adni_full['EXAMDATE_bl'])
    adni.loc[:, 'PTEDUCAT'] = adni_full.PTEDUCAT.astype('int')
    adni.loc[:, 'Month'] = adni_full.Month.astype('int')
    adni.loc[:, 'RID'] = adni_full.RID.astype('int')

    # create a list of float data columns, loop and assign float dtypes
    floats = ['AGE', 'CDRSB', 'ADAS11', 'ADAS13', 'MMSE', 'RAVLT_immediate', 'Hippocampus',
              'Ventricles', 'WholeBrain', 'Entorhinal', 'MidTemp', 'FDG', 'AV45']

    # loop and assign dtypes
    for i in floats:
        adni.loc[:, i] = adni_full[i].astype('float')

        # age has no baseline '_bl' equivalent
        if i == 'AGE':
            continue

        # every other column has a '_bl' equivalent to convert as well
        else:    
            y = i + '_bl'
            adni.loc[:, y] = adni_full[y].astype('float')

    # drop columns with too much missing data
    adni.drop(labels=['FDG', 'FDG_bl', 'AV45', 'AV45_bl'], axis='columns', inplace=True)

    # set the index
    adni.set_index(adni.RID, inplace=True)

    # sort the index
    adni.sort_index(inplace=True)

    # remove redundant columns
    adni.drop('RID', axis='columns', inplace=True)

    # calculate dynamic age
    adni.loc[:, 'AGE_dynamic'] = adni.AGE + (adni.Month / 12)
    
    # create dataframe with only patients that have complete data
    adni_rmv = adni.dropna(how='any')
    
    # filter those results to only patients with multiple visits
    num_comp_exams = adni_rmv.groupby('RID')['EXAMDATE_bl'].count()
    adni_comp_filter = num_comp_exams[num_comp_exams > 1]
    adni_comp = adni_rmv.loc[adni_comp_filter.index]

    # map baseline diagnosis categories to match subsequent diagnosis categories
    # map new column for DX_bl to categorize based on subsequent DX categories
    # 'SMC' -> 'CN' due to medical definitions
    # combine 'LMCI' and 'EMCI' into 'MCI'
    mapper = {'SMC': 'CN', 'LMCI': 'MCI', 'EMCI': 'MCI', 'CN': 'CN', 'AD': 'AD'}
    adni_comp.loc[:, 'DX_bl2'] = adni_comp.DX_bl.map(mapper)
    
    # isolate clinical data
    clin_cols = ['EXAMDATE', 'EXAMDATE_bl', 'Month', 'PTGENDER', 'DX', 'DX_bl', 'PTEDUCAT', 'AGE', 'AGE_dynamic',
                 'CDRSB', 'CDRSB_bl', 'ADAS11', 'ADAS11_bl', 'ADAS13', 'ADAS13_bl', 'MMSE',
                 'MMSE_bl', 'RAVLT_immediate', 'RAVLT_immediate_bl', 'DX_bl2']
    clin_data = pd.DataFrame()
    clin_data = adni.reindex(columns=clin_cols)

    # filter the scan data
    scan_cols = ['EXAMDATE', 'EXAMDATE_bl', 'Month', 'PTGENDER', 'DX', 'DX_bl', 'PTEDUCAT', 'AGE', 'AGE_dynamic',
                 'Hippocampus', 'Hippocampus_bl', 'Ventricles', 'Ventricles_bl', 'WholeBrain', 'WholeBrain_bl',
                 'Entorhinal', 'Entorhinal_bl', 'MidTemp', 'MidTemp_bl', 'DX_bl2']
    scan_data = pd.DataFrame()
    scan_data = adni.reindex(columns=scan_cols)

    return adni_comp, clin_data, scan_data