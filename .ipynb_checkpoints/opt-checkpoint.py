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

def optimize(fe, biomarker, fp_rate, dt_rate, size, gender):
    fp_rate = 100 - fp_rate
    dt_rate = 100 - dt_rate
    
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
        th_value_at_fp_rate = pd.DataFrame({biomarker: [np.percentile(np.random.choice(ad[biomarker], 
                                                                    size=len(ad)),fp_rate) for i in range(size)]})
        
        th_value_at_dt_rate = pd.DataFrame({biomarker: [np.percentile(np.random.choice(non_ad[biomarker], 
                                                                    size=len(non_ad)),dt_rate) for i in range(size)]})
        
    else:
        # create the bootstrap distribution for AD
        th_value_at_fp_rate = pd.DataFrame({biomarker: [np.percentile(np.random.choice(ad[biomarker], 
                                                                    size=len(ad)),fp_rate) for i in range(size)]})
    
        th_value_at_dt_rate = pd.DataFrame({biomarker: [np.percentile(np.random.choice(non_ad[biomarker], 
                                                                    size=len(non_ad)),dt_rate) for i in range(size)]})
    
    fp_th_value = np.mean(th_value_at_dt_rate[biomarker])
    dr_th_value = np.mean(th_value_at_fp_rate[biomarker])
    
    # display further results     
    print(100-fp_rate, '% false positive threshold value: ', fp_th_value)
    #print('Mean 95th percentile of bootstrap samples for non AD patients: ', np.mean(bs_non_95[biomarker]))
    print(100-dt_rate, '% detection threshold value: ', dr_th_value)
    #print('Mean 5th percentile of bootstrap samples for non AD patients: ', np.mean(bs_ad_5[biomarker]))
    
    return th_value_at_dt_rate, th_value_at_fp_rate

def get_reverse(th_value_at_dt_rate, th_value_at_fp_rate, fp_rate, dt_rate, fe, biomarker, gender, increase=True):
    if gender == 'males':
        ad = fe[(fe.DX == 'AD') & (fe.PTGENDER == 'Male')]
        non = fe[(fe.DX != 'AD') & (fe.PTGENDER == 'Male')]
    else:
        ad = fe[(fe.DX == 'AD') & (fe.PTGENDER == 'Female')]
        non = fe[(fe.DX != 'AD') & (fe.PTGENDER == 'Female')]
        
    bs1 = pd.DataFrame({'percentiles': 
                         [scipy.stats.percentileofscore(ad.sample(len(ad),replace=True)[biomarker],
                                                        np.mean(th_value_at_dt_rate).values) for i in range(10000)]})
    
    bs2 = pd.DataFrame({'percentiles': 
                         [scipy.stats.percentileofscore(non.sample(len(non),replace=True)[biomarker],
                                                        np.mean(th_value_at_fp_rate).values) for i in range(10000)]})
    
    ad = round(np.mean(bs1.percentiles),2)
    non = round(np.mean(bs2.percentiles),2)
    
    if increase:
        print('The detection rate for AD at', fp_rate, '% false positive rate: ', round(100-ad,2), '%')
        print('The false positive rate at', dt_rate, '% AD detection: ', round(100-non,2), '%')
        return round(100-ad,2), round(100-non,2)
    else:    
        print('The detection rate for AD at',fp_rate, '% false positive rate: ', ad, '%')
        print('The false positive rate at', dt_rate, '% AD detection: ', non, '%')
        return ad, non
    
    
def optimize_combo(fe, biomarker, fp_rate, dt_rate, size, gender, increase=True):
    
    if increase == False:
        fp_rate = 100 - fp_rate
        dt_rate = 100 - dt_rate
    
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
        th_value_at_fp_rate = pd.DataFrame({biomarker: [np.percentile(np.random.choice(ad[biomarker], 
                                                                    size=len(ad)),fp_rate) for i in range(size)]})
        
        th_value_at_dt_rate = pd.DataFrame({biomarker: [np.percentile(np.random.choice(non_ad[biomarker], 
                                                                    size=len(non_ad)),dt_rate) for i in range(size)]})
        
    else:
        # create the bootstrap distribution for AD
        th_value_at_fp_rate = pd.DataFrame({biomarker: [np.percentile(np.random.choice(ad[biomarker], 
                                                                    size=len(ad)),fp_rate) for i in range(size)]})
    
        th_value_at_dt_rate = pd.DataFrame({biomarker: [np.percentile(np.random.choice(non_ad[biomarker], 
                                                                    size=len(non_ad)),dt_rate) for i in range(size)]})
    
    fp_th_value = np.mean(th_value_at_dt_rate[biomarker])
    dr_th_value = np.mean(th_value_at_fp_rate[biomarker])
    
    # display results     
    print(100-fp_rate, '% false positive threshold value: ', fp_th_value)
    print(100-dt_rate, '% detection threshold value: ', dr_th_value)
    
    # next function begins here
           
    bs1 = pd.DataFrame({'percentiles': 
                         [scipy.stats.percentileofscore(ad.sample(len(ad),replace=True)[biomarker],
                                                        np.mean(th_value_at_dt_rate).values) for i in range(10000)]})
    
    bs2 = pd.DataFrame({'percentiles': 
                         [scipy.stats.percentileofscore(non_ad.sample(len(non_ad),replace=True)[biomarker],
                                                        np.mean(th_value_at_fp_rate).values) for i in range(10000)]})
    
    ad_score = round(np.mean(bs1.percentiles),2)
    non_score = round(np.mean(bs2.percentiles),2)
    
    if increase:
        print('The detection rate for AD at', fp_rate, '% false positive rate: ', round(100-ad_score,2), '%')
        print('The false positive rate at', dt_rate, '% AD detection: ', round(100-non_score,2), '%')
        #return round(100-ad_score,2), round(100-non_score,2)
    else:    
        print('The detection rate for AD at',100-fp_rate, '% false positive rate: ', ad_score, '%')
        print('The false positive rate at', 100-dt_rate, '% AD detection: ', non_score, '%')
        #return ad_score, non_score