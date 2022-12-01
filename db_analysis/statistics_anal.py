import csv

import researchpy as researchpy
import rp as rp
from sklearn.metrics import cohen_kappa_score

import statistics_anal
from math import log
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

from sklearn.feature_selection import VarianceThreshold
from statsmodels.stats.weightstats import ttest_ind
import numpy
import scipy
import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, LinearRegression
from statsmodels.formula.api import ols
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import numpy as np
import scipy.stats as stats

import pandas as pd
from scipy.stats import chisquare, chi2_contingency, fisher_exact, power_divergence

from plot_graphs import GRAPH_DIR
from user_behaviour_table import get_filename, LATEX_TABLE_ORDER, PRINT_ORDER
from utils import connect_to_db

ANSWERS_MAP = {'Y':1,'M':2,'N':3}

def anova_from_file(fname, metric_field, group_by):
    df = pd.read_csv(fname)
    result = df.groupby(group_by)[metric_field].apply(list)
    print(result)
    F, p = stats.f_oneway(*result)
    print(p)
    print(F)


def anova_from_count(y,m,n):
    sum = m+n+y
    y_array = [1]*y +[0]*(sum-y)
    m_array = [1]*m +[0]*(sum-m)
    n_array = [1]*n +[0]*(sum-n)
    F, p = stats.f_oneway(y_array,m_array,n_array)
    print(p)
    print(F)


def three_way_anova(c1,t1,g1,c2,t2,g2,c3,t3,g3):
    print(g1 + ':' + str(c1/t1))
    print(g2 + ':' + str(c2/t2))
    print(g3 + ':' + str(c3/t3))
    a1 = [1]*c1 + [0]*(t1-c1)
    a2 = [1]*c2 + [0]*(t2-c2)
    a3 = [1]*c3 + [0]*(t3-c3)
    three_way_anova_from_array(a1,g1,a2,g2,a3,g3,'CTR')

def three_way_anova_from_array(a1,g1,a2,g2,a3,g3,measure):
    pd.set_option("display.max_rows", None, "display.max_columns", None)


    d = []
    for g, a in {g1: a1, g2: a2, g3: a3}.items():
        for x in a:
            d.append({measure: x, 'group': g})

    # df = pd.DataFrame({'score':[],'group': np.repeat(['a1', 'a2', 'a3'], repeats=10)})
    df = pd.DataFrame(d)
    model = ols(measure + ' ~ group', data=df).fit()
    anova_table =  sm.stats.anova_lm(model, typ=2)
    print(anova_table)

    F, p = stats.f_oneway(a1,a2,a3)
    print('p=' + str(p))
    print('F= ' + str(F))

    if p < 0.05:
        print ('p < 0.05')
    else:
        return

    print('POST HOC')

    df = pd.DataFrame(d)
 #   print(df)
    tukey = pairwise_tukeyhsd(endog=df[measure],
                              groups=df['group'],
                              alpha=0.05)

    print(tukey)


def paired_ttest(c1,t1,c2,t2, print_res = True):
    a1 = [1] * c1 + [0] * (t1 - c1)
    a2 = [1] * c2 + [0] * (t2 - c2)
    t = scipy.stats.ttest_rel(a1, a2)
    s = t.statistic
    p = t.pvalue
    if print_res:
        print('s = ' + str(s) + ' p = ' + str(p))
    return t


def ttest_from_count(c1,t1,c2,t2):
    a1 = [1]*c1 + [0]*(t1-c1)
    a2 = [1]*c2 + [0]*(t2-c2)
    #ttest = scipy.stats.ttest_ind(a1,a2)
   # if ttest.pvalue < 0.05:
   #     print ('TTest significant, p < 0.05')
   # else:
   #     print('TTest NOT significant, p >= 0.05')
    ttest = ttest_ind(a1, a2, alternative='larger')
    print('larger')
    print(ttest)
    ttest = ttest_ind(a1, a2, alternative='smaller')
    print('smaller')
    print(ttest)

def link1_phys_ctr_test():
    print('anova')
    three_way_anova(c1 = 91,t1=151, g1= 'no ads', c2=11,t2=105, g2 = 'direct marketing',  c3=24,t3=65, g3 = 'spnsored')
    print('ttest no ads - ads')
    ttest_from_count(c1 = 91,t1=151,c2=11,t2=105)
    print('ttest no ads - sponsored')
    ttest_from_count(c1 = 91,t1=151,c2=24,t2=65)
    print('ttest ads - sponsored')
    ttest_from_count(c1 = 11,t1=105,c2=24,t2=65)
    print('ttest no ads - sm')
    ttest_from_count(c1=91, t1=151, c2=18, t2=32)

def first_organic_result_test():
    o1_clicks_noads = 91
    o1_link_vis = 151

    o1_ads_clicks = 58
    o1_ads_link_vis = 105

    o1_s_clocks =  44
    o1_s_vis = 65

    print('anova')
    three_way_anova(c1 = o1_clicks_noads,t1=o1_link_vis,g1='noads',c2=o1_ads_clicks,t2=o1_ads_link_vis,g2='direct marketing',c3=o1_s_clocks,t3=o1_s_vis,g3='sponosored content')

    print('ttest no ads - ads')
    ttest_from_count(c1 = o1_clicks_noads,t1=o1_link_vis,c2=o1_ads_clicks,t2=o1_ads_link_vis)
    print('ttest no ads - sponsored')
    ttest_from_count(c1 = o1_clicks_noads,t1=o1_link_vis,c2=o1_s_clocks,t2=o1_s_vis)
    print('ttest ads - sponsored')
    ttest_from_count(c1 = o1_ads_clicks,t1=o1_ads_link_vis,c2=o1_s_clocks,t2=o1_s_vis)


def first_two_results_ctr():
    no_ads_first2 = 29
    no_ads_first_2_vis = 151

    am_first_2 = 1
    an_first_2 = 1
    first_2_vis_ads = 105

    sm_first_2 =11
    sn_first_2 =3
    sm_first2_vis = 32
    sn_first2_vis = 33

    ttest_from_count(c1=sm_first_2, t1=sm_first2_vis, c2=sn_first_2, t2=sn_first2_vis)
    #ttest_from_count(c1=am_first_2, t1=first_2_vis_ads, c2=an_first_2, t2=first_2_vis_ads)


def ad_first_organic_result_test():
    ad_click_an =5
    vis_an = 105

    ad_click_am = 6
    vis_am = 105

    print('an-am ttest')
    ttest_from_count(c1=ad_click_an, t1=vis_an, c2=ad_click_am, t2=vis_am)


    ad_click_sn =  6
    vis_sn =  65

    ad_click_sm = 18
    vis_sm = 65

    print('sn-sm ttest')
    ttest_from_count(c1=ad_click_sn, t1=vis_sn, c2=ad_click_sm, t2=vis_sm)


def log_learn(c1,t1,c2,t2, c1_c2_group_index,var_name):
    data = [{c1_c2_group_index:0,var_name:1}]*c1
    data.extend([{c1_c2_group_index:0,var_name:0}]*(t1-c1))
    data.extend([{c1_c2_group_index:1,var_name:1}]*c2)
    data.extend([{c1_c2_group_index:1,var_name:0}]*(t2-c2))
    df = pd.DataFrame(data)
   # print(df)
    Xtrain = df[[c1_c2_group_index]]
    ytrain = df[[var_name]]
    model = sm.Logit(ytrain, Xtrain)
    model_fit = model.fit()
    print(model_fit.summary())

    logisticRegr = LogisticRegression(solver='newton-cg',  fit_intercept=False, penalty='none')
    f = logisticRegr.fit(Xtrain, ytrain)
    print(f.coef_)
    #print(f)
    #cov = model_fit.cov_params()
    #print('cov_params')
    #print(cov)
    #std_err = np.sqrt(np.diag(cov))
    #print('std_err')
    #print(std_err)
    #z_values = model_fit.params / std_err

    #x_train = numpy.array([1]*(y1+m1+n1)+[2]*(y2+m2+n2))
    #y_train =  numpy.array(['y']*y1 + ['m']*m1+['n']*n1+['y']*y2+['m']*m2+['n']*n2)

    #x_train = x_train.reshape(-1, 1)
    #y_train = y_train.reshape(-1, 1)

    #logit_model = sm.Logit(y_train, x_train)
    #result = logit_model.fit()
    #print(result.summary2())

    #logisticRegr = LogisticRegression()
    #f = logisticRegr.fit(x_train, y_train)
    #print(f)



def multivariate_log_learn(c1_c2_group_index,y1,m1,n1,y2,m2,n2):
    data = [{c1_c2_group_index:0,'response':'Y'}]*y1
    data.extend([{c1_c2_group_index:0,'response':'M'}]*m1)
    data.extend([{c1_c2_group_index:0,'response':'N'}]*n1)
    data.extend([{c1_c2_group_index:1,'response':'Y'}]*y2)
    data.extend([{c1_c2_group_index:1,'response':'M'}]*m2)
    data.extend([{c1_c2_group_index:1,'response':'N'}]*n2)
    df = pd.DataFrame(data)
    #print(df)
    Xtrain = df[[c1_c2_group_index]]
    ytrain = df[['response']] #log
    model = sm.MNLogit(ytrain, Xtrain)
    model_fit = model.fit()
    print(model_fit.summary())
    #model = LogisticRegression(multi_class='multinomial', solver='newton-cg', fit_intercept=False, penalty = 'none')
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs', fit_intercept=False, penalty = 'none')

    f = model.fit(Xtrain, ytrain)


    print(f.coef_ )
    #cov = model_fit.cov_params()
    #print('cov_params')
    #print(cov)
    #std_err = np.sqrt(np.diag(cov))
    #print('std_err')
    #print(std_err)
    #z_values = model_fit.params / std_err

    #x_train = numpy.array([1]*(y1+m1+n1)+[2]*(y2+m2+n2))
    #y_train =  numpy.array(['y']*y1 + ['m']*m1+['n']*n1+['y']*y2+['m']*m2+['n']*n2)

    #x_train = x_train.reshape(-1, 1)
    #y_train = y_train.reshape(-1, 1)

    #logit_model = sm.Logit(y_train, x_train)
    #result = logit_model.fit()
    #print(result.summary2())

    #logisticRegr = LogisticRegression()
    #f = logisticRegr.fit(x_train, y_train)
    #print(f)


def decision_diff_two_way(y1,m1,n1,y2,m2,n2,c1_c2_group_index):
    t1 = y1+m1+n1
    t2 = y2+m2+n2
    log_learn(c1=y1,t1=t1,c2=y2,t2=t2, c1_c2_group_index =c1_c2_group_index,var_name ='Y')
    #log_learn(c1=m1,t1=t1,c2=m2,t2=t2, c1_c2_group_index =c1_c2_group_index,var_name ='M')
    #log_learn(c1=n1,t1=t1,c2=n2,t2=t2, c1_c2_group_index =c1_c2_group_index,var_name ='N')

def decision_diff():
    decision_diff_two_way(y2=17, m2=23, n2=12, y1=13, m1=12, n1=7, c1_c2_group_index = 'M_SM')
    #decision_diff_two_way(y1=17, m1=23, n1=12, y2=13, m2=12, n2=7, c1_c2_group_index = 'M_SM')


def chi_negative_bias():
    results = [[10, 20, 17], [11, 18, 23], [7, 7, 17]]
    chi_cont(results)


def chi_inconclusive_bias():
    results = [[17, 23, 12], [16, 24, 12], [13,12,7]]
    chi_cont(results)

def chi_cont(results, row_index, col_index=['Y','M','N']):
  #  results = [r1, r2]
    print(results)
    F = np.array(results)


   # ddsr, p = fisher_exact(F, alternative='two-sided')
   # print('Fisher:'+ str(p))
    df = pd.DataFrame(data=F, index=row_index, columns=col_index)


 #   table = sm.stats.Table(df)


  #  print('table.cumulative_oddsratios')
  #  print(table.cumulative_oddsratios)

#    print('table.chi2_contribs')
#    print(table.chi2_contribs)


    #print('table.local_oddsratios')
    #print(table.local_oddsratios)
  #  print(table)



#    print('resid_pearson Residuals')


  #  print(table)
  #  print('resid_pearson Residuals')
  #  print(table.resid_pearson)
  #  print('standardized_resids Residuals')
  #  print(table.standardized_resids)

    obs = np.array(results)
    t = chi2_contingency(obs)



    print('test statistics:' + str(t[0]))
    print('p-value:' + str(t[1]))
    print('df ' + str(t[2]))
    e = 'expected' + str(t[3])
    print(e)
    sig = t[1] <= 0.05
    if sig:
        result = 'Chi test significant'
    else:
        result = 'Chi test NOT significant'


    #r = power_divergence(obs, lambda_="log-likelihood")
    #print('power_divergence')
    #print(r)

    if len(col_index) == 2 and sig:
        t = sm.stats.Table2x2(np.array(df))
        print(t.summary())

    return  result

def chi_no_ads():
    results = [[26,13,10], [17,23,12],[10,20,17]]
    chi_cont(results)


def chi_all():
    results = [[26, 13, 10], [17, 23, 12], [10, 20, 17], [16, 24, 12], [11, 18, 23], [13, 12, 7], [7, 7, 17]]
    chi_cont(results, col_index=['Y','M','N'], row_index=['Y','M','N','AM','AN','SM','SN'])


def query_chi():
    #meltaonin,omega,ginko

    y_results = [[8,5,4],[12,3,5],[6,5,1]]
    m_results = [[5,7,5],[7,8,4],[5,8,3]]
    n_results = [[7, 5, 4], [0, 8, 7], [3, 7, 6]]
    print('y results')
    chi_cont(y_results)
    print('m_results')
    chi_cont(m_results)
    print('n_results')
    chi_cont(n_results)


def chi_tests():
    sn_norm = [0.225,0.225,0.55]
    n_norm =  [0.22,0.42,0.36]
    dummy_sn = [x*80 for x in sn_norm]
    dummy_n = [x * 70 for x in n_norm]
    print('SN-N')
    chi_cont([[10, 20, 17], [7, 7, 17]])
    #chi_cont([14,  17], [37,  10])
    print('SN-N-dummy')
    chi_cont([[10, 20, 17], dummy_sn])
    #chi_cont(dummy_n, dummy_sn)


    print('SM-M')
    chi_cont([13, 12, 7], [17, 23, 12])
    #chi_cont( [17, 12, 23], [13, 7, 12])

    print('SM-M-dummy')
    sm_norm = [0.4,0.38,0.22]
    m_norm =  [0.33,0.44,0.23]
    dummy_sm = [x*100 for x in sm_norm]
    dummy_m = [x*100 for x in m_norm]
  #  chi_cont([21, 20, 11], dummy_sm)
    chi_cont(dummy_m, dummy_sm)


def threeway_log_linear_models():
    df = pd.read_csv('../resources/reports/responses_per_sequnce.csv')
    print(df)
    #columnsNamesArr = df.columns.values
    #rowNamesArr = df.rows.values
    Xtrain = []
    Ytrain = []
    rows = len(df.index)
    columns = 3 # 3 possible answers
    expected = []
    ad_config = []
    total = 0
    #generating expected table vals
    for i in range(0, rows):
        l = list(df.iloc[i])
        ad_c = df.iloc[i, 0][0]
        if ad_c == 'A':
            ad_config.append(rows+columns+2)
        elif ad_c == 'S':
            ad_config.append(rows+columns+3)
        else:
            ad_config.append(rows+columns +1)



        row_sum = sum(l[1:])
        total +=row_sum
        row = []

        for j in range(0, columns):
            c_vals = list(df.iloc[:, j+1])
            col_sum = sum(c_vals)
            row.append(row_sum*col_sum)
        expected.append(row)

    for i in range(0, rows):
        for j in range(0, columns):
            expected[i][j] =expected[i][j]/total

    for i in range(0,rows):
        for j in range(0,columns):
            x = [0]*(rows+columns+3)
            x[i]=1
            x[j+rows]=1
            x[ad_config[i]] = 1
            Xtrain.append(x)
            Ytrain.append(log(expected[i][j]))

            #Ytrain.append(expected[i][j])
    print(Xtrain)
    print(Ytrain)
    #print([log(y) for y in Ytrain])
    #Y = np.array([log(y) for y in Ytrain])
    X = np.array(Xtrain)
    Y = np.array(Ytrain)
    reg = LinearRegression().fit(X,Y )
    print(str(reg.intercept_))
    print(str(reg.coef_))

    X = sm.add_constant(X)  # adding a constant

    model = sm.OLS(Y, X).fit()
    print(model.params)
    print(model.summary())


def log_linear_models(sequences = None):
    df = pd.read_csv('../resources/reports/responses_per_sequnce.csv')
    print(df)
    if sequences:
        sequences_cond = (df['sequence'] == sequences[0])
        for i in range(1,len(sequences)):
            sequences_cond = sequences_cond | (df['sequence'] == sequences[i])
        #df2 = df.loc[((df['sequence'] == 1) & (df['b'] > 0)) | ((df['a'] < 1) & (df['c'] == 100))]
        df = df.loc[sequences_cond]
    print(df)
    #columnsNamesArr = df.columns.values
    #rowNamesArr = df.rows.values
    Xtrain = []
    Ytrain = []
    rows = len(df.index)
    columns = 3 # 3 possible answers
    expected = []
    total = 0
    #generating expected table vals
    for i in range(0, rows):
        l = list(df.iloc[i])
        row_sum = sum(l[1:])
        total +=row_sum
        row = []
        for j in range(0, columns):
            c_vals = list(df.iloc[:, j+1])
            col_sum = sum(c_vals)
            row.append(row_sum*col_sum)
        expected.append(row)
    for i in range(0, rows):
        for j in range(0, columns):
            expected[i][j] =expected[i][j]/total

    for i in range(0,rows):
        for j in range(0,columns):
            x = [0]*(rows+columns)
            x[i]=1
            x[j+rows]=1
            Xtrain.append(x)
            Ytrain.append(log(expected[i][j]))
            #Ytrain.append(expected[i][j])
    print(Xtrain)
    print(Ytrain)
    #print([log(y) for y in Ytrain])
    #Y = np.array([log(y) for y in Ytrain])
    X = np.array(Xtrain)
    Y = np.array(Ytrain)
    reg = LinearRegression().fit(X,Y)
    print(str(reg.intercept_))
    print(str(reg.coef_))

    X = sm.add_constant(X)  # adding a constant

    model = sm.OLS(Y, X).fit()
    print(model.params)
    print(model.summary())


def chi_no_melatonin():
    print('SN-N')
    chi_cont([[3, 15, 13], [7, 7, 17]])
    # chi_cont([14,  17], [37,  10])

    print('SM-M')
    chi_cont([[13, 12, 7], [12, 16, 7]])


def chi_user_bias():
    n_results = [[37,10],[29,23],[14,17]]
    chi_cont(n_results)
    m_results = [[7,12,23],[16,12,24],[13,7,12]]
    chi_cont(m_results)


def two_by_two_test():
    print('#Y/M-SM')
    chi_cont([[17,23+12],[13,12+7]],row_index=['M','SM'], col_index=['Y','N+M'])

    print('#M/M-SM')
    res = np.array([[23,17+12],[12,13+7]]) #M/M-SM
    chi_cont([[23,17+12],[12,13+7]], row_index=['M', 'SM'], col_index=['M', 'N+Y'])


    print('#N/M-SM')
    chi_cont([[12,17+23],[7,13+12]], row_index=['M', 'SM'], col_index=['N', 'M+Y'])


    print('#Y/N-SN')
    chi_cont([[10,20+17],[7,7+17]], row_index=['N', 'SN'], col_index=['Y', 'N+M'])


    print('#M/N-SN')
    chi_cont([[20,10+17],[7, 7+17]], row_index=['N', 'SN'], col_index=['M', 'Y+N'])

    print('#N/N-SN')
    chi_cont([[17,20+10],[17,7+7]], row_index=['N', 'SN'], col_index=['N', 'Y+M'])

    print('#N/N-AN')
    chi_cont([[17,20+10],[23,11+18]], row_index=['N', 'SN'], col_index=['N', 'Y+M'])

    #res = np.array([[,17+12],[24,16+12]]) #N/M-AM



def odds_ratio_tests():
    print('#Y/M-SM')
    scale = 70
    ym = [17 /(23+12+17),(23+12)/(23+12+17)]
    ysm = [13 /(13+12+7),(7+12)/(13+12+17)]

    ym = [x * scale for x in ym]
    ysm = [x * scale for x in ysm]

    #res = np.array([[17,23+12],[13,12+7]]) #Y/M-SM
    res = np.array([ym,ysm])  # Y/M-SM
    t = sm.stats.Table2x2(res)
    print(t.summary())


    print('#M/M-SM')
    res = np.array([[23,17+12],[12,13+7]]) #M/M-SM
    t = sm.stats.Table2x2(res)
    print(t.summary())

    print('#N/M-SM')
    res = np.array([[12,17+23],[7,13+12]]) #N/M-SM
    t = sm.stats.Table2x2(res)
    print(t.summary())


    print('#Y/N-SN')
    res = np.array([[10,20+17],[7,7+17]]) #Y/N-SN
    t = sm.stats.Table2x2(res)
    print(t.summary())

    print('#M/M-SM')

    res = np.array([[20,10+17],[7, 7+17]]) #M/N-SN
    t = sm.stats.Table2x2(res)
    print(t.summary())

    print('#N/M-SM')
    #res = np.array([[17,20+10],[17,7+7]]) #N/N-SN
    nm = [17/47,30/47]
    nsm = [17/31,14/31]
    scale = 60
    m = [x*scale for x in nm]
    ms = [x*scale for x in nsm]
    res = np.array([m,ms]) #N/N-SN
    #res = np.array([[17,20+10],[17,7+7]]) #N/N-SN
    t = sm.stats.Table2x2(res)
    print(t.summary())

    #res = np.array([[,17+12],[24,16+12]]) #N/M-AM


def cont_exp(scale):
     nn = [0.36, 0.64]
     nsn = [0.54, 0.46]
     #chi_cont([[17, 20 + 10], [17, 7 + 7]], row_index=['N', 'SN'], col_index=['N', 'Y+M'])
     #res = [[x * 50 for x in nn], [x * scale for x in nsn]]
     res = [[17, 20 + 10], [x * scale for x in nsn]]
     chi_cont(res, row_index=['N', 'SN'], col_index=['N','Y+M'])
     t = sm.stats.Table2x2(np.array(res))
     print(t.summary())

     mp = [0.33,0.67]
     smp = [0.4,0.6]
     res = [[x*scale for x in mp],[x*scale for x in smp]]
     chi_cont(res, row_index=['M','SM'], col_index=['Y','N+M'])
     t = sm.stats.Table2x2(np.array(res))
     print(t.summary())


#    n = [0.21,0.43,0.36]
#    sn = [0.23,0.23,0.54]
#    res = [[x*scale for x in n],[x*scale for x in sn]]
#    chi_cont(res, row_index=['N','SN'])
#    m = [0.33,0.44,0.23]
#    sm = [0.41,0.375,0.22]
#    res = [[x*scale for x in m],[x*scale for x in sm]]
#    chi_cont(res, row_index=['M','SM'])
def kmeans():
    clusters = [[0.40625,0.375,0.21875],
                [0.233,0.25,0.483],
                [0.43,0.32,0.25],
                [0.11,0.77,0.11],
                [0.3,0.45,0.22],
                [0.33,0.167,0.44],
                [0.17,0.75,0.08],
                [0.55,0.09,0.27],
                [0.211,0.34,0.44],
                [0.2,0.5,0.3],
                [0.46,0.15,0.38],
                [0.54,0.33,0.12],
                [0.36,0.27,0.36]]
    kmeans = KMeans(n_clusters=4, random_state=0).fit(np.array(clusters))
    print(kmeans.labels_)


def logistic_regression(f, group, mode = 'clicks'):
    if f:
        data = pd.read_csv('../resources/reports//'+f+'.csv')  # load data set
    elif group:
        data = pd.read_csv('../resources/reports//user_response_labels_'+group+'.csv')  # load data set
    else:
        data = pd.read_csv('../resources/reports//user_response_labels.csv')  # load data set



    X = data.iloc[:, 1:]  # select columns 1 through end
    if mode == 'seq':
        X = X.iloc[:, 32:]
    elif mode == 'clicks':
        X = X.iloc[:, 32:]

    #selector = VarianceThreshold( 0.0001)
    #selector.fit_transform(X)
   # X.columns ^ low_variance.columns
   # X.shape
   # X.shape
   # X = low_variance

   # low_variance=  data[data.columns[selector.get_support(indices=True)]]

    #        if group != 'O':
#            X = X.iloc[:, :27]
#        else:
#            X = X.iloc[:, :30]
#    elif mode == 'clicks':
#        if group != 'O':
#            X = X.iloc[:, 27:]
#        else:
#            X = X.iloc[:, 30:]

    y = data.iloc[:, 0]  # select column 0, the response
   # sm.add_constant(X)
   # model = sm.MNLogit(y, X)

    #model_fit = model.fit_regularized(method='l1')
    #r = model_fit.summary().as_csv()
    #print(model_fit.summary())

    clf = LogisticRegression( solver='saga', penalty='l1', max_iter=400, C=1.0).fit(X, y)
    #clf = LogisticRegression( solver='saga', penalty='l1', max_iter=400).fit(X, y)
    #clf = LogisticRegression(random_state=0).fit(X, y)
    print('Non Zero weights: ' + str(np.count_nonzero(clf.coef_)))

    #print(clf.classes_)

    weights = []
    weights.append(clf.coef_[2])
    weights.append(clf.coef_[0])
    weights.append(clf.coef_[1])
    for i in [2,0,1]:
        s = ''
        for c in clf.coef_[i]:
            s += str(c) + ','
        print(s)
    return weights


def weight_stats(group, answer, vals):
    non_zero = [x for x in vals if x != 0]
    #print ('non zero  =  ' + len(non_zero ))
    min_v =  min(vals)
    mins = "%.2f" % round(min_v, 2)
    max_v =  max(vals)
    maxs = "%.2f" % round(max_v, 2)
    avg = np.mean(vals)
    avgs ="%.2f" % round(avg, 2)
#    print('min = ' + min(vals))
#    print('max = ' + max(vals))
#    print('avg = ' + mean(vals))
    print(group + '&' + answer +'&'+str(len(non_zero)) + '&' + mins +'&' + maxs + '&' + avgs)

def kendel_tau(input_file):
    ranking = {}
    with open('../resources/reports/' + input_file + '.csv', 'r', newline='') as inputCSV:
        reader = csv.DictReader(inputCSV)
        for row in reader:

            vals = list(row.values())
            if len(vals) > 10 and vals[10] == "":
                vals = vals[:-1]
            vals = vals[1:]
            ys = []
            for i in range(0, len(vals)):
                if vals[i] != "":
                    ys.append(float(vals[i]))
            ranking[row['series']] = ys

    print('M-N')
    tau, p_value = stats.kendalltau(ranking['M'], ranking['N'])
    print(tau, p_value)

    print('Y-M')
    tau, p_value = stats.kendalltau(ranking['Y'], ranking['M'])
    print(tau, p_value)
    print('Y-N')
    tau, p_value = stats.kendalltau(ranking['Y'], ranking['N'])
    print(tau, p_value)

def generete_features_list_ss():
    fl = ['$f_d^1$','$f_i^1$']
    for i in range (1,11):
        fl.append('$f_y^{'+str(i)+'}$')
        fl.append('$f_m^{'+str(i)+'}$')
        fl.append('$f_n^{'+str(i)+'}$')
    return  fl

def generete_features_list_ss2():
    fl = ['$_D^1$','$_I^1$']
    for i in range (1,11):
        fl.append('$_Y^{'+str(i)+'}$')
        fl.append('$_M^{'+str(i)+'}$')
        fl.append('$_N^{'+str(i)+'}$')
    return  fl


def generete_features_list():
    fl = ['(1,d)','(1,i))$']
    for i in range (1,11):
        fl.append((str(i)+'y)'))
        fl.append((str(i)+'m)'))
        fl.append((str(i)+'n)'))
#        fl.append('$f_m^{'+str(i)+'}$')
#        fl.append('$f_n^{'+str(i)+'}$')
    return  fl

def heatmap():
    params = {'mathtext.default': 'regular'}
    plt.rcParams.update(params)
    data = []
    print('NO ADS')
#    groups = ['No Ads, Y', 'No Ads, M', 'No Ads, N',
#              'DMarketing Ads, Y', 'Direct Marketing Ads, M', 'Direct Marketing Ads, N',
#              'Indirect Marketing Ads, Y', 'Indirect Marketing Ads, M', 'Indirect Marketing Ads, N']



    plt.subplots(figsize=(10, 5))
    #features = generete_features_list()
    features = generete_features_list_ss2()
    reg_res = logistic_regression(f=None, group='O', mode='clicks')
    weight_stats('No Ads','Y',reg_res[0] )
    weight_stats('No Ads','M',reg_res[1] )
    weight_stats('No Ads','N',reg_res[2] )
    data.append(reg_res[0])
    data.append(reg_res[1])
    data.append(reg_res[2])
    reg_res = logistic_regression(f=None, group='A', mode='clicks')
    weight_stats('DM','Y',reg_res[0] )
    weight_stats('DM','M',reg_res[1] )
    weight_stats('DM','N',reg_res[2] )

    data.append(reg_res[0])
    data.append(reg_res[1])
    data.append(reg_res[2])

    reg_res = logistic_regression(f=None, group='S', mode='clicks')
    weight_stats('IM','Y',reg_res[0] )
    weight_stats('IM','M',reg_res[1] )
    weight_stats('IM','N',reg_res[2] )

    data.append(reg_res[0])
    data.append(reg_res[1])
    data.append(reg_res[2])

    groups = ['No Ads, Y', 'No Ads, M', 'No Ads, N',
              'DM, Y', 'DM, M', 'DM, N',
              'IM, Y', 'IM, M', 'IM, N']

    ax = sns.heatmap(data, xticklabels=features, yticklabels=groups, cmap='coolwarm')
    ax.tick_params(axis='x', which='major', labelsize=12)
    ax.hlines([3, 6, 9], *ax.get_xlim())
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=0, ha="left",
             rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.

    #    plt.imshow(data, cmap='coolwarm', interpolation='nearest')

    #ax = sns.heatmap(data, linewidth=0.5, cmap='coolwarm')

#    ax.set_title("Logistic Regression Feature Weights ")

    plt.show()
#    plt.savefig(GRAPH_DIR + 'heatmap_C_1_rec.pdf')


def heatmap_per_answer_order(mode='clicks'):
    params = {'mathtext.default': 'regular'}
    plt.rcParams.update(params)
    data_dict = {'O':{'Y':[],'M':[],'N':[]},'A':{'Y':[],'M':[],'N':[]},'S':{'Y':[],'M':[],'N':[]}}
    data = []
    print('NO ADS')
    #    groups = ['No Ads, Y', 'No Ads, M', 'No Ads, N',
    #              'DMarketing Ads, Y', 'Direct Marketing Ads, M', 'Direct Marketing Ads, N',
    #              'Indirect Marketing Ads, Y', 'Indirect Marketing Ads, M', 'Indirect Marketing Ads, N']

    plt.subplots(figsize=(10, 5))
    # features = generete_features_list()
    features = generete_features_list_ss2()
    reg_res = logistic_regression(f=None, group='O', mode=mode)

    data_dict['O']['Y'] = reg_res[0]
    data_dict['O']['M'] = reg_res[1]
    data_dict['O']['N'] = reg_res[2]
    reg_res = logistic_regression(f=None, group='A', mode=mode)

    data_dict['A']['Y'] = reg_res[0]
    data_dict['A']['M'] = reg_res[1]
    data_dict['A']['N'] = reg_res[2]

    reg_res = logistic_regression(f=None, group='S', mode=mode)

    data_dict['S']['Y'] = reg_res[0]
    data_dict['S']['M'] = reg_res[1]
    data_dict['S']['N'] = reg_res[2]

    weight_stats('No Ads', 'Y', data_dict['O']['Y'])
    weight_stats('DM', 'Y', data_dict['A']['Y'])
    weight_stats('IM', 'Y', data_dict['S']['Y'])

    weight_stats('No Ads', 'M', data_dict['O']['M'])
    weight_stats('DM', 'M', data_dict['A']['M'])
    weight_stats('IM', 'M', data_dict['S']['M'])
    weight_stats('No Ads', 'N', data_dict['O']['N'])
    weight_stats('DM', 'N', data_dict['A']['N'])
    weight_stats('IM', 'N', data_dict['S']['N'])



    groups = ['No Ads, Yes','Direct Marketing, Yes', 'Indirect Marketing, Yes',
              'No Ads, Maybe','Direct Marketing, Maybe', 'Indirect Marketing, Maybe',
              'No Ads, No','Direct Marketing, No', 'Indirect Marketing, No']

    for answer in ['Y','M','N']:
        for ad_c in ['O', 'A', 'S']:
            data.append(data_dict[ad_c][answer])
    #sns.set(font_scale=1.4)
    ax = sns.heatmap(data, xticklabels=features, yticklabels=groups, cmap='coolwarm')
    ax.tick_params(axis='x', which='major', labelsize=15)
    ax.hlines([3, 6, 9], *ax.get_xlim())
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=0, ha="left",
             rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.

    #    plt.imshow(data, cmap='coolwarm', interpolation='nearest')

    # ax = sns.heatmap(data, linewidth=0.5, cmap='coolwarm')

    #    ax.set_title("Logistic Regression Feature Weights ")
    #plt.show()


    plt.savefig(GRAPH_DIR + 'heatmap_C_1_by_answer2.pdf',bbox_inches='tight')

def chi_3way(r1,r2,l1,l2, to_print = False):
    print(l1 + '-' + l2)
    result = chi_cont([r1,r2],row_index=[l1,l2],col_index=['Y','M','N'])
    print(result)
    print('-------------------------------------------------------------')
    print(l1 + '-' + l2 + ' Yes')
    result = chi_cont([[r1[0],r1[1]+r1[2]], [r2[0],r2[1]+r2[2]]], row_index=[l1, l2], col_index=['Y', 'M+N'])
    print(result)
    print(l1 + '-' + l2 + ' Maybe')
    result = chi_cont([[r1[1],r1[0]+r1[2]], [r2[1],r2[0]+r2[2]]], row_index=[l1, l2], col_index=['M', 'Y+N'])
    print(result)
    print(l1 + '-' + l2 + ' No')
    result = chi_cont([[r1[2], r1[0] + r1[1]], [r2[2], r2[0] + r2[1]]], row_index=[l1, l2], col_index=['N', 'Y+m'])
    print(result)


def results_chi_stats():
    #Y-AY-SY

    #Y-AY
    chi_3way([52, 24,21], [53, 35,11],'Y','AY')
    chi_3way([52, 24,21], [42, 33,20],'Y','SY')

    chi_3way([31, 45,22], [30, 45,22],'M','AM')
    chi_3way([31, 45,22], [40, 40,20],'M','SM')

    chi_3way([27, 34, 34], [21, 34,44],'N','AN')
    chi_3way([27, 34, 34], [23, 25,48],'N','SN')


def get_response_stats():
    ad_config_counter = {'S':0,'A':0,'X':0}
    query_counters = {}
    bias_counters = {}
    fname = get_filename('user_behaviour', None)
    condition_counter = {}
    with open(fname, newline='', encoding='utf8') as csvf:
        reader = csv.DictReader(csvf)
        for row in reader:
            url = row['url']
            query = url.split('-')[0]
            sequence = row['sequence']
            ad_condition = sequence[0]
            if ad_condition == 'S' or ad_condition == 'A':
                ad_config_counter[ad_condition] += 1
                bias = sequence[1]
            else:
                ad_config_counter['X'] += 1
                bias = sequence[0]
            if bias not in bias_counters:
                bias_counters[bias] = 0

            bias_counters[bias]+=1
            if query not in query_counters:
                query_counters[query] = 0
            query_counters[query] += 1
            condition=query+'-'+ad_condition+'-'+bias
            if condition not in condition_counter:
                condition_counter[condition] = 0
            condition_counter[condition] += 1

    ad_vals = list(ad_config_counter.values())
    for a,v in ad_config_counter.items():
        print(a + ':' + str(v))
    print('Ad config mean:' + str(np.mean(ad_vals)))
    print('Ad config STD:' + str(np.std(ad_vals)))

    query_vals = list(query_counters.values())

    for a,v in query_counters.items():
        print(a + ':' + str(v))

    print('Query mean:' + str(np.mean(query_vals)))
    print('Query stdev:' + str(np.std(query_vals)))

    bias_vals = list(bias_counters.values())

    for a,v in bias_counters.items():
        print(a + ':' + str(v))

    print('Bias mean:' + str(np.mean(bias_vals)))
    print('Bias stdev:' + str(np.std(bias_vals)))

    condition_vals = list(condition_counter.values())
    print ('Condition mean:' + str(np.mean(condition_vals)))
    print ('Condition stdev:' + str(np.std(condition_vals)))

    print('No Ads & ' + str(ad_config_counter['X']) + ' \\\\')
    print('\hline')
    print('Direct marketing ads & ' + str(ad_config_counter['A']) + "\\\\")
    print('\hline')
    print('Indirect marketing  ads & ' + str(ad_config_counter['S']) + " \\\\")
    print('\hline')
    for query in query_counters:
        print(query +' & ' + str(query_counters[query]) + "\\\\")
        print('\hline')





def get_user_stats():
    time_spent = []
    num_links = []
    ad_clicks = 0
    all_clicks = 0
    users_str = '('
    fname = get_filename('user_behaviour', None)
    with open(fname, newline='', encoding='utf8') as csvf:
        reader = csv.DictReader(csvf)
        for row in reader:
            time_spent.append(float(row['search_time_exp']))
            nl = float(row['num_links_pressed'])
            all_clicks += nl
            num_links.append(nl)
            users_str += "'" + row['WorkerId'] + "'" + ','
            sequence = row['sequence']
            if sequence[0] == 'S' or sequence[0] =='A' :
                ad_clicks += int(row['link1'])

    print('---------------NUM LINKS------------------')
    print('All links:' + str(all_clicks))
    print('ad clicks:' + str(ad_clicks))
    print('orgainc clicks:' + str(all_clicks - ad_clicks))
    print('% ad clicks:' + str(ad_clicks/all_clicks))
    print('% orgainc clicks:' + str(1 - ad_clicks/all_clicks))

    print('Num links min: ' + str(min(num_links)))
    print('Num links max: ' + str(max(num_links)))
    print ('Num links mean: ' + str(np.mean(num_links)))
    print ('Num links stdev: ' + str(np.std(num_links)))

    print('---------------EXP TIME------------------')
    print ('Exp time min: ' + str(min(time_spent)))
    print ('Exp time max: ' + str(max(time_spent)))

    print ('Exp time mean: ' + str(np.mean(time_spent)))
    print ('Exp time stdev: ' + str(np.std(time_spent)))

    users_str = users_str[:-1]+')'
    db = connect_to_db('shared')
    mycursor = db.cursor()
    # links = get_links_entered_in_exps(mycursor)
    exp_data_query_string = "SELECT * FROM serp_shared.user_data where user_id in " + users_str;
    mycursor.execute(exp_data_query_string)
    results = mycursor.fetchall()
    age_list = []
    gender_dict = {'female':0,'male':0,'other':0}
    education_level_dict = {'high':0,'low':0}

    for x in results:
        age = x[1]
        gender = x[2]
        education_level = x[3]
        if education_level == "Master's degree" or education_level == "Bachelor's degree":
            education_level_dict['high'] += 1
        else:
            education_level_dict['low'] += 1
        age_list.append(int(age))
        gender_dict[gender] += 1

    num_users = len(age_list)
    print('---------------Num responses------------------')
    print('Num responses: ' + str(num_users) )
    print('---------------AGE------------------')
    print('Age min: ' + str(min(age_list)))
    print('Age max: ' + str(max(age_list)))
    print('Age mean: ' + str(np.mean(age_list)))
    print('Age stdev: ' + str(np.std(age_list)))

    print('---------------GENDER------------------')
    print('Men"' + str(gender_dict['male']/num_users))
    print('Women: ' + str(gender_dict['female']/num_users))
    print('Other: ' + str(gender_dict['other']/num_users))

    print('---------------EDUCATION_LEVEL------------------')
    print('higher_education:' + str(education_level_dict['high'] / num_users))
    print('lower education: ' + str(education_level_dict['low'] / num_users))


def print_ctr_all_stats():
    fname = get_filename('feedback_all_posterior_bias', None)
    click_info = {}
    with open(fname, newline='', encoding='utf8') as csvf:
        reader = csv.DictReader(csvf)
        for row in reader:
            series = row['sequence']
            click_info[series]= {}
            if series:
                click_info[series]['link1'] = int(row['link1'])
                click_info[series]['link2'] = int(row['link2'])
                if row['link1 + link2']:
                    click_info[series]['link1 + link2'] = int(row['link1 + link2'])

    print('---------------------First phsycal link ctr all----------------------')
    link1_no_ads = click_info['Y']['link1']+ click_info['M']['link1']+ click_info['N']['link1']
    link1_direct_ads = click_info['AY']['link1']+ click_info['AM']['link1']+ click_info['AN']['link1']
    link1_indirect_ads = click_info['SY']['link1']+ click_info['SM']['link1']+ click_info['SN']['link1']
    three_way_anova(link1_no_ads,click_info['link_visibility_no_ads']['link1'],'No Ads',
                    link1_direct_ads, click_info['link_visibility_A']['link1'],'Direct Marketing Ads',
                    link1_indirect_ads,  click_info['link_visibility_S']['link1'],'Indirect Marketing Ads')

    link_2_no_ads = click_info['Y']['link2']+ click_info['M']['link2']+ click_info['N']['link2']
    link2_direct_ads = click_info['AY']['link2'] + click_info['AM']['link2'] + click_info['AN']['link2']
    link2_indirect_ads = click_info['SY']['link2'] + click_info['SM']['link2'] + click_info['SN']['link2']

    print('------------------------Second phsycal link ctr all----------------------------------------------------')
    three_way_anova(link_2_no_ads, click_info['link_visibility_no_ads']['link2'], 'No Ads',
                    link2_direct_ads, click_info['link_visibility_A']['link2'], 'Direct Marketing Ads',
                    link2_indirect_ads, click_info['link_visibility_S']['link2'], 'Indirect Marketing Ads')
    print('------------------------First organic link ctr all----------------------------------------------------')
    three_way_anova(link1_no_ads,click_info['link_visibility_no_ads']['link1'],'No Ads',
                    link2_direct_ads, click_info['link_visibility_A']['link2'],'Direct Marketing Ads',
                    link2_indirect_ads, click_info['link_visibility_S']['link2'], 'Indirect Marketing Ads')


def get_click_info():
    fname = get_filename('feedback_all_posterior_bias', None)
    click_info = {}
    with open(fname, newline='', encoding='utf8') as csvf:
        reader = csv.DictReader(csvf)
        for row in reader:
            series = row['sequence']
            click_info[series] = {}
            if series:
                for i in range(1, 11):
                    click_info[series]['link' + str(i)] = int(row['link' + str(i)])
    return click_info




def print_ctr_all_links_stats(organic):
    click_info = get_click_info()

    for i in range(1,10):
        link_no_ads = click_info['Y']['link'+str(i)] + click_info['M']['link'+str(i)] + click_info['N']['link'+str(i)]
        if organic:
            link_direct_ads = click_info['AY']['link'+str(i+1)] + click_info['AM']['link'+str(i+1)] + click_info['AN']['link'+str(i+1)]
            link_indirect_ads = click_info['SY']['link'+str(i+1)] + click_info['SM']['link'+str(i+1)] + click_info['SN']['link'+str(i+1)]

        else:
            link_direct_ads = click_info['AY']['link' + str(i)] + click_info['AM']['link' + str(i)] + click_info['AN'][
                'link' + str(i)]
            link_indirect_ads = click_info['SY']['link' + str(i)] + click_info['SM']['link' + str(i)] + \
                                click_info['SN']['link' + str(i)]

        print('------------------------Link ' + str(i) +  ' ctr all----------------------------------------------------')
        if organic:
            three_way_anova(link_no_ads, click_info['link_visibility_no_ads']['link'+str(i)], 'No Ads',
                            link_direct_ads, click_info['link_visibility_A']['link'+str(i+1)], 'Direct Marketing Ads',
                            link_indirect_ads, click_info['link_visibility_S']['link'+str(i+1)], 'Indirect Marketing Ads')
        else:
            three_way_anova(link_no_ads, click_info['link_visibility_no_ads']['link'+str(i)], 'No Ads',
                            link_direct_ads, click_info['link_visibility_A']['link'+str(i)], 'Direct Marketing Ads',
                            link_indirect_ads, click_info['link_visibility_S']['link'+str(i)], 'Indirect Marketing Ads')


def print_ctr_ads_links_stats(prefix):
    click_info = get_click_info()

    for i in range(1,10):
        links_Y = click_info[prefix + 'Y']['link'+str(i)]
        links_M = click_info[prefix + 'M']['link'+str(i)]
        links_N = click_info[prefix + 'N']['link'+str(i)]
        print('------------------------Link ' + str(i) +  ' ctr all----------------------------------------------------')
        three_way_anova(links_Y, click_info['link_visibility_'+prefix+'Y']['link'+str(i)], 'Y',
                        links_M, click_info['link_visibility_'+prefix+'M']['link'+str(i)], 'M',
                        links_N, click_info['link_visibility_'+prefix+'N']['link'+str(i)], 'N')


def pairwise_rank_configuration_ads(prefix):
    click_info = get_click_info()
    for bias in ['Y', 'M', 'N']:
        for i in range(1,4):
            print(prefix + bias + ': link' + str(i) + ' - link ' + str(i+1))
            l1 = click_info[prefix + bias]['link'+str(i)]
            l2 = click_info[prefix + bias]['link'+str(i+1)]

            paired_ttest(l1, click_info['link_visibility_'+prefix+bias]['link'+str(i)],
                             l2, click_info['link_visibility_'+prefix+bias]['link'+str(i+1)])

def pairwise_rank_configuration_no_ads():
    click_info = get_click_info()
    for i in range(1,4):
        print('No Ads : link' + str(i) + ' - link ' + str(i+1))
        l1 = click_info['Y']['link'+ str(i)] + click_info['M']['link'+ str(i)] + click_info['N']['link'+ str(i)]
        l2 = click_info['Y']['link'+ str(i+1)] + click_info['M']['link'+ str(i+1)] + click_info['N']['link'+ str(i+1)]

        paired_ttest(l1, click_info['link_visibility_no_ads']['link'+str(i)],
                         l2, click_info['link_visibility_no_ads']['link'+str(i+1)])

def response_anova():
    fname = get_filename('feedback_all_posterior_bias', None)
    answers = {x: {} for x in LATEX_TABLE_ORDER}
    normed_answers = {x: {} for x in LATEX_TABLE_ORDER}
    with open(fname, newline='', encoding='utf8') as csvf:
        reader = csv.DictReader(csvf)
        for row in reader:
            seq = row['sequence']
            if seq not in LATEX_TABLE_ORDER:
                break
            for x in ['Y','M','N']:
                answers[seq][x] = int(row[x])
        for seq in LATEX_TABLE_ORDER:
            total = sum(answers[seq].values())
            for x in ['Y', 'M', 'N']:
                normed_answers[seq][x] = [1] * answers[seq][x] + [0] * (total - answers[seq][x])

    for bias in ['Y','M','N']:
        for answer in ['Y','M','N']:
            g1 = bias
            g2 = 'A'+bias
            g3 = 'S'+bias
            print('----------------------------------------')
            print('Bias: ' + bias + ', answer: ' + answer )
            print('----------------------------------------')
            three_way_anova_from_array(normed_answers[g1][answer],g1,normed_answers[g2][answer],g2,normed_answers[g3][answer],g3,answer+'_answers')

def get_response_array(fname):
    fname = get_filename(fname, None)
    answers = {x: {} for x in LATEX_TABLE_ORDER}
    normed_answers = {x: {} for x in LATEX_TABLE_ORDER}
    with open(fname, newline='', encoding='utf8') as csvf:
        reader = csv.DictReader(csvf)
        for row in reader:
            seq = row['sequence']
            if seq not in LATEX_TABLE_ORDER:
                break
            for x in ['Y','M','N']:
                answers[seq][x] = int(row[x])
        for seq in LATEX_TABLE_ORDER:
            if not answers[seq].values():
                continue
            total = sum(answers[seq].values())
            for x in ['Y', 'M', 'N']:
                normed_answers[seq][x] = [1] * answers[seq][x] + [0] * (total - answers[seq][x])
    return normed_answers


def compare_all_queries():
    normed_answers = {}

    normed_answers['Melatonin'] = get_response_array('feedback_all_posterior_bias_Melatonin')
    normed_answers['Omega'] = get_response_array('feedback_all_posterior_bias_Omega')
    normed_answers['Ginko'] = get_response_array('feedback_all_posterior_bias_Ginko')

    for (q1,q2) in [('Omega','Melatonin'), ('Omega','Ginko'), ('Melatonin','Ginko')]:
        print('-----------------------------------------------')
        print(q1 + '-' + q2)
        print('-----------------------------------------------')
        for seq in LATEX_TABLE_ORDER:
            if seq[0] == 'S' and (q1 == 'Melatonin' or q2 == 'Melatonin'):
                continue
            for answer in ['Y', 'M', 'N']:
                print('seq' + ':' + seq + ', Answer:' + answer)
                a1 = normed_answers[q1][seq][answer]
                a2 = normed_answers[q2][seq][answer]
                do_ttest(a1,a2)


def do_ttest(a1,a2):
    ttest = ttest_ind(a1, a2, alternative='larger')
    if ttest[1] < 0.1:
        print('larger')
        print('t(' + str(ttest[2]) + ')=' + str(ttest[0]) + 'p=' + str(ttest[1]))
    ttest = ttest_ind(a2, a1, alternative='larger')
    if ttest[1] < 0.1:
        print('smaller')
        #print(ttest)
        print('t('+str(ttest[2])+')='+str(ttest[0])+ 'p='+str(ttest[1]))


def response_ttest(fname):
    normed_answers = get_response_array(fname)
    for bias in ['Y','M','N']:
        for cmp in ['A','S','AS']:
            if 'Melatonin' in fname and 'S' in cmp:
                continue
            for answer in ['Y', 'M', 'N']:
                if len(cmp) == 1:
                    print( bias + '-' + cmp[0] + bias + ', Answer:' + answer)
                    a1= normed_answers[bias][answer]
                    a2 = normed_answers[cmp[0] + bias][answer]
                else:
                    print(cmp[0] + bias + '-' + cmp[1] + bias + ', Answer:' + answer)
                    a1 = normed_answers[cmp[0]+bias][answer]
                    a2 = normed_answers[cmp[1]+bias][answer]
                do_ttest(a1,a2)


def response_ttest_between_answers(fname):
    normed_answers = get_response_array(fname)
    for config in ['Y', 'AY', 'SY', 'M', 'AM', 'SM', 'N', 'AN', 'SN']:
        print('-----------------------------------------------')
        print(config)
        print('-----------------------------------------------')
        for cmp in [('Y','M'),('Y','N'),('M','N')]:
            print(config + ':' + cmp[0] + '-' + cmp[1])
            a1 = normed_answers[config][cmp[0]]
            a2 = normed_answers[config][cmp[1]]
            t = scipy.stats.ttest_rel(a1, a2)
            s = t.statistic
            p = t.pvalue
            if p < 0.05:
                print('s = ' + str(s) + ' p = ' + str(p))


def response_anova(fname):
    normed_answers = get_response_array(fname)
    for config in ['Y','AY','SY','M','AM','SM','N','AN','SN']:
        print('-----------------------------------------------')
        print( config)
        print('-----------------------------------------------')
        y= normed_answers[config]['Y']
        m= normed_answers[config]['M']
        n= normed_answers[config]['N']
        three_way_anova_from_array(y, 'Y', m, 'M', n, 'N', 'Answers')


def response_anova_between_viewpoints(fname):
    normed_answers = get_response_array(fname)
    for bias in ['Y','M','N']:
        for answer in ['Y', 'M', 'N']:
            print( 'Bias:' + bias + ', Answer:' + answer)
            no_ads= normed_answers[bias][answer]
            direct_marketing = normed_answers['A' + bias][answer]
            indirect_marketing = normed_answers['S' + bias][answer]
            three_way_anova_from_array(no_ads, 'No Ads', direct_marketing, 'direct Marketing', indirect_marketing, 'Indirect Marketing', 'Answers')

def encode_answers(df):
    return [ANSWERS_MAP[x] for x in list(df)]


def cohens_kappas():
    fname = get_filename('ctr_decision_correlation', None)
    df = pd.read_csv(fname)
    print('overall agreement organic')
    feedback = encode_answers(df['feedback'])
    organic = encode_answers(df['ctr_expected_organic'])
    k = cohen_kappa_score(feedback,organic)
    print(k)
    print('overall agreement all')
    all = encode_answers(df['ctr_expected_phys'])
    k = cohen_kappa_score(feedback,all)
    print('All queries per config')
    print_kappa_per_all_configs(df)
    for query in ['Does Omega Fatty Acids treat Adhd','Does Ginkgo Biloba treat tinnitus','Does Melatonin  treat jetlag']:
        print('---------------------------------------------------')
        print(query)
        print('---------------------------------------------------')
        dfq = df.query("query == '" + query + "'")
        print_kappa_per_all_configs(dfq)


def print_kappa_per_all_configs(df):
    for config in PRINT_ORDER:
        query = "config == '" + config + "'"
        print_kappa_for_config(df, query)


def print_kappa_for_config(df, query):
    print(query)
    df2 = df.query(query)
    feedback = encode_answers(df2['feedback'])
    print('organic')
    organic = encode_answers(df2['ctr_expected_organic'])
    k = cohen_kappa_score(feedback, organic)
    print(k)
    print('overall agreement all')
    all = encode_answers(df2['ctr_expected_phys'])
    k = cohen_kappa_score(feedback, all)
    print(k)


def num_links_pressed():
    fname = get_filename('user_behaviour', None)
    num_clicks = {}
    with open(fname, newline='', encoding='utf8') as csvf:
        reader = csv.DictReader(csvf)
        for row in reader:
            seq = row['sequence']
            if seq[0] == 'A' or seq[0] == 'S':
                config = seq[:2]
            else:
                config = seq[0]
            if config not in num_clicks:
                num_clicks[config] = []
            num_clicks[config].append(int(row['num_links_pressed']))

    for k,v in num_clicks.items():
        print(k)
        #print(v)
        print(np.mean(v))

    low_contrast = num_clicks['SY'] + num_clicks['SM']
    high_contrast  = num_clicks['SN']
    no_ads  = num_clicks['Y'] + num_clicks['M']+ num_clicks['N']
    ttest = ttest_ind(low_contrast, high_contrast)
    print(ttest)
    ttest = ttest_ind(no_ads, high_contrast)
    print(ttest)
    ttest = ttest_ind(num_clicks['AY'], num_clicks['SY'])
    print(ttest)
    ttest = ttest_ind(num_clicks['M'], num_clicks['SM'])
    print(ttest)

def print_ttests(print_order = PRINT_ORDER):
    fname = get_filename('feedback_all_posterior_bias', None)
    answers = {x:{} for x in print_order}

    with open(fname, newline='', encoding='utf8') as csvf:
        reader = csv.DictReader(csvf)
        for row in reader:
            seq = row['sequence']
            if seq not in print_order:
                break
            for x in ['Y','M','N']:
                answers[seq][x] = int(row[x])

        for seq in print_order:
            s = '\\textbf{' + seq+'}&'
            total = sum(answers[seq].values())
            df = total - 1
         #   s +=str(df)+'&'
            for cmp in [('Y', 'M'), ('Y', 'N'), ('M', 'N')]:
                t = paired_ttest(answers[seq][cmp[0]], total, answers[seq][cmp[1]], total, print_res=False)
                st= "{0:0.2f}".format(t.statistic)
                if t.pvalue < 0.01:
                     p = '\checkmark \checkmark'
                   # p = 'p < 0.01'
                elif t.pvalue <= 0.05:
                     p = '\checkmark '
                  #  p = 'p < 0.05'
#                elif t.pvalue ==  0.05:
#                    print('X ')
                 #   p = 'p = 0.05'
                else:
                    p = 'X'
                s += p + '&'
                #    p = 'p > 0.05'
                #s += 't(' + str(df) + ')=' + st +',' + p + '&'
               # s += 't=' + st +',' + p + '&'

            print(s[:-1]+'\\\\')
            print('\hline')

def print_feedback_table_with_ttest(print_order = PRINT_ORDER):
    fname = get_filename('feedback_all_posterior_bias', None)
    answers = {x:{} for x in print_order}
    normed_answers = {x:{} for x in print_order}

    with open(fname, newline='', encoding='utf8') as csvf:
        reader = csv.DictReader(csvf)
        for row in reader:
            seq = row['sequence']
            if seq not in print_order:
                break
            for x in ['Y','M','N']:
                answers[seq][x] = int(row[x])
        for seq in print_order:
            total = sum(answers[seq].values())
            for x in ['Y','M','N']:
                normed_answers[seq][x] = answers[seq][x]/total


        for seq in print_order:
            s = '\\textbf{' + seq+'}&'
            for x in ['Y', 'M', 'N']:
                normed = "{0:0.2f}".format(normed_answers[seq][x])
                s += str(answers[seq][x])+'('+ normed +')'+'&'
            df = sum(answers[seq].values()) - 1
            s +=str(df)+'&'
            for cmp in [('Y', 'M'), ('Y', 'N'), ('M', 'N')]:
                t = paired_ttest(answers[seq][cmp[0]], total, answers[seq][cmp[1]], total, print_res=False)
                st= "{0:0.2f}".format(t.statistic)
                if t.pvalue < 0.01:
                    p = 'p < 0.01'
                elif t.pvalue < 0.05:
                    p = 'p < 0.05'
                elif t.pvalue ==  0.05:
                    p = 'p = 0.05'
                else:
                    p = 'p > 0.05'
                s += st+','+p+'&'
            print(s[:-1]+'\\\\')
            print('\hline')


if __name__ == "__main__":
    #heatmap_per_answer_order()
    #ttest_from_count()
    #response_ttest('feedback_all_posterior_bias_Omega')
    #print_feedback_table_with_ttest()
    #print_ttests()
    #response_ttest_between_answers('feedback_all_posterior_bias')
    response_ttest_between_answers('feedback_all_posterior_bias_Ginko')
    #response_ttest_between_answers('feedback_all_posterior_bias_Xclude Melatonin')
    #response_anova('feedback_all_posterior_bias')
    #response_anova('feedback_all_posterior_bias_Melatonin')
    #response_anova('feedback_all_posterior_bias_Ginko')
    #response_anova('feedback_all_posterior_bias_Omega')


    #response_ttest('feedback_all_posterior_bias_Melatonin')
    #response_ttest('feedback_all_posterior_bias_Ginko')
    #response_ttest('feedback_all_posterior_bias_Omega')


    #response_ttest('feedback_all_posterior_bias')
    #chi_cont([[26, 20], [14+9, 29]], row_index=['M', 'SM'], col_index=['M', 'Y + N'])
    #num_links_pressed()
    #pairwise_rank_configuration_no_ads()
    #pairwise_rank_configuration_ads(prefix = 'A')
    #pairwise_rank_configuration_ads(prefix = 'S')
    #print_ctr_ads_links_stats(prefix = 'S')
    #print_ctr_all_links_stats(organic = True)
    #print_ctr_all_stats()

    #get_response_stats()
    #get_user_stats()
#    cohens_kappas()

    #compare_all_queries()

    #response_ttest('feedback_all_posterior_bias_Melatonin')
    #response_ttest('feedback_all_posterior_bias_Ginko')
    #response_ttest('feedback_all_posterior_bias_Omega')
    #response_ttest('feedback_all_posterior_bias_Xclude Melatonin')
    #ttest_from_count(c1=13, t1=32, c2=29, t2=56)
    #response_anova()
#
    #chi_cont([[8, 15], [16, 19]], row_index=['Y', 'AY'],col_index=['1','2'])

    #chi_cont([[52, 24,21], [53, 35,11],[42, 33,20]], row_index=['Y','AY','SY'])
    #chi_cont([[31, 45,22], [30, 45,22],[40, 40,20]], row_index=['M','AM','SM'])
    #chi_cont([[27, 34, 34], [21, 34,44],[23, 25,48]], row_index=['N','AN','SN'])
    #results_chi_stats()
    #chi_cont([[36,36+28], [50,50]], row_index=['N', 'SN'], col_index=['N', 'M + Y'])
    #chi_cont([[36,36+28], [44,56]], row_index=['N', 'AN'], col_index=['N', 'M + Y'])
    #chi_cont([[53,47], [42,58]], row_index=['Y', 'SY'], col_index=['Y', 'M + N'])
    #chi_cont([[38,7,3], [162,53,37]], row_index=['r1', 'r2'], col_index=['c1', 'c2','c3'])



    #ttest_from_count(c1=11, t1=32, c2=33, t2=172)

   # chi_cont([[43,51+34], [15,36+26]], row_index=['M', 'SM'], col_index=['Y', 'N+M'])
   # chi_cont([[34,51+43],[26,36+15]], row_index=['SM', 'M'], col_index=['N', 'Y+M'])
   # chi_cont([[51,34+43],[36,26+15]], row_index=['SM', 'M'], col_index=['M', 'Y+N'])
   # chi_cont([[15,36,26],[43,51,34]], row_index=['M', 'SM'], col_index=['Y', 'M','N'])
   # chi_cont([[19,25,65],[20,51,34]], row_index=['M', 'SM'], col_index=['Y', 'M','N'])

# res = chi2_contingency([[16,20,20],[11,18,23]])
    #kendel_tau('ctr_viewpoint_S')
    #ttest_from_count(c1=11, t1=32, c2=4, t2=60)
  #  logistic_regression(f = 'user_response_labels_A', mode='seq')
  #  logistic_regression(f = 'user_response_labels_A', mode='clicks')
    #print('NO ADS')
    #logistic_regression(f = None, group= 'O', mode='clicks')
    #print('A')
    #logistic_regression(f = None, group = 'A', mode='clicks')
    #print('S')
    #logistic_regression(f = None, group = 'S', mode='clicks')
#    print('#N/N-SN')






    #print(res)
    #chi_cont([[0.34*70, (1-0.34)*70], [35, 35]], row_index=['N', 'SN'], col_index=['N', 'Y+M'])
    #chi_cont([[36,  20], [29, 29]], row_index=['N', 'SN'], col_index=['N', '!N'])
    #chi_cont([[16, 20, 20], [14, 15, 29]], row_index=['N', 'SN'], col_index=['Y', 'M', 'N'])
    #chi_cont([[20, 20 + 16], [40, 40]], row_index=['N', 'SN'], col_index=['N', 'Y+M'])
    #chi_cont([[0.34*80, 0.66*80], [0.5*80, 0.5*80]], row_index=['N', 'SN'], col_index=['N', 'Y+M'])

    #chi_cont([[18, 26 + 13], [13, 12 + 7]], row_index=['M', 'SM'], col_index=['Y', 'N+M'])
  #  chi_cont([[17, 20 + 10], [35, 25]], row_index=['N', 'SN'], col_index=['N', 'Y+M'])
    #cont_exp(70)

    #two_by_two_test()
    #odds_ratio_tests()
    #threeway_log_linear_models()
    #chi_user_bias()
    #chi_all()
    #chi_negative_bias()
    #chi_inconclusive_bias()
    #chi_inconclusive_bias()
    #chi_no_ads()
    #chi_tests()
    #query_chi()
    #chi_no_melatonin()
    #log_linear_models(sequences = ['N','SN'])
    #log_linear_models(sequences = ['M','SM'])
    #log_linear_models()
    #ttest_from_count(c1=7, t1=33, c2=20, t2=49)
   # pd.set_option("display.max_rows", None, "display.max_columns", None)
    #decision_diff()
    #multivariate_log_learn(c1_c2_group_index = 'M_SM', y1=17, m1=23, n1=12, y2=13, m2=12, n2=7)
#    print('PHYS')
    #link1_phys_ctr_test()
    #print('ORG')
    #decision_diff()
#    first_two_results_ctr()
    #first_organic_result_test()
    #print('ad_first_organic_result_test')
    #ad_first_organic_result_test()
    #SM-M




    #anova_from_file(BEHAVIOUR_FILE,'num_links_pressed', 'feedback','links_per_feedback.csv')