import csv
from math import log
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from numpy.core import mean

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


def three_way_anova(c1,t1,g1,c2,t2,g2,t3,c3,g3):
    pd.set_option("display.max_rows", None, "display.max_columns", None)

    a1 = [1]*c1 + [0]*(t1-c1)
    a2 = [1]*c2 + [0]*(t2-c2)
    a3 = [1]*c3 + [0]*(t3-c3)

    d = []
    for g, a in {g1: a1, g2: a2, g3: a3}.items():
        for x in a:
            d.append({'CTR': x, 'group': g})

    # df = pd.DataFrame({'score':[],'group': np.repeat(['a1', 'a2', 'a3'], repeats=10)})
    df = pd.DataFrame(d)
    model = ols('CTR ~ group', data=df).fit()
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
    print(df)
    tukey = pairwise_tukeyhsd(endog=df['CTR'],
                              groups=df['group'],
                              alpha=0.05)

    print(tukey)

def ttest_from_count(c1,t1,c2,t2):
    a1 = [1]*c1 + [0]*(t1-c1)
    a2 = [1]*c2 + [0]*(t2-c2)
    #ttest = scipy.stats.ttest_ind(a1,a2)
   # if ttest.pvalue < 0.05:
   #     print ('TTest significant, p < 0.05')
   # else:
   #     print('TTest NOT significant, p >= 0.05')
    ttest = ttest_ind(a1, a2)
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
    F = np.array(results)


   # ddsr, p = fisher_exact(F, alternative='two-sided')
   # print('Fisher:'+ str(p))
    df = pd.DataFrame(data=F, index=row_index, columns=col_index)


    table = sm.stats.Table(df)


  #  print('table.cumulative_oddsratios')
  #  print(table.cumulative_oddsratios)

    print('table.chi2_contribs')
    print(table.chi2_contribs)


    #print('table.local_oddsratios')
    #print(table.local_oddsratios)
  #  print(table)



#    print('resid_pearson Residuals')


    print(table)
    print('resid_pearson Residuals')
    print(table.resid_pearson)
    print('standardized_resids Residuals')
    print(table.standardized_resids)

    obs = np.array(results)
    t = chi2_contingency(obs)



    print('test statistics:' + str(t[0]))
    print('p-value:' + str(t[1]))
    print('df ' + str(t[2]))
    print('expected' + str(t[3]))


    r = power_divergence(obs, lambda_="log-likelihood")
    print('power_divergence')
    print(r)

    if len(col_index) == 2:
        t = sm.stats.Table2x2(np.array(df))
        print(t.summary())

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

    clf = LogisticRegression( solver='saga', penalty='l1', max_iter=400).fit(X, y)
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
    avg = mean(vals)
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

    groups = ['No Ads, Y', 'No Ads, M', 'No Ads, N',
              'DM, Y', 'DM, M', 'DM, N',
              'IM, Y', 'IM, M', 'IM, N']
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

if __name__ == "__main__":
    #ttest_from_count(c1=11, t1=32, c2=4, t2=60)
    #ttest_from_count(c1=11, t1=32, c2=33, t2=172)
    heatmap()
    #chi_cont([[43,51+34], [15,36+26]], row_index=['M', 'SM'], col_index=['Y', 'N+M'])
    #chi_cont([[43,51+34], [15,36+26]], row_index=['M', 'SM'], col_index=['Y', 'N+M'])
    #chi_cont([[34,51+43],[26,36+15]], row_index=['SM', 'M'], col_index=['N', 'Y+M'])
    #chi_cont([[51,34+43],[36,26+15]], row_index=['SM', 'M'], col_index=['M', 'Y+N'])
    #chi_cont([[15,36,26],[43,51,34]], row_index=['M', 'SM'], col_index=['Y', 'M','N'])
    #chi_cont([[19,25,65],[20,51,34]], row_index=['M', 'SM'], col_index=['Y', 'M','N'])

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