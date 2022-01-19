import csv
from math import log

import numpy
import scipy
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression, LinearRegression
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import numpy as np
import scipy.stats as stats

import pandas as pd
from scipy.stats import chisquare, chi2_contingency


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
    ttest = scipy.stats.ttest_ind(a1,a2)
    if ttest.pvalue < 0.05:
        print ('TTest significant, p < 0.05')
    else:
        print('TTest NOT significant, p >= 0.05')

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


def chi_cont(r1,r2):
    results = [r1, r2]
    F = np.array(results)
    table = sm.stats.Table(F)
    print(table)

    obs = np.array(results)
    t = chi2_contingency(obs)
    print('test statistics:' + str(t[0]))
    print('p-value:' + str(t[1]))
    print('df ' + str(t[2]))
    print('expected' + str(t[3]))

def chi_no_ads():
    results = [[26,13,10], [17,23,12],[10,20,17]]
    F = np.array(results)
    table = sm.stats.Table(F)
    print(table.resid_pearson)
    obs = np.array(results)
    t = chi2_contingency(obs)
    print('test statistics:' + str(t[0]))
    print('p-value:' + str(t[1]))
    print('df' + str(t[2]))
    print('expected' + str(t[3]))

def chi_tests():
    sn_norm = [0.225,0.225,0.55]
    n_norm =  [0.22,0.42,0.36]
    dummy_sn = [x*80 for x in sn_norm]
    dummy_n = [x * 70 for x in n_norm]
    print('SN-N')
    chi_cont([10, 20, 17], [7, 7, 17])
    #chi_cont([14,  17], [37,  10])
    print('SN-N-dummy')
    chi_cont([10, 20, 17], dummy_sn)
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
    reg = LinearRegression().fit(X,Y )
    print(str(reg.intercept_))
    print( str(reg.coef_))

    X = sm.add_constant(X)  # adding a constant

    model = sm.OLS(Y, X).fit()
    print(model.params)
    print(model.summary())

if __name__ == "__main__":
    #chi_no_ads()
    chi_tests()
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

    #ttest_from_count(c1=7,t1=19,c2=12,t2=16)


    #anova_from_file(BEHAVIOUR_FILE,'num_links_pressed', 'feedback','links_per_feedback.csv')