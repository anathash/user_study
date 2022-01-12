import csv
import scipy

import scipy.stats as stats

import pandas as pd
from scipy.stats import chisquare


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

def three_way_anova(c1,t1,c2,t2,t3,c3):
    a1 = [1]*c1 + [0]*(t1-c1)
    a2 = [1]*c2 + [0]*(t2-c2)
    a3 = [1]*c3 + [0]*(t3-c3)
    F, p = stats.f_oneway(a1,a2,a3)
    if p < 0.05:
        print ('p < 0.05')
    print('p=' + str(p))
    print('F= ' + str(F))

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
    three_way_anova(c1 = 91,t1=151,c2=11,t2=105,c3=24,t3=65)
    print('ttest no ads - ads')
    ttest_from_count(c1 = 91,t1=151,c2=11,t2=105)
    print('ttest no ads - sponsored')
    ttest_from_count(c1 = 91,t1=151,c2=24,t2=65)
    print('ttest ads - sponsored')
    ttest_from_count(c1 = 11,t1=105,c2=24,t2=65)


def first_organic_result_test():
    o1_clicks_noads = 91
    o1_link_vis = 151

    o1_ads_clicks = 58
    o1_ads_link_vis = 105

    o1_s_clocks =  44
    o1_s_vis = 65

    print('anova')
    three_way_anova(c1 = o1_clicks_noads,t1=o1_link_vis,c2=o1_ads_clicks,t2=o1_ads_link_vis,c3=o1_s_clocks,t3=o1_s_vis)
    print('ttest no ads - ads')
    ttest_from_count(c1 = o1_clicks_noads,t1=o1_link_vis,c2=o1_ads_clicks,t2=o1_ads_link_vis)
    print('ttest no ads - sponsored')
    ttest_from_count(c1 = o1_clicks_noads,t1=o1_link_vis,c2=o1_s_clocks,t2=o1_s_vis)
    print('ttest ads - sponsored')
    ttest_from_count(c1 = o1_ads_clicks,t1=o1_ads_link_vis,c2=o1_s_clocks,t2=o1_s_vis)


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

if __name__ == "__main__":
    print('PHYS')
    link1_phys_ctr_test()
    print('ORG')
    first_organic_result_test()
    print('ad_first_organic_result_test')
    ad_first_organic_result_test()
    #SM-M
    #t = chisquare([7, 7, 17,2], f_exp=[10, 20, 17,2])
    #print(t)
    #ttest_from_count(c1=7,t1=19,c2=12,t2=16)


    #anova_from_file(BEHAVIOUR_FILE,'num_links_pressed', 'feedback','links_per_feedback.csv')