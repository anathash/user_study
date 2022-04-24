# importing the required module
import csv
from collections import Counter

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from scipy.interpolate import make_interp_spline

GRAPH_DIR = 'C:\\research\\falseMedicalClaims\\user study\\SIGIR ads\\'
REPORTS_DIR = '../resources/reports/'

color_cycle = ['blue','orange','green']
markers = ['o','s','^']

def plot_ex():
    # x axis values
    x = [1, 2, 3]
    # corresponding y axis values
    y = [2, 4, 1]

    # plotting the points
    plt.plot(x, y)

    # naming the x axis
    plt.xlabel('x - axis')
    # naming the y axis
    plt.ylabel('y - axis')

    # giving a title to my graph
    plt.title('My first graph!')

    # function to show the plot
    plt.show()
    #plt.savefig('histogram.pdf')


def plot_ctr_old(input_file, title):
    plt.xticks([i for i in range(1,11)])
    series = []
    ys = []
    with open('../resources/reports/' + input_file + '.csv','r', newline='') as inputCSV:
        reader = csv.DictReader(inputCSV)
        for row in reader:
            series.append(row['series'])
            vals = list(row.values())
            if vals[10] == "":
                vals = vals[:-1]
            vals = vals[1:]
            vals = [float(x) for x in vals]
            ys.append(vals)
    for y in ys:
        x = [i for i in range(1, len(y)+1)]
        print(x)
        print(y)
        X_Y_Spline = make_interp_spline(x, y)

        # Returns evenly spaced numbers
        # over a specified interval.
        X_ = np.linspace(min(x), max(x), 500)
        Y_ = X_Y_Spline(X_)

        plt.plot(X_, Y_)
        #plt.plot(x, y, '-o')
        plt.scatter(x, y, marker='o');


    plt.legend(series)
    # giving a title to my graph
    plt.title(title)

    plt.xlabel('rank')
    # naming the y axis
    plt.ylabel('CTR')

    # giving a title to my graph
    plt.title(title)

    # function to show the plot
    plt.show()
    #plt.savefig(GRAPH_DIR + input_file+'_py3.pdf')


def matplot_def():
    matplotlib.use("pgf")
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'text.usetex': True,
        'pgf.rcfonts': False,
    })


def ads_bar_chart(f, title):
    df = pd.read_csv(REPORTS_DIR + f + '.csv')
    #print(df)
    vals = list(df.iloc[:, 0])
    #print(vals)
    total = len(vals)
    data = Counter(vals)
    print(data)
    dat_n = {x:y/total for x,y in data.items()}
    print(dat_n)
    scores = list(dat_n.keys())
    count = list(dat_n.values())

    #fig = plt.figure(figsize=(10, 5))

    # creating the bar plot
    plt.bar(scores, count, color='blue',
            width=0.4)

    #plt.xlabel(title)
    #plt.ylabel("No. of students enrolled")
    #plt.title("Students enrolled in different courses")
    plt.show()
    #plt.savefig(GRAPH_DIR + f+'_py.pdf')


def bar_charts_ads_effect():
    df = pd.read_csv(REPORTS_DIR + 'ads_effect.csv')
    X = [-2,-1,0,+1,+2]
    #print(df)
    ads_exp_effect = list(df['ad_exp_effect'])
    ads_dec_effect = list(df['ad_dec_effect'])
    vals = list(df.iloc[:, 0])
    #print(vals)
    total = len(ads_exp_effect)
    exp_c = Counter(ads_exp_effect)
    dec_c = Counter(ads_dec_effect)
    #print(exp_c)
    data_exp = {x:y/total for x,y in exp_c.items()}
    data_dec = {x:y/total for x,y in dec_c.items()}
    #print(data_exp)
    scores_exp = list(data_exp.keys())
    count_exp = list(data_exp.values())
    scores_dec = list(data_dec.keys())
    exp_sorted = dict(sorted(data_exp.items(), key=lambda kv: kv[0]))
    print('exp_sorted')
    print(exp_sorted)
    dec_sorted = dict(sorted(data_dec.items(), key=lambda kv: kv[0]))
    print('dec_sorted')
    print(dec_sorted)

    X_axis = np.arange(5) #5 point likert scale
    #fig = plt.figure(figsize=(10, 5))

    # creating the bar plot
    #plt.bar(X_axis - 0.2, scores_exp, count_exp, label = 'Experience Effect')
    #plt.bar(X_axis + 0.2, scores_dec, count_dec, label = 'Decision Effect')
    plt.bar(X_axis - 0.2, list(exp_sorted.values()), 0.4, label = 'Experience Effect')
    plt.bar(X_axis + 0.2, list(dec_sorted.values()), 0.4, label = 'Decision Effect')

    #plt.xlabel(title)
    #plt.ylabel("No. of students enrolled")
    #plt.title("Students enrolled in different courses")
    plt.xticks(X_axis, X)
   # plt.xlabel("Groups")
    #plt.ylabel("Number of Students")
    #plt.title("Number of Students in each group")
    plt.legend()
    #plt.show()
    plt.savefig(GRAPH_DIR + 'ads_effect.pdf')


def plot_ctr(input_file, title, mode):
    plt.xticks([i for i in range(1,11)])
    series = []
    plots = []

    with open('../resources/reports/' + input_file + '.csv','r', newline='') as inputCSV:
        reader = csv.DictReader(inputCSV)
        for row in reader:
            series.append(row['series'])
            vals = list(row.values())
            if len(vals) > 10 and vals[10] == "":
                vals = vals[:-1]
            vals = vals[1:]
            ys = []
            xs = []
            for i in range(0,len(vals)):
                if vals[i] != "":
                   ys.append(float(vals[i]))
                   xs.append(i+1)
            #vals = [float(x) for x in vals if x!=""]
            plots.append({'xs':xs, 'ys':ys})


    for  i in range (0,len(plots)):
        #x = [i for i in range(1, len(y)+1)]
        p = plots[i]
        x = p['xs']
        y = p['ys']
        print(x)
        print(y)
        X_Y_Spline = make_interp_spline(x, y)

        # Returns evenly spaced numbers
        # over a specified interval.
        X_ = np.linspace(min(x), max(x), 500)
        Y_ = X_Y_Spline(X_)

        plt.plot(X_, Y_, label=series[i], color=color_cycle[i])


        #plt.plot(x, y, '-o')
        plt.scatter(x, y, marker=markers[i],  color=color_cycle[i]);

    custom_lines = [Line2D([0], [0], color=color_cycle[0],marker= markers[0]),
                    Line2D([0], [0], color=color_cycle[1],marker= markers[1]),
                    Line2D([0], [0], color=color_cycle[2],marker= markers[2])]
    plt.legend(custom_lines, series)
    #plt.legend(handles=handles, fontsize='x-large')
    #    plt.legend(series, fontsize=16)
    # giving a title to my graph
    plt.title(title, fontsize=18)

    plt.xlabel('rank', fontsize=18)
    # naming the y axis
    plt.ylabel('CTR', fontsize=18)

    # giving a title to my graph
    #plt.title(title)

    # function to show the plot
    if mode == 'show':
        plt.show()
    if mode == 'save':
        plt.savefig(GRAPH_DIR + input_file+'.pdf')



if __name__ == "__main__":
    #bar_charts_ads_effect()
    #ads_bar_chart('ad_exp_effect', 'Ads Effect on Users Experience')
    #ads_bar_chart('ad_exp_effect_a', 'ad_exp_effect_a')
    #ads_bar_chart('ads_exp_s', 'ads_exp_s')



    #ads_bar_chart('ad_dec_effect_A', 'ad_dec_effect_A')
    #ads_bar_chart('ad_dec_effect_S', 'ad_dec_effect_S')
    #ads_bar_chart('ad_dec_effect_all', 'Ads Effect on Users Decision Making')
    #plot_ex()
    #plot_ctr('ctr_viewpoint_A','Direct Marketing Ads')
    #plot_ctr('ctr_viewpoint_S','Indirect Marketing Ads')
    #plot_ctr('ctr_viewpoint_no_ads','No Ads')
    #plot_ctr('ctr_all','CTR per ad configuration', mode='show')
    #plot_ctr('ctr_direct_marketing','Direct Marketing CTR', mode='show')
    plot_ctr('ctr_indirect_marketing','Indirect Marketing CTR', mode='show')
    #plot_ctr('ctr_sm_sn','IM-IN')
