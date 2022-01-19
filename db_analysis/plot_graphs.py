# importing the required module
import csv
from collections import Counter

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import make_interp_spline

GRAPH_DIR = 'C:\\research\\falseMedicalClaims\\user study\\SIGIR ads\\'
REPORTS_DIR = '../resources/reports/'

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


def plot_ctr(input_file, title):
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
    #plt.show()
    plt.savefig(GRAPH_DIR + input_file+'_py2.pdf')


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

if __name__ == "__main__":
    #ads_bar_chart('ad_exp_effect', 'Ads Effect on Users Experience')
    #ads_bar_chart('ad_exp_effect_a', 'ad_exp_effect_a')
    #ads_bar_chart('ads_exp_s', 'ads_exp_s')



    #ads_bar_chart('ad_dec_effect_A', 'ad_dec_effect_A')
    #ads_bar_chart('ad_dec_effect_S', 'ad_dec_effect_S')
    #ads_bar_chart('ad_dec_effect_all', 'Ads Effect on Users Decision Making')
    #plot_ex()
    plot_ctr('ctr_all','CTR')
    #plot_ctr('ctr_am_an','AM-AN')
    #plot_ctr('ctr_sm_sn','SM-SN')
