import csv
import scipy.stats as stats

import pandas as pd

from user_behaviour_table import BEHAVIOUR_FILE


def anova_from_file(fname, metric_field, group_by):
    df = pd.read_csv(fname)
    result = df.groupby(group_by)[metric_field].apply(list)
    print(result)
    F, p = stats.f_oneway(*result)
    print(p)
    print(F)


anova_from_file(BEHAVIOUR_FILE,'num_links_pressed', 'feedback','links_per_feedback.csv')