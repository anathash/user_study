import csv
import math
from collections import Counter
import scipy.stats as stats
import statsmodels

from statsmodels.stats.weightstats import ttest_ind

from db_analysis.utils import connect_to_db, get_time_diff, get_time_diff_from_actions, \
    TREATMENT_CORRECT_ANSWERS, CONDITION_CORRECT_ANSWERS, get_links_entered_by_worker, filter_user, get_links_stats, \
    string_to_datetime, get_server_url
from process_batch import get_worker_id_list, BATCH_FILE_PREFIX
#BEHAVIOUR_FILE = '../resources/reports//user_behaviour_limit_5.csv'

def write_to_file(exp_data, filtered_users, limit):

    if limit:
        filename = '../resources/reports//user_behaviour_limit_'+str(limit)+'.csv'
    else:
        filename = '../resources/reports//user_behaviour.csv'

    with open(filename, 'w', newline='', encoding='utf8') as csvfile:
        fieldnames = ['exp_id', 'WorkerId', 'sequence','url','start_time', 'search_time_exp','search_time_actions',
                      'num_links_pressed','knowledge','feedback','reason','treatment_answer_correct','condition_answer_correct','comments',
                      'ad_exp_effect', 'ad_dec_effect', 'ad_comments',
                      'link1', 'link2', 'link3', 'link4', 'link5', 'link6', 'link7', 'link8', 'link9', 'link10',
                      'link_order1', 'link_order2', 'link_order3', 'link_order4', 'link_order5', 'link_order6', 'link_order7', 'link_order8', 'link_order9', 'link_order10',
                      'link1_time', 'link2_time', 'link3_time', 'link4_time', 'link5_time', 'link6_time', 'link7_time', 'link8_time', 'link9_time', 'link10_time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for query, sequences in exp_data.items():
            for seqeunce, entries in sequences.items():
                soreted_entries = sorted(entries.items(), key=lambda kv: kv[0], reverse=True) #sort newest_first
                if limit:
                    n = min(len(soreted_entries), limit)
                else:
                    n = len(soreted_entries)
                for i in range(0,n):
                    row = soreted_entries[i][1]
                    writer.writerow(row)
    filterf = '../resources/reports/filtered_users'
    if limit:
        filterf +='_limit_'+str(limit)+'.csv'
    else:
        filterf +='.csv'

    with open( '../resources/reports/filtered_users.csv', 'w', newline='', encoding='utf8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['WorkerId', 'Filter Reason'])
        writer.writeheader()
        for row in filtered_users:
            writer.writerow(row)

def process_results(mycursor, results,links,  link_times, link_orders, filter_users = True, add_time_diff_actions = False):
    exp_data = {}
    filtered_users = []
    for x in results:
        url = x[2][35:]
        start = x[7]
        end = x[8]
        exp_id = x[0]
        if end:
            time_diff_exp = get_time_diff(start, end)
        else:
            time_diff_exp = 0

        #        time_diff_actions, time_diff_exp = get_time_diff_from_actions(mycursor, x[1], time_diff_exp, end)
        if add_time_diff_actions:
            time_diff_actions = get_time_diff_from_actions(mycursor, x[1], time_diff_exp, end)

        # This means that the exp_id was regenerated - so we take the difference between the first clicked link and the end time. This bug should be fixed
        answer_treatment = x[12].strip()
        answer_condition = x[13].strip()
        query = x[5]
        sdt = string_to_datetime(start)
        if filter_users:
            filter_msg = filter_user(x[1], query, answer_treatment, answer_condition, start, end, x[9], x[11])
            if filter_msg:
                if 'test' not in filter_msg:
                    filtered_users.append({'WorkerId': x[1], 'Filter Reason': filter_msg})
                continue

        time_diff_exp = "%.4f" % round(time_diff_exp, 2)
        if add_time_diff_actions:
            time_diff_actions = "%.4f" % round(time_diff_actions, 2)
        else:
            time_diff_actions = 'omit'

        treatment_answer_correct = 1 if answer_treatment == TREATMENT_CORRECT_ANSWERS[query] else 0
        condition_answer_correct = 1 if answer_condition == CONDITION_CORRECT_ANSWERS[query] else 0
        worker_id = x[1]
        sequence = x[6]

        entry = {'exp_id': exp_id, 'WorkerId': worker_id, 'sequence': sequence, 'url': url, 'start_time': start,
                 'search_time_exp': time_diff_exp, 'search_time_actions': time_diff_actions,
                 'knowledge': x[9], 'feedback': x[10], 'reason': x[11],
                 'treatment_answer_correct': treatment_answer_correct,
                 'condition_answer_correct': condition_answer_correct, 'comments': x[14],
                 'ad_exp_effect':x[15], 'ad_dec_effect':x[16], 'ad_comments':x[17]}

        num_links = 0
        for i in range(1, 11):
            # link_pressed = links[exp_id][i]
            if worker_id in links:
                link_pressed = links[worker_id][i]
                entry['link' + str(i)] = link_pressed
                num_links += link_pressed
                entry['link' + str(i) + '_time'] = link_times[worker_id][i]
                entry['link_order' + str(i)] = link_orders[worker_id][i]

        entry['num_links_pressed'] = num_links
        if query not in exp_data:
            exp_data[query] = {}
        if sequence not in exp_data[query]:
            exp_data[query][sequence] = {}
        exp_data[query][sequence][sdt] = entry
    return exp_data, filtered_users

#TODO : filter by knowledge?? YES /NO/ ALL


def generate_user_behaviour_table(limit = None, from_batch = None, to_batch=None, local = True, filter_users = True, add_time_diff_actions = False):
    db = connect_to_db(local)
    mycursor = db.cursor()
    #links = get_links_entered_in_exps(mycursor)

    if from_batch:
        if not to_batch:
            to_batch = from_batch
        amazon_results, ids_sql_string = get_worker_id_list(from_batch, to_batch)
        exp_data_query_string = "SELECT * FROM serp.exp_data where user_id in " + ids_sql_string

    else:
        exp_data_query_string = "SELECT * FROM serp.exp_data"

    links, link_times, link_orders = get_links_stats(mycursor)
    mycursor.execute(exp_data_query_string)
    results = mycursor.fetchall()

    exp_data, filter_users = process_results(mycursor=mycursor, results = results,
                                             links=links,  link_times=link_times,
                                             link_orders=link_orders, filter_users = filter_users, add_time_diff_actions = add_time_diff_actions)

    write_to_file(exp_data=exp_data, filtered_users=filter_users, limit=limit)

def get_answer_count(mode = 'url', print_update_query = False, local  = True, prefix = None):
    server_url = get_server_url(local)
    db = connect_to_db(local)
    mycursor = db.cursor()
    answer_seq_dict = {}
    if prefix:
        query = 'SELECT URL FROM serp.config_data where sequence like "' + prefix + '%";'
    else:
        query = 'SELECT URL FROM serp.config_data;'
    mycursor.execute(query)
    myresult = mycursor.fetchall()
    for x in myresult:
        db_url = x[0]
        db_seq = db_url.split('SERP/')[1]
        if mode == 'seq':
            db_seq = db_seq[:-6]
        answer_seq_dict[db_seq] = 0


    with open(BEHAVIOUR_FILE, newline='', encoding='utf8') as csvf:
        reader = csv.DictReader(csvf)
        for row in reader:
            query_seq = row['url']
            if mode == 'seq':
                query_seq = query_seq[:-6]
            if query_seq not in answer_seq_dict:
                answer_seq_dict[query_seq] = 0
            answer_seq_dict[query_seq] += 1
    if mode == 'seq':
        fname = '../resources/reports/answers_per_query_seq.csv'
    else:
        fname = '../resources/reports/answers_per_url.csv'
    with open(fname, 'w', newline='') as csvfile:
        if mode == 'seq':
            writer = csv.DictWriter(csvfile, fieldnames = ['seq', 'query', 'count'])
        else:
            writer = csv.DictWriter(csvfile, fieldnames=['query_seq', 'count'])

        writer.writeheader()
        total =0
        for seq, count in answer_seq_dict.items():
            qs = seq.split('-')
            q = qs[0]
            s = qs[1]
            if mode == 'seq':
                writer.writerow({'seq': s,'query':q, 'count': answer_seq_dict[seq]})
            else:
                writer.writerow({'query_seq': seq, 'count': answer_seq_dict[seq]})
            if print_update_query:
                if not prefix or (prefix and s.startswith(prefix)):
                    total += count
                    print('UPDATE serp.config_data set used =0, answered = ' + str(count) + " where URL='"+server_url+seq+"';")
    print(total)


def group_behaviour(metric_field, group_by, output_file_name, csv=None):
    data = {}
    with open(BEHAVIOUR_FILE, newline='', encoding='utf8') as csvf:
        reader = csv.DictReader(csvf)
        for row in reader:
            metric_val = row[metric_field]
            group_by_value = row[group_by]
            if group_by_value not in data:
                data[group_by_value] = []
            data[group_by_value].append(metric_val)
    with open('../resources/reports/'+output_file_name,'w',newline='') as csvfile:
        filednames = list(data.keys())
        writer = csv.DictWriter(csvfile, filednames)
        writer.writeheader()

def get_filename(prefix, limit):
    fname = '../resources/reports//' + prefix
    if limit:
        fname += '_limit_'+str(limit)
    fname += '.csv'
    return fname

def get_sequence_score(seq, answer):
    score = 0
    for i in range(0,len(seq)):
        if answer == seq[i]:
            score += 1/(i+1)
    return score

def get_posterior(seq, answer, links):
    score = 0
    for i in range(0,len(seq)):
        if answer == seq[i]:
            score += (1/(i+1))*links[i]
    return score

def get_bucket(score, bucket_step = 1.0):
    b = score / bucket_step
    f = bucket_step * (math.floor(b))
    t = bucket_step * (math.ceil(b))
    if f == t:
        f -= bucket_step
    fp = "%.2f" % f
    ft = "%.2f" % t
    return fp + ' to ' + ft


def print_seq_to_answers(seq_to_answers, skip_ad, limit):
    prefix = 'seq_to_answer_skip_ad_' + str(skip_ad)
    fname = get_filename(prefix, limit)
    with open(fname,'w',newline='') as outcsv:
        fieldnames = ['sequence','Y_score','M_score','N_score','NS_score','Y','M','N','NS','Y_count','M_count','N_count','NS_count']
        writer = csv.DictWriter(outcsv, fieldnames)
        writer.writeheader()
        for seq, stats in seq_to_answers.items():
            sum_answers = sum([x['count'] for x in stats.values()])
            row = {'sequence': seq}
            for answer, answer_stats in stats.items():
                row.update({answer+'_score':answer_stats['score'],answer:answer_stats['count']/sum_answers,answer+'_count':answer_stats['count']})
            writer.writerow(row)


def get_links(row):
    links = []
    for i in range (1,11):
        links.append(row['link'+str(i)])
    return links


def sequence_score_to_answer(limit, buckets = True,  skip_ad = True):
    fname = get_filename('user_behaviour', limit)
    seq_to_answers = {}
    with open(fname, newline='', encoding='utf8') as csvf:
        reader = csv.DictReader(csvf)
        for row in reader:
            sequence = row['sequence']
            if sequence not in seq_to_answers:
                seq_to_answers[sequence] = {}
                for answer in ['Y','M','N','NS']:
                    if (sequence.startswith('A') or sequence.startswith('S')) and skip_ad:
                        score = get_sequence_score(sequence[1:], answer)
                    else:
                        score = get_sequence_score(sequence, answer)
                    seq_to_answers[sequence][answer]= {'score':score,'count':0}
            feedback = row['feedback']
            seq_to_answers[sequence][feedback]['count'] += 1
    if not buckets:
        print_seq_to_answers(seq_to_answers, skip_ad, limit)
        return
    buckets =  {'Y':{},'M':{},'N':{}}
    for seq, stats in seq_to_answers.items():
        for answer, answer_stats in stats.items():
            if answer == 'NS':
                continue
            score = answer_stats['score']
            bucket = score
            #bucket = get_bucket(score)
            if bucket not in buckets[answer]:
                buckets[answer][bucket] = {'Y':0,'M':0,'N':0,'NS':0}
            for a in ['Y','M','N','NS']:
                buckets[answer][bucket][a] += stats[a]['count']


    prefix = 'seq_score_to_answer_stats_' + str(skip_ad)
    fname = get_filename(prefix, limit)
    with open(fname, 'w', newline='') as outcsv:
        fieldnames = ['Y_score', 'M_score', 'N_score',  'Y', 'M', 'N', 'NS', 'Y_count',
                      'M_count', 'N_count', 'NS_count']
        writer = csv.DictWriter(outcsv, fieldnames)
        writer.writeheader()
        for answer, answer_scores_dict in buckets.items():
            for score, stats in answer_scores_dict.items():
                sum_answers = sum(stats.values())
                row = {answer + '_score': score}
                for a, count in stats.items():
                    row.update({a: count / sum_answers,
                                a + '_count': count})
                writer.writerow(row)


def is_add(seq):
    return  seq.startswith('S') or seq.startswith('A')


def sequence_score_to_answer_posterior(limit,   skip_ad = True):
    results = dict()
    fname = get_filename('user_behaviour', limit)
    with open(fname, newline='', encoding='utf8') as csvf:
        reader = csv.DictReader(csvf)
        for row in reader:
            config = row['sequence']
            if config not in results:
                results[config] = {'Y_score': 0, 'M_score': 0, 'N_score': 0, 'Y': 0, 'N': 0, 'M': 0, 'NS': 0, 'sum_answers': 0, 'links':[0]*10}
            answer = row['feedback']
            results[config][answer] += 1
            results[config]['sum_answers'] += 1

            for i in range(1, 11):
                link = 'link' + str(i)
                results[config]['links'][i-1] += int(row[link])

        for seq, stats in results.items():
            s = seq
            l = stats['links']
            if skip_ad and is_add(seq):
                s = seq[1:]
                l = l[1:]
            for answer in ['Y', 'M', 'N']:
                score = get_posterior(s, answer, l)
                stats[answer+'_score'] = score


        fname = get_filename('sequnce_posterior_stats', limit)
        with open(fname, 'w', newline='') as csvfile:
            fieldnames = ['sequence', 'Y_score', 'M_score', 'N_score', 'Y', 'M', 'N', 'NS','max_score','popular_answer','Y_count', 'M_count', 'N_count', 'NS_count', 'sum_answers']
            for i in range(1, 11):
                fieldnames.append('link' + str(i))

            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for config, stats in results.items():
                row = {'sequence': config}
                row['sum_answers'] = stats['sum_answers']
                for answer in ['Y', 'M', 'N']:
                    row[answer+'_score'] = stats[answer+'_score']
                    row[answer+'_count'] = stats[answer]
                    row[answer] = stats[answer]/stats['sum_answers']
                row['NS_count'] = stats['NS']
                row['NS'] = stats['NS']/stats['sum_answers']

                row['max_score'] = get_max_key({'Y': row['Y_score'], 'M': row['M_score'], 'N': row['N_score']})
                row['popular_answer'] = get_max_key({'Y': row['Y'], 'M': row['M'], 'N': row['N']})

                for i in range(1, 11):
                    row['link' + str(i)] = stats['links'][i-1]
                writer.writerow(row)


def get_max_key(dict):
    s = sorted(dict.items(), key=lambda kv: kv[1], reverse=True)
    return s[0][0]

def get_first_rank(seq, answer):
    for i in range(0, len(seq)):
        if answer == seq[i]:
            return 1 / (i + 1)


def sequence_score_to_answer_stats(limit=None, ad_prefix = None):
    fname = get_filename('user_behaviour', limit)
    seq_to_answers = {}
    with open(fname, newline='', encoding='utf8') as csvf:
        reader = csv.DictReader(csvf)
        for row in reader:
            sequence = row['sequence']
            feedback = row['feedback']
            if not sequence in seq_to_answers:
                seq_to_answers
            if ad_prefix:
                if (not sequence.startswith(ad_prefix)):
                    continue
                #score = get_sequence_score(sequence[1:], answer)
                score = get_first_rank(sequence[1:], answer)
            else:
                if sequence.startswith('S') or sequence.startswith('A'):
                    continue
                #score = get_sequence_score(sequence, answer)
                score = get_first_rank(sequence[1:], answer)
            #bucket = get_bucket(score)
            bucket = "%.2f" % score

            if bucket not in seq_to_answers:
                seq_to_answers[bucket] = {'Y':{},'M':{},'N':{}}
            for answer in ['Y','M','N','NS']:
                seq_to_answers[bucket][answer] += 1
    if not ad_prefix:
        ad_prefix = 'no_ads'
    prefix = 'seq_to_answer_'+ ad_prefix
    fname = get_filename(prefix, limit)
    with open(fname,'w',newline='') as outcsv:
        fieldnames = ['score','Y','M','N','NS','Y_count','M_count','N_count']
        writer = csv.DictWriter(outcsv, fieldnames)
        writer.writeheader()
        for s, stats in seq_to_answers.items():
            sum_answers = sum(stats.values())
            row = {'score': s, 'Y_count':stats['Y'], 'M_count':stats['M'], 'N_count':stats['N'],
                   'Y':stats['Y']/sum_answers, 'M':stats['M']/sum_answers, 'N':stats['N']/sum_answers}
            writer.writerow(row)


def sequence_scores(add_posterior = True, limit=None, skip_ads = True, print_latex = True):
    if add_posterior:
        posterior = {}
        with open('../resources/reports/posterior_bias.csv', newline='', encoding='utf8') as csvf:
            reader = csv.DictReader(csvf)
            for row in reader:
                sequence = row['sequence']
                posterior[sequence] = {'Y':row['Y'],'M':row['M'],'N':row['N']}

    fname = get_filename('user_behaviour', limit)
    seq_to_answers = {}
    with open(fname, newline='', encoding='utf8') as csvf:
        reader = csv.DictReader(csvf)
        for row in reader:
            sequence = row['sequence']
            if is_add(sequence):
                continue
            if sequence not in seq_to_answers and skip_ads:
                seq_to_answers[sequence] = {}
                answer_scores = {}
                for answer in ['Y','M','N']:
                    answer_scores[answer] = get_sequence_score(sequence, answer)
                sum_answers = sum(answer_scores.values())
                for answer in ['Y','M','N']:
                    seq_to_answers[sequence][answer] = answer_scores[answer] / sum_answers
                sorted_scored = sorted(seq_to_answers[sequence].items(), key=lambda kv: kv[1], reverse=True)
                seq_to_answers[sequence]['arg_max']  = sorted_scored[0][0]
    if print_latex:
        colors = {'Y':'00FF00','M': 'FFFF00','N':'FF0000'}
        for answer in ['Y', 'M', 'N']:
            seqs = {x:y for x,y in seq_to_answers.items() if y['arg_max'] == answer}
            sorted_seqs = dict(sorted(seqs.items(), key=lambda kv: kv[1][answer], reverse=True))
            for s, scores in sorted_seqs.items():
                y = round(scores['Y'], 2)
                m = round(scores['M'], 2)
                n = round(scores['N'], 2)
                if (y+m+n) != 1:
                    print("(y+m+n) !=  1 :" + str(y+m+n))
                print('\\rowcolor[HTML]{'+colors[answer]+'}')

                if add_posterior:
                    post = posterior[s]
                    yp = float(post['Y'])
                    mp = float(post['M'])
                    np = float(post['N'])
                    sump = yp + mp + np
                    yp = round(yp/sump, 2)
                    mp = round(mp/sump, 2)
                    np = round(np/sump, 2)
                    if (yp+mp+np) != 1:
                        print("(yp+mp+np) !=  1 :" + str(yp+mp+np))
                    pstr = s + "&{:.2f} ({:.2f})&{:.2f} ({:.2f})&{:.2f} ({:.2f}) \\\\".format(y, yp, m, mp , n, np)
                else:
                    pstr = s + "&{:.2f}&{:.2f}&{:.2f} \\\\".format(y, m, n)

                print(pstr)
                print('\hline')
    return

    fname = get_filename('sequence_to_score', limit)
    with open(fname,'w',newline='') as outcsv:
        fieldnames = ['sequence','Y','M','N']
        writer = csv.DictWriter(outcsv, fieldnames)
        writer.writeheader()
        for seq, scores in seq_to_answers.items():
            row = {'sequence': seq, 'Y':scores['Y'], 'M':scores['M'], 'N':scores['N']}
            writer.writerow(row)


def get_posterior_bias(row, ctr_dict):
    sequence = row['sequence']
    bias = {'Y': 0, 'M': 0, 'N': 0}
    if sequence[0] == 'A':
        ctr = ctr_dict['Direct Marketing Ads']
        ad = True
    elif sequence[0] == 'S':
        ctr = ctr_dict['Indirect Marketing Ads']
        ad = True
    else:
        ctr = ctr_dict['No ads']
        ad = False
    for i in range(0, len(sequence)):
        if ad and i ==0:
            continue
        v = sequence[i]
        bias[v] += float(ctr[i])

    bias_sorted = sorted(bias.items(), key=lambda kv: kv[1], reverse=True)  # sort newest_first
    print(sequence + ',' + str(bias['Y']) + ',' + str(bias['M']) + ',' + str(bias['N']) + ',' + bias_sorted[0][0])
    if ad:
        if bias_sorted[0][1] == 0:
            return sequence[0]
        else:
            return sequence[0] + bias_sorted[0][0]
    else:
        return bias_sorted[0][0]




def gen_ctr_dict():
    dict = {}
    with open('../resources/reports/ctr_all.csv', newline='') as csvF:
        reader = csv.DictReader(csvF)
        for row in reader:
            dict[row['series']] = list(row.values())[1:]
    return dict


def get_maximal_exposure_bias(row):
    sequence = row['sequence']
    l = len(sequence)
    ad = (sequence[0] == 'A' or sequence[0] =='S')
    bias = {'Y':0,'M':0,'N':0}
    for i in range (0,l):
        if ad:
            if i == 0:
                continue
        click = int(row['link'+str(i+1)])
        v = sequence[i]
        if ad:
            r = i
        else:
            r = i+1
        bias[v] += (1/(r))*click
    bias_sorted = sorted(bias.items(), key=lambda kv: kv[1], reverse=True)  # sort newest_first
    if ad:
        if bias_sorted[0][1] == 0:
            return sequence[0]
        else:
            return sequence[0]+bias_sorted[0][0]
    else:
        return bias_sorted[0][0]


def extract_clicks_from_behaviour_table(posterior_bias, prefix = None, filter_func=None, filter_title=None, limit = None):
    results = dict()
    possible_answers = set()
    visibility = {}
    ctr_dict = gen_ctr_dict()
    filename = "clicks_all"
    if prefix:
        filename += '_prefix_'+str(prefix)
    elif posterior_bias:
        filename += '_posterior_bias'
    if filter_title:
        filename += '_'+filter_title
    fname = get_filename('user_behaviour', limit)
    with open(fname, newline='', encoding='utf8') as csvf:
        reader = csv.DictReader(csvf)
        for row in reader:
            if filter_title:
                filter_ok = filter_func(row)
                if not filter_ok:
                    continue
            config = row['sequence']

            if prefix:
                if prefix == 1 and (config.startswith('A') or config.startswith('S')):
                    config = config[:(prefix+1)]
                else:
                    config = config[:prefix]
            elif posterior_bias:
                config = get_posterior_bias(row, ctr_dict)

            if config not in results:
                results[config] = {'Y': 0, 'N': 0, 'M': 0,'sum_answers':0}
                visibility[config] = {'Y': 0, 'N': 0, 'M': 0}
                results[config + '_NORM'] = {}
            seq = row['sequence']
            for i in range(0, len(seq)):
                answer = seq[i]
                if i == 0 and answer not in ['Y','M','N']:
                    continue
                results[config][answer] += int(row['link'+str(i+1)])
                visibility[config][answer] += 1
                results[config]['sum_answers']  += int(row['link'+str(i+1)])

        for config in results.keys():
            if config.startswith('link') or 'NORM' in config or config == 'sum_links':
                continue
            r = results[config]
            sum_answers = results[config]['sum_answers']
#            sum_answers = r['Y']+r['N']+r['M']+r['NS']
#            results[config]['sum_answers'] = sum_answers
            results[config+'_NORM']['Y'] = r['Y']/ visibility[config]['Y']
            results[config+'_NORM']['N'] = r['N']/ visibility[config]['N']
            results[config+'_NORM']['M'] = r['M']/ visibility[config]['M']


        fname = get_filename(filename, limit)
        with open(fname, 'w', newline='') as csvfile:
            fieldnames = ['sequence'] + ['Y','M','N','sum_answers']


            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for config, counters in results.items():
                row = {'sequence': config}
                row.update(counters)
                for answer in possible_answers:
                    if answer not in row:
                        row[answer] = 0
                writer.writerow(row)

def extract_answers_from_behaviour_table(posterior_bias, prefix = None, filter_func=None, filter_title=None, limit = None):
    results = dict()
    possible_answers = set()
    ctr_dict = gen_ctr_dict()
    filename = "feedback_all"
    if prefix:
        filename += '_prefix_'+str(prefix)
    elif posterior_bias:
        filename += '_posterior_bias'
    if filter_title:
        filename += '_'+filter_title
    link_visibility = {x:0 for x in range(1,11)}
    link_visibility_no_ads = {x:0 for x in range(1,11)}
    link_visibility_S = {x: 0 for x in range(1, 11)}
    link_visibility_SM = {x: 0 for x in range(1, 11)}
    link_visibility_SN = {x: 0 for x in range(1, 11)}
    link_visibility_A = {x: 0 for x in range(1, 11)}
    link_visibility_AM = {x: 0 for x in range(1, 11)}
    link_visibility_AN = {x: 0 for x in range(1, 11)}

    results['sum_links'] = {'link' + str(i) :0 for i in range(1,11)}
    fname = get_filename('user_behaviour', limit)
    with open(fname, newline='', encoding='utf8') as csvf:
        reader = csv.DictReader(csvf)
        for row in reader:
            if filter_title:
                filter_ok = filter_func(row)
                if not filter_ok:
                    continue
            config = row['sequence']
            for i in range (1, len(config)+1):
                link_visibility[i] += 1
                if config.startswith('A'):
                    link_visibility_A[i] += 1
                    if config.startswith('AM'):
                        link_visibility_AM[i] += 1
                    if config.startswith('AN'):
                        link_visibility_AN[i] += 1

                elif config.startswith('S'):
                    link_visibility_S[i] += 1
                    if config.startswith('SM'):
                        link_visibility_SM[i] += 1
                    if config.startswith('SN'):
                        link_visibility_SN[i] += 1
                else:
                    link_visibility_no_ads[i]+=1

            if prefix:
                if prefix == 1 and (config.startswith('A') or config.startswith('S')):
                    config = config[:(prefix+1)]
                else:
                    config = config[:prefix]
            elif posterior_bias:
                config = get_posterior_bias(row, ctr_dict)

            if config not in results:
                results[config] = {'Y': 0, 'N': 0, 'M': 0, 'NS': 0,'sum_answers':0}
                results[config + '_NORM'] = {}
                for i in range(1, 11):
                    results[config]['link' + str(i)] = 0
                    results[config+'_NORM']['link' + str(i)] = 0
                results[config]['link1 + link2'] = 0
            answer = row['feedback']
            results[config][answer] += 1
            results[config]['sum_answers'] += 1

            for i in range(1,11):
                link = 'link' + str(i)
                results[config][link] += int(row[link])
            if row['link1'] == '1' and row['link2'] =='1':
                results[config]['link1 + link2'] +=1

        for config in results.keys():
            if config.startswith('link') or 'NORM' in config or config == 'sum_links':
                continue
            r = results[config]
            sum_answers = results[config]['sum_answers']
#            sum_answers = r['Y']+r['N']+r['M']+r['NS']
#            results[config]['sum_answers'] = sum_answers
            results[config+'_NORM']['Y'] = r['Y']/sum_answers
            results[config+'_NORM']['N'] = r['N']/sum_answers
            results[config+'_NORM']['M'] = r['M']/sum_answers
            results[config+'_NORM']['NS'] = r['NS']/sum_answers
            for i in range(1, 11):
                link = 'link' + str(i)
                results['sum_links'][link] += r[link]

        results['link_visibility'] = {'link' + str(x): link_visibility[x] for x in range(1,11) }
        results['link_visibility_no_ads'] = {'link' + str(x): link_visibility_no_ads[x] for x in range(1,11) }
        results['link_visibility_A'] = {'link' + str(x): link_visibility_A[x] for x in range(1,11) }
        results['link_visibility_AM'] = {'link' + str(x): link_visibility_AM[x] for x in range(1, 11)}
        results['link_visibility_AN'] = {'link' + str(x): link_visibility_AN[x] for x in range(1, 11)}
        results['link_visibility_S'] = {'link' + str(x): link_visibility_S[x] for x in range(1,11) }
        results['link_visibility_SM'] = {'link' + str(x): link_visibility_SM[x] for x in range(1,11) }
        results['link_visibility_SN'] = {'link' + str(x): link_visibility_SN[x] for x in range(1,11) }

        fname = get_filename(filename, limit)
        with open(fname, 'w', newline='') as csvfile:
            fieldnames = ['sequence'] + ['Y','M','N','NS','sum_answers','link1 + link2']
            for i in range(1, 11):
                fieldnames.append('link' + str(i))


            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for config, counters in results.items():
                row = {'sequence': config}
                row.update(counters)
                for answer in possible_answers:
                    if answer not in row:
                        row[answer] = 0
                writer.writerow(row)


def gen_rank_order(limit, filter_field = None, filter_func= None, filter_title = None, ):
    orders = {}
    for i in range(1, 11):
        orders[i] = {'pressed_'+str(i): 0 for i in range(1, 11)}
    fname = get_filename('user_behaviour', limit)
    with open(fname, newline='', encoding='utf8') as csvf:
        reader = csv.DictReader(csvf)
        for row in reader:
            if filter_field and not filter_func(row[filter_field]):
                continue
            for order in range(1,11):
                rank = int(row['link_order'+str(order)])
                if rank > 0:
                    orders[rank]['pressed_'+str(order)] += 1
    if filter_field:
        fname = '../resources/reports/order_per_rank_'+filter_title+'.csv'
    else:
        fname = '../resources/reports/order_per_rank.csv'

    with open(fname, 'w', newline='') as csvfile:
        fieldnames = ['rank']
        for i in range(1, 11):
            fieldnames.append('pressed_'+ str(i))

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range (1,11):
            row = {'rank':i}
            row.update(orders[i])
            writer.writerow(row)


def gen_order_rank(filter_field = None, filter_func= None, filter_title = None):
    orders = {}
    for i in range(1, 11):
        orders[i] = {'rank_' + str(i): 0 for i in range(1, 11)}

    with open(BEHAVIOUR_FILE, newline='', encoding='utf8') as csvf:
        reader = csv.DictReader(csvf)
        for row in reader:
            if filter_field and not filter_func(row[filter_field]):
                continue
            for order in range(1,11):
                rank = int(row['link_order'+str(order)])
                if rank > 0:
                    orders[order]['rank_'+str(rank)] += 1
                else:
                    break
    if filter_field:
        fname = '../resources/reports/rank_per_order_'+filter_title+'.csv'
    else:
        fname = '../resources/reports/rank_per_order.csv'

    with open(fname, 'w', newline='') as csvfile:
        fieldnames = ['order']
        for i in range(1, 11):
            fieldnames.append('rank_'+ str(i))

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range (1,11):
            row = {'order':i}
            row.update(orders[i])
            writer.writerow(row)


def get_user_data(limit = None):
    db = connect_to_db(local=False)
    users = {}
    num_links = []
    num_links_per_config = {'AM':[],'AN':[],'SN':[],'SM':[],'NO ADS':[]}
    time_spent = []
    ad_config = []
    mycursor = db.cursor()
    fname = get_filename('user_behaviour', limit)
    with open(fname, newline='', encoding='utf8') as csvf:
        reader = csv.DictReader(csvf)
        for row in reader:
            user = row['WorkerId'].strip()
            users[user] = row
    query = 'SELECT * FROM serp.user_data;'
    mycursor.execute(query)
    results = mycursor.fetchall()
    fname =  get_filename('user_data', limit)
    with open(fname, 'w', newline='') as csvfile:
        fieldnames = ['user_id','age','gender','education_level','education_field']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        age = []
        gender = []
        for x in results:
            uid = x[0].strip()
            if uid in users:
                age.append(x[1])
                gender.append(x[2])
                writer.writerow({'user_id':x[0],'age':x[1],'gender':x[2],'education_level':x[3],'education_field':x[4]})
                time_spent.append(float(row['search_time_exp']))
                row = users[uid]
                nl = int(row['num_links_pressed'])
                num_links.append(nl)
                seq = row['sequence']
                seq2 = seq[:2]
                if seq2 in num_links_per_config:
                    num_links_per_config[seq2].append(nl)
                else:
                    num_links_per_config['NO ADS'].append(nl)

           #     print(row)
                if seq[0] == 'S' or seq[0] == 'A':
                    ad_config.append(seq[0])
                else:
                    ad_config.append('NONE')

        c = Counter(gender)
        c_ad = Counter(ad_config)
        print('num participants: ' + str(len(age)))
        print('num participants: ' + str(len(gender)))
        print('min age:' + str(min(age)))
        print('max age:' + str(max(age)))
        print('mean age: ' + str(mean(age)))
        print('STD age: ' + str(std(age)))
        print('male: ' + str(c['male']) + ':' + str(c['male']/len(gender)))
        print('female: ' + str(c['female']) + ':' + str(c['female']/len(gender)))
        print('average time: ' + str(mean(time_spent)))
        print('average num_links: ' + str(mean(num_links)))
        print('num S: ' + str(c_ad['S'] ))
        print('num A: ' + str(c_ad['A'] ))
        print('num no ads: ' + str(c_ad['NONE']))
        print('num links _per configuration')
        for c, l in num_links_per_config.items():
            print(c + ': ' + str(mean(l)))
        ttest = stats.ttest_ind(num_links_per_config['SM'], num_links_per_config['SN'])
        print(ttest)
        ttest = ttest_ind(num_links_per_config['SM'], num_links_per_config['SN'])
        print(ttest)



def ctr_per_viewpoint():
    link_visibility = {}
    ctr_dict = {}
    ad_config_list = ['NA','A','S','AM','AN','SM','SN']
    for c in ad_config_list:
        link_visibility[c] = {}
        ctr_dict[c] = {}
        for a in ['Y','M','N']:
            link_visibility[c][a] = {x: 0 for x in range(1, 11)}
            ctr_dict[c][a] = {x: 0 for x in range(1, 11)}

    link_visibility['S']['S'] = {x: 0 for x in range(1, 11)}
    link_visibility['A']['A'] = {x: 0 for x in range(1, 11)}
    ctr_dict['S']['S'] = {x: 0 for x in range(1, 11)}
    ctr_dict['A']['A'] = {x: 0 for x in range(1, 11)}

    fname = get_filename('user_behaviour', None)
    with open(fname, newline='', encoding='utf8') as csvf:
        reader = csv.DictReader(csvf)
        for row in reader:
            sequence = row['sequence']
            ad_config2 = None
            if sequence[0] == 'A' or sequence[0] == 'S':
                ad_config1 = sequence[0]
                ad_config2 = sequence[:2]
            else:
                ad_config1 = 'NA'
            for i in range(0,len(sequence)):
                v = sequence[i]
                if v not in  ctr_dict[ad_config1]:
                    ctr_dict[ad_config1][v] = {x: 0 for x in range(1, 11)}
                    link_visibility[ad_config1][v] = {x: 0 for x in range(1, 11)}

                ctr_dict[ad_config1][v][i+1] += int(row['link'+str(i+1)])
                link_visibility[ad_config1][v][i+1] += 1
                if ad_config2:
                    if v not in ctr_dict[ad_config2]:
                        ctr_dict[ad_config2][v] =  {x: 0 for x in range(1, 11)}
                        link_visibility[ad_config2][v] = {x: 0 for x in range(1, 11)}
                    ctr_dict[ad_config2][v][i+1] += int(row['link' + str(i + 1)])
                    link_visibility[ad_config2][v][i+1] += 1

    ctr_normed = {}
    for ad_config,answers_dicts in ctr_dict.items():
        ctr_normed[ad_config] = {}
        for a, ctrs in answers_dicts.items():
            ctr_normed[ad_config][a] = {'link'+str(x): 0 if  link_visibility[ad_config][a][x] == 0 else y/link_visibility[ad_config][a][x] for x,y in ctrs.items()}

    with open('../resources/reports/ctr_per_view_point.csv', 'w', newline='') as csvfile:
        fieldnames = ['sequence_prefix', 'viewpoint']
        for i in range(1,11):
            fieldnames.append('link'+str(i))
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for ad_config, answers_dicts in ctr_normed.items():
            for a, ctrs in answers_dicts.items():
                row = {'sequence_prefix':ad_config,'response':a}
                row.update(ctrs)
                writer.writerow(row)

def create_response_prediction_features_vectors(group):
    features = []
    fname = get_filename('user_behaviour', None)
    with open(fname, newline='', encoding='utf8') as csvf:
        reader = csv.DictReader(csvf)
        for row in reader:
            if row['feedback'] == 'NS':
                continue
            seq = row['sequence']
            if group == 'S' or group == 'A':
                if not seq.startswith(group):
                    continue
#                else:
#                    f = {'l1_' + group: 0, 'lc1_'+group: 0}
#                    start_i = 2
            else:
                if seq.startswith('A') or seq.startswith('S'):
                    continue
#                f = {}
#                start_i = 1
            f = {'l1_A': 0,'l1_S': 0, 'lc1_A': 0, 'lc1_S': 0,}
            for i in range(1, 11):
                f['l'+str(i)+'_Y']=0
                f['l'+str(i)+'_M']=0
                f['l'+str(i)+'_N']=0
                f['lc'+str(i)+'_Y']=0
                f['lc'+str(i)+'_M']=0
                f['lc'+str(i)+'_N']=0

            for i in range(0, len(seq)):
                v = seq[i]
                f['l'+str(i+1)+'_'+v]= 1
                f['lc' + str(i + 1) +'_'+ v] = row['link'+str(i+1)]
            f['response'] = row['feedback']
            features.append(f)
    with open('../resources/reports//user_response_labels_'+group+'.csv','w',newline='') as csvfile:
        fieldnames = ['response','l1_A','l1_S']
#        if group == 'A' or group == 'S':
#            fieldnames.append('l1_'+group)
#            start_i = 2
#        else:
#            start_i = 1
        for i in range(1, 11):
            fieldnames.append('l'+str(i)+'_Y')
            fieldnames.append('l'+str(i)+'_M')
            fieldnames.append('l'+str(i)+'_N')

#        if group == 'A' or group == 'S':
#            fieldnames.append('lc1_'+group)
#            start_i = 2
#        else:
#            start_i = 1
        fieldnames.append('lc1_A')
        fieldnames.append('lc1_S')
        for i in range(1, 11):
            fieldnames.append('lc'+str(i)+'_Y')
            fieldnames.append('lc'+str(i)+'_M')
            fieldnames.append('lc'+str(i)+'_N')

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for f in features:
            writer.writerow(f)


if __name__ == "__main__":
    create_response_prediction_features_vectors('S')
    create_response_prediction_features_vectors('A')
    create_response_prediction_features_vectors('O')
    #sequence_scores()
    #ctr_per_viewpoint()
    #create_response_prediction_features_vectors()
    #generate_user_behaviour_table(filter_users=True, local=False, add_time_diff_actions=True)
    #extract_answers_from_behaviour_table(prefix=1, limit=None)
    #extract_answers_from_behaviour_table(prefix = False, posterior_bias=True, limit=None)
    #extract_clicks_from_behaviour_table(prefix = False, posterior_bias=True, limit=None)
    #extract_answers_from_behaviour_table(prefix=2, limit=None)
    #get_user_data()
    #sequence_scores()
    #sequence_score_to_answer(limit= 5, buckets=False, skip_ad=True)
    #sequence_score_to_answer_posterior(limit= 5)

    #generate_user_behaviour_table(filter_users=True, local=False,  add_time_diff_actions=True)

    #generate_user_behaviour_table(limit=5, filter_users=True, local=False, add_time_diff_actions=False)

    #gen_order_rank(filter_field='sequence', filter_func =  lambda  x: len(x) > 7, filter_title="long_seq")
    #gen_order_rank(filter_field='sequence', filter_func =  lambda  x: 'A' in x, filter_title="only_ads")
    #gen_order_rank(filter_field='sequence', filter_func =  lambda  x: 'A' not in x, filter_title="no_ads")

    #get_answer_count(mode='seq')
    #get_answer_count(mode='url', print_update_query=True, local =False, prefix = 'S')

    #extract_answers_from_behaviour_table(prefix = 1, limit=5)
    #

    #extract_answers_from_behaviour_table(prefix=1, limit = 5, filter_title='_xclude_melatonin',
    #                                     filter_func=lambda x: not x['url'].startswith('Does Melatonin  treat jetlag'))
    #extract_answers_from_behaviour_table(prefix=1, limit = 5, filter_title = 'Does Ginkgo Biloba treat tinnitus', filter_func  = lambda  x: x['url'].startswith('Does Ginkgo Biloba treat tinnitus'))
    #extract_answers_from_behaviour_table(prefix=1, limit = 5,  filter_title = 'Does Melatonin  treat jetlag', filter_func  = lambda  x: x['url'].startswith('Does Melatonin  treat jetlag'))
    #extract_answers_from_behaviour_table(prefix=1, limit = 5,  filter_title = 'Does Omega Fatty Acids treat Adhd', filter_func  = lambda  x: x['url'].startswith('Does Omega Fatty Acids treat Adhd'))


