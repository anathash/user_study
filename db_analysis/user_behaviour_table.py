import csv

from db_analysis.utils import connect_to_db, get_time_diff, get_time_diff_from_actions, \
    TREATMENT_CORRECT_ANSWERS, CONDITION_CORRECT_ANSWERS, get_links_entered_by_worker, filter_user, get_links_stats, \
    string_to_datetime, get_server_url
from process_batch import get_worker_id_list, BATCH_FILE_PREFIX
BEHAVIOUR_FILE = '../resources/reports//user_behaviour_limit_5.csv'

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


def generate_user_behaviour_table(limit = None, from_batch = None, to_batch=None, local = True, filter_users = True, time_diff_actions = False):
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
                                             link_orders=link_orders, filter_users = filter_users, time_diff_actions = time_diff_actions)

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


def extract_answers_from_behaviour_table(prefix = None, filter_func=None, filter_title=None):
    results = dict()
    possible_answers = set()
    filename = "feedback_all"
    if prefix:
        filename += '_prefix_'+str(prefix)
    if filter_title:
        filename += '_'+filter_title
    filename += '.csv'
    link_visibility = {x:0 for x in range(1,11)}
    link_visibility_no_ads = {x:0 for x in range(1,11)}

    results['sum_links'] = {'link' + str(i) :0 for i in range(1,11)}
    with open(BEHAVIOUR_FILE, newline='', encoding='utf8') as csvf:
        reader = csv.DictReader(csvf)
        for row in reader:
            if filter_title:
                filter_ok = filter_func(row)
                if not filter_ok:
                    continue
            config = row['sequence']
            for i in range (1, len(config)+1):
                link_visibility[i] += 1
                if 'A' not in config and 'S' not in config:
                    link_visibility_no_ads[i]+=1

            if prefix:
                if prefix == 1 and (config.startswith('A') or config.startswith('S')):
                    config = config[:(prefix+1)]
                else:
                    config = config[:prefix]

            if config not in results:
                results[config] = {'Y': 0, 'N': 0, 'M': 0, 'NS': 0,'sum_answers':0}
                results[config + '_NORM'] = {}
                for i in range(1, 11):
                    results[config]['link' + str(i)] = 0
                    results[config+'_NORM']['link' + str(i)] = 0
            answer = row['feedback']
            results[config][answer] += 1
            results[config]['sum_answers'] += 1

            for i in range(1,11):
                link = 'link' + str(i)
                results[config][link] += int(row[link])

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

        with open('../resources/reports//' + filename, 'w', newline='') as csvfile:
            fieldnames = ['sequence'] + ['Y','M','N','NS','sum_answers']
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


def gen_rank_order(filter_field = None, filter_func= None, filter_title = None):
    orders = {}
    for i in range(1, 11):
        orders[i] = {'pressed_'+str(i): 0 for i in range(1, 11)}
    with open(BEHAVIOUR_FILE, newline='', encoding='utf8') as csvf:
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



if __name__ == "__main__":
    generate_user_behaviour_table(filter_users=True, local=True, add_time_diff_actions=False)
    #generate_user_behaviour_table(limit=5, filter_users=True, local=False, add_time_diff_actions=False)

    #gen_order_rank(filter_field='sequence', filter_func =  lambda  x: len(x) > 7, filter_title="long_seq")
    #gen_order_rank(filter_field='sequence', filter_func =  lambda  x: 'A' in x, filter_title="only_ads")
    #gen_order_rank(filter_field='sequence', filter_func =  lambda  x: 'A' not in x, filter_title="no_ads")

    #get_answer_count(mode='seq')
    #get_answer_count(mode='url', print_update_query=True, local =False, prefix = 'S')

    #extract_answers_from_behaviour_table()
    #

    #extract_answers_from_behaviour_table(prefix=1, filter_title='_xclude_melatonin',
     #                                    filter_func=lambda x: not x['url'].startswith('Does Melatonin  treat jetlag'))
    #extract_answers_from_behaviour_table(prefix=1, filter_title = 'Does Ginkgo Biloba treat tinnitus', filter_func  = lambda  x: x['url'].startswith('Does Ginkgo Biloba treat tinnitus'))
    #extract_answers_from_behaviour_table(prefix=1, filter_title = 'Does Melatonin  treat jetlag', filter_func  = lambda  x: x['url'].startswith('Does Melatonin  treat jetlag'))
    #extract_answers_from_behaviour_table(prefix=1, filter_title = 'Does Omega Fatty Acids treat Adhd', filter_func  = lambda  x: x['url'].startswith('Does Omega Fatty Acids treat Adhd'))
    #extract_answers_from_behaviour_table()

