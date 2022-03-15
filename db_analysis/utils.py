import mysql.connector
from datetime import datetime


TREATMENT_CORRECT_ANSWERS={'Does Omega Fatty Acids treat Adhd':'nutrients',
                           'Does Melatonin  treat jetlag':'hormone',
                           'Does Ginkgo Biloba treat tinnitus':'tree'}

CONDITION_CORRECT_ANSWERS={'Does Omega Fatty Acids treat Adhd':'neurological',
                           'Does Melatonin  treat jetlag':'sleep',
                           'Does Ginkgo Biloba treat tinnitus':'ringing'}

FILTER_TIME = 1

WORKERS_WITH_NO_LINKS = ['A0505289TSH2NC1YOHYK', 'A26UIS59SY4NM6', 'A272X64FOZFYLB', 'A2EED3HLTA96CP', 'AERUGBNS48Z4N', 'A3N6DWJC7P3HSI', 'A2TLN8489YGY81','A3S3WYVCVWW8IZ']


test_users_names_prefix = ['anat', 'anst','sharadhi','test', 'tamar','tetst','zw3hk']

EXP_START_DATE = datetime.strptime('12/21/2021, 01:01:31 AM', '%m/%d/%Y, %I:%M:%S %p')
SERVER_URL = {'local': "http://localhost/SERP/",'virginia':"http://cs.virginia.edu/~zw3hk/SERP/",'biu':'http://experiments.biu-ai.com:15080/serp/'}



def connect_to_db(db_name):
    if db_name == 'local':
        db = mysql.connector.connect(
             host="localhost",
             user="root",
             passwd="anat",
             database="serp"
         )  # database connection
        return db
    elif db_name == 'virginia':
        db = mysql.connector.connect(
            host="hcdm3.cs.virginia.edu",
            user="zw3hk",
            passwd="Fall2021!!",
            database="serp"
        )  # database connection
        return db
    elif db_name == 'biu':
        db = mysql.connector.connect(
            host="experiments.biu-ai.com",
            user="serp",
            passwd="aimsql",
            database="serp",
            port = 15036,
           # auth_plugin = 'mysql_native_password'
        )  # database connection
        return db
    elif db_name == 'tamar':
        db = mysql.connector.connect(
            host="experiments2.biu-ai.com",
            user="serp_user",
            passwd="YpkYNyMA4qWtJA",
            database="serp",
            port = 15136,
           # auth_plugin = 'mysql_native_password'
        )  # database connection
        return db
    elif db_name == 'shared':
        db = mysql.connector.connect(
            host="experiments2.biu-ai.com",
            user="serp_user",
            passwd="YpkYNyMA4qWtJA",
            database="serp_shared",
            port = 15136,
           # auth_plugin = 'mysql_native_password'
        )  # database connection
        return db
    else:
        raise Exception


def string_to_datetime(str):
    return datetime.strptime(str, '%m/%d/%Y, %I:%M:%S %p')


def get_workers_with_no_link(dbcursor, worker_ids = None):
    clicked_links = set()
    closed_page = set()
    if worker_ids:
        sql_user_action_query_string = "SELECT user_id, action, date FROM serp.user_action where user_id in " + worker_ids
    else:
        sql_user_action_query_string = "SELECT user_id, action, date FROM serp.user_action"

    dbcursor.execute(sql_user_action_query_string)
    user_actions = dbcursor.fetchall()
    for x in user_actions:
        if x[1] == 'click link':
            clicked_links.add(x[0])
        elif x[1] == 'close page':
            closed_page.add(x[0])

    users_with_no_links = closed_page.difference(clicked_links)
    return users_with_no_links


def unsatisfactory(query, treatment_answer, condition_answer, time, print_msg):
    if treatment_answer.strip() != TREATMENT_CORRECT_ANSWERS[query]:
        if print_msg:
            print(treatment_answer + '!=' + TREATMENT_CORRECT_ANSWERS[query] + ' for ' + query)
        return 'treatment answer error'
    if condition_answer.strip() != CONDITION_CORRECT_ANSWERS[query]:
        if print_msg:
            print(condition_answer + '!=' + CONDITION_CORRECT_ANSWERS[query] + ' for ' + query)
        return 'condition_answer answer error'

    if time < FILTER_TIME:
        if print_msg:
            print(str(time) + '<' + str(FILTER_TIME))
        return 'too quick'
    return False


def print_filter_message(sdt, id, reason, print = False):
    if print:
        if sdt > string_to_datetime('12/27/2021, 12:01:31 PM'):
            print('filter user ' + id +' because ' + reason)


def filter_user(user, query, treatment_answer, condition_answer, start,end, prev_know,
                reason, filter_prev_know = True):
    if not end or not start:
        return 'no date'
    sdt = string_to_datetime(start)
    edt = string_to_datetime(end)
    time = get_time_spent(sdt, edt)
#    if filter_early and filter_exp_start and sdt < EXP_START_DATE:
#        print_filter_message(sdt, user,'filter_exp_start and sdt < EXP_START_DATE')
#        return 'early'

    for test_user in test_users_names_prefix:
        if user.lower().startswith(test_user):
            print_filter_message(sdt, user, 'user.lower().startswith(test_user')
            return 'test'

    if not reason or reason == 'None':
        print_filter_message(sdt, user, 'no reason')
        return 'no reason'

    if user in WORKERS_WITH_NO_LINKS:
        print_filter_message(sdt, user, 'no links')
        return 'no links'

    #print_msg = sdt > string_to_datetime('12/27/2021, 12:01:31 PM')
    print_msg = False
    bad_wok_msg = unsatisfactory(query, treatment_answer, condition_answer, time, print_msg)
    if bad_wok_msg:
        print_filter_message(sdt, user, 'unsatisfactory work')
        return bad_wok_msg

    if filter_prev_know and prev_know == 'yes':
        print_filter_message(sdt, user, 'previous knowledge')
        return 'previous knowledge'

    return False


def filter_user_new(user, query, treatment_answer, condition_answer, start,end, prev_know,
                reason, filter_prev_know = True):
    if not end or not start:
        return 'no date'
    time = get_time_spent(start, end)
#    if filter_early and filter_exp_start and sdt < EXP_START_DATE:
#        print_filter_message(sdt, user,'filter_exp_start and sdt < EXP_START_DATE')
#        return 'early'

    for test_user in test_users_names_prefix:
        if user.lower().startswith(test_user):
            print_filter_message(start, user, 'user.lower().startswith(test_user')
            return 'test'

    if not reason or reason == 'None':
        print_filter_message(start, user, 'no reason')
        return 'no reason'

    if user in WORKERS_WITH_NO_LINKS:
        print_filter_message(start, user, 'no links')
        return 'no links'

    #print_msg = sdt > string_to_datetime('12/27/2021, 12:01:31 PM')
    print_msg = False
    bad_wok_msg = unsatisfactory(query, treatment_answer, condition_answer, time, print_msg)
    if bad_wok_msg:
        print_filter_message(start, user, 'unsatisfactory work')
        return bad_wok_msg

    if filter_prev_know and prev_know == 'yes':
        print_filter_message(start, user, 'previous knowledge')
        return 'previous knowledge'

    return False


def get_time_spent(start_datetime,end_datetime, to_minutes=True):
    if to_minutes:
        time_spent = (end_datetime - start_datetime).total_seconds() / 60.0
    else:
        time_spent = (end_datetime - start_datetime).total_seconds()
    return time_spent


def get_time_diff_from_actions(dbcursor, user_id, exp_time, exp_end_time, to_minutes = True):
    #sql_user_action_query_string = "SELECT action, date FROM serp.user_action where exp_id ='"+exp_id+"'"
    sql_user_action_query_string = "SELECT action, date FROM serp.user_action where user_id ='"+user_id+"'"
    dbcursor.execute(sql_user_action_query_string)
    user_actions = dbcursor.fetchall()
    start = None
    end = None
    for action in user_actions:
        time_str = datetime.strptime(action[1], '%m/%d/%Y, %I:%M:%S %p')
        if action[0] == 'close page':
            end = time_str
        if not start or time_str < start:
            start = time_str
    actions_time = get_time_spent(start, end, to_minutes)
    return actions_time
 #   if exp_time > 0 and actions_time > exp_time:
 #       end_str = datetime.strptime(exp_end_time, '%m/%d/%Y, %I:%M:%S %p')
 #       exp_time = get_time_spent(start, end_str, to_minutes)
  #  return actions_time, exp_time


def get_time_diff(start, end, to_minutes = True):
    if not end:
        return 0
    start_datetime = datetime.strptime(start, '%m/%d/%Y, %I:%M:%S %p')
    end_datetime = datetime.strptime(end, '%m/%d/%Y, %I:%M:%S %p')
    return get_time_spent(start_datetime, end_datetime, to_minutes)


def get_links_stats(dbcursor, worker_ids = None):
    if worker_ids:
        sql_user_action_query_string = "SELECT  user_id, link_id, date FROM serp.user_action where action ='click link' and user_id in " + worker_ids
    else:
        sql_user_action_query_string = "SELECT  user_id, link_id, date FROM serp.user_action where action ='click link'"

    dbcursor.execute(sql_user_action_query_string)
    user_actions = dbcursor.fetchall()
    links_pressed = {}

    links = {}
    for l in user_actions:
        user_id = l[0]
        link_id = l[1]
        t_press = l[2]
        if user_id not in links:
            links[user_id] = []
            links_pressed[user_id] = {l: 0 for l in range(1, 11)}
        links_pressed[user_id][link_id] = 1
        links[user_id].append({'link_id':link_id,'date':string_to_datetime(t_press)})
    link_times = {}
    link_orders = {}
    for user, l in links.items():
        links_by_date = sorted(l, key = lambda i: i['date'])
        if user not in link_times:
            link_times[user] = {x: 0 for x in range(0, 11)}
            link_orders[user] = {x: 0 for x in range(0, 11)}
        l = len(links_by_date)
        for i in range (0, l-1):
            rank = links_by_date[i]['link_id']
            date = links_by_date[i]['date']
            next_action = links_by_date[i+1]['date']
            time_diff = get_time_spent(date,next_action, False)
            link_times[user][rank] += time_diff
            link_orders[user][i+1] = rank
        #for order we need the last link as well
        rank = links_by_date[l-1]['link_id']
        link_orders[user][l] = rank
    return links_pressed, link_times, link_orders


def get_links_stats_by_exp(dbcursor, worker_ids = None, date_str = None):
    if worker_ids:
        sql_user_action_query_string = "SELECT  exp_id, link_id, date, action FROM serp.user_action where  user_id in " + worker_ids
    else:
        if date_str:
            sql_user_action_query_string = "SELECT  exp_id, link_id, date, action FROM serp.user_action where date > '" + date_str + "'"
        else:
            sql_user_action_query_string = "SELECT  exp_id, link_id, date, action FROM serp.user_action"

    dbcursor.execute(sql_user_action_query_string)
    user_actions = dbcursor.fetchall()
    links_pressed = {}
    close_pages = {}
    time_diff_actions = {}
    links = {}
    for l in user_actions:
        exp_id = l[0]
        link_id = l[1]
        t_press = l[2]
        action = l[3]

        if action == 'click link':
            if exp_id not in links:
                links[exp_id] = []
                links_pressed[exp_id] = {l: 0 for l in range(1, 11)}

            links_pressed[exp_id][link_id] = 1
            links[exp_id].append({'link_id':link_id,'date':t_press})
        elif action == 'close page':
            close_pages[exp_id] = t_press
        else:
            raise Exception
    link_times = {}
    link_orders = {}
    #        if not start or time_str < start:
            #start = time_str
    for exp_id, l in links.items():
        links_by_date = sorted(l, key = lambda i: i['date'])
        first_link = links_by_date[0]['date']
        if exp_id in close_pages:
            time_diff_actions[exp_id] = get_time_spent(first_link, close_pages[exp_id], True)
        else:
            time_diff_actions[exp_id] = 0
        if exp_id not in link_times:
            link_times[exp_id] = {x: 0 for x in range(0, 11)}
            link_orders[exp_id] = {x: 0 for x in range(0, 11)}
        l = len(links_by_date)
        for i in range (0, l-1):
            rank = links_by_date[i]['link_id']
            date = links_by_date[i]['date']
            next_action = links_by_date[i+1]['date']
            time_diff = get_time_spent(date,next_action, False)
            link_times[exp_id][rank] += time_diff
            link_orders[exp_id][i+1] = rank
        #for order we need the last link as well
        rank = links_by_date[l-1]['link_id']
        link_orders[exp_id][l] = rank
    return links_pressed, link_times, link_orders, time_diff_actions

def get_links_entered_by_worker(dbcursor, worker_ids = None):
    if worker_ids:
        sql_user_action_query_string = "SELECT exp_id, user_id, link_id, date FROM serp.user_action where action ='click link' and user_id in " + worker_ids
    else:
        sql_user_action_query_string = "SELECT exp_id, user_id, link_id, date FROM serp.user_action where action ='click link'"

    dbcursor.execute(sql_user_action_query_string)
    user_actions = dbcursor.fetchall()
    links = {}
    for x in user_actions:
        user_id = x[1]
        link_id = x[2]
        t_press = x[3]
        if user_id not in links:
            links[user_id] = {l: 0 for l in range(1, 11)}
        links[user_id][link_id] = 1
    return links


def get_num_links_per_worker__(dbcursor, worker_ids = None):
    if worker_ids:
        sql_user_action_query_string = "SELECT exp_id, user_id, link_id FROM serp.user_action where action ='click link' and user_id in " + worker_ids
    else:
        sql_user_action_query_string = "SELECT exp_id, user_id, link_id FROM serp.user_action where action ='click link'"
    dbcursor.execute(sql_user_action_query_string)
    user_actions = dbcursor.fetchall()
    num_links_pers_workers = {}
    for x in user_actions:
        exp_id = x[0]
        worker_id = x[1]
        if worker_id not in num_links_pers_workers:
            num_links_pers_workers[worker_id] = {}
            if exp_id not in worker_id:
                num_links_pers_workers[worker_id][exp_id] = set()
        num_links_pers_workers[worker_id][exp_id].add(x[1])
    return num_links_pers_workers


def get_links_per_worker(dbcursor, worker_ids = None):
    if worker_ids:
        sql_user_action_query_string = "SELECT exp_id, user_id, link_id FROM serp.user_action where action ='click link' and user_id in " + worker_ids
    else:
        sql_user_action_query_string = "SELECT exp_id, user_id, link_id FROM serp.user_action where action ='click link'"
    dbcursor.execute(sql_user_action_query_string)
    user_actions = dbcursor.fetchall()
    num_links_pers_workers = {}
    for x in user_actions:
#        exp_id = x[0]
        worker_id = x[1]
        if worker_id not in num_links_pers_workers:
            num_links_pers_workers[worker_id] = set()
#            if exp_id not in num_links_pers_workers[worker_id]:
#                num_links_pers_workers[worker_id][exp_id] = set()
#        num_links_pers_workers[worker_id][exp_id].add(x[2])
        num_links_pers_workers[worker_id].add(x[2])
    return num_links_pers_workers


