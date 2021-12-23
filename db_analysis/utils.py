import mysql.connector
from datetime import datetime


TREATMENT_CORRECT_ANSWERS={'Does Omega Fatty Acids treat Adhd':'nutrients',
                           'Does Melatonin  treat jetlag':'hormone',
                           'Does Ginkgo Biloba treat tinnitus':'tree'}

CONDITION_CORRECT_ANSWERS={'Does Omega Fatty Acids treat Adhd':'neurological',
                           'Does Melatonin  treat jetlag':'sleep',
                           'Does Ginkgo Biloba treat tinnitus':'ringing'}

FILTER_TIME = 1

WORKERS_WITH_NO_LINKS = ['A0505289TSH2NC1YOHYK', 'A26UIS59SY4NM6', 'A272X64FOZFYLB', 'A2EED3HLTA96CP', 'AERUGBNS48Z4N', 'A3N6DWJC7P3HSI', 'A2TLN8489YGY81']

test_users_names_prefix = ['anat', 'anst','sharadhi','test']



def connect_to_db(test = False):
    if test:
        db = mysql.connector.connect(
             host="localhost",
             user="root",
             passwd="anat",
             database="serp"
         )  # database connection
        return db
    else:
        db = mysql.connector.connect(
            host="hcdm3.cs.virginia.edu",
            user="zw3hk",
            passwd="Fall2021!!",
            database="serp"
        )  # database connection
        return db


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

def unsatisfactory(query, treatment_answer, condition_answer, time):
    if treatment_answer.strip() != TREATMENT_CORRECT_ANSWERS[query]:
        return True
    if condition_answer.strip() != CONDITION_CORRECT_ANSWERS[query]:
        return True

    if time < FILTER_TIME:
        return True


def filter_user(user, query, treatment_answer, condition_answer, time, prev_know, reason, filter_prev_know = True):
    for test_user in test_users_names_prefix:
        if user.lower().startswith(test_user):
            return True
    if not reason or reason == 'None':
        return True

    if user in WORKERS_WITH_NO_LINKS:
        return True

    if unsatisfactory(query, treatment_answer, condition_answer, time):
        return True

    if filter_prev_know and prev_know == 'yes':
        return True

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
    for user, l in links.items():
        links_by_date = sorted(l, key = lambda i: i['date'])
        if user not in link_times:
            link_times[user] = {x: 0 for x in range(0, 11)}
        for i in range (0, len(links_by_date)-1):
            rank = links_by_date[i]['link_id']
            date = links_by_date[i]['date']
            next_action = links_by_date[i+1]['date']
            time_diff = get_time_spent(date,next_action, False)
            link_times[user][rank] += time_diff
    return links_pressed, link_times


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


