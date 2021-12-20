import mysql.connector
from datetime import datetime


TREATMENT_CORRECT_ANSWERS={'Does Omega Fatty Acids treat Adhd':'nutrients',
                           'Does Melatonin  treat jetlag':'hormone',
                           'Does Ginkgo Biloba treat tinnitus':'tree'}

CONDITION_CORRECT_ANSWERS={'Does Omega Fatty Acids treat Adhd':'neurological',
                           'Does Melatonin  treat jetlag':'sleep',
                           'Does Ginkgo Biloba treat tinnitus':'ringing'}

FILTER_TIME = 1

test_users_names_prefix = ['anat', 'anst','sharadhi','test']

FILTER_PERV_KNOWLEDGE = False

def connect_to_db():
    db = mysql.connector.connect(
        host="hcdm3.cs.virginia.edu",
        user="zw3hk",
        passwd="Fall2021!!",
        database="serp"
    )  # database connection
    return db


def filter_user(user, query, treatment_answer, condition_answer, time, prev_know):
    for test_user in test_users_names_prefix:
        if user.startswith(test_user):
            return True
    if treatment_answer != TREATMENT_CORRECT_ANSWERS[query]:
        return True
    if condition_answer != CONDITION_CORRECT_ANSWERS[query]:
        return True

    if time < FILTER_TIME:
        return  True

    if FILTER_PERV_KNOWLEDGE and prev_know == 'Y':
        return True

    return False


def get_time_spent(start_datetime,end_datetime, to_minutes=True):
    if to_minutes:
        time_spent = (end_datetime - start_datetime).total_seconds() / 60.0
    else:
        time_spent = (end_datetime - start_datetime).total_seconds()
    return time_spent


def get_time_diff_from_actions(dbcursor, exp_id, to_minutes = True):
    sql_user_action_query_string = "SELECT action, date FROM serp.user_action where exp_id ='"+exp_id+"'"
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
    return get_time_spent(start, end, to_minutes)


def get_time_diff(start, end, to_minutes = True):
    start_datetime = datetime.strptime(start, '%m/%d/%Y, %I:%M:%S %p')
    end_datetime = datetime.strptime(end, '%m/%d/%Y, %I:%M:%S %p')
    return get_time_spent(start_datetime, end_datetime, to_minutes)


def get_links_entered_by_worker(dbcursor, worker_ids = None):
    if worker_ids:
        sql_user_action_query_string = "SELECT exp_id, user_id, link_id FROM serp.user_action where action ='click link' and user_id in " + worker_ids
    else:
        sql_user_action_query_string = "SELECT exp_id, user_id, link_id FROM serp.user_action where action ='click link'"

    dbcursor.execute(sql_user_action_query_string)
    user_actions = dbcursor.fetchall()
    links = {}
    for x in user_actions:
        user_id = x[1]
        link_id = x[2]
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


def get_num_links_per_worker(dbcursor, worker_ids = None):
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
            if exp_id not in num_links_pers_workers[worker_id]:
                num_links_pers_workers[worker_id][exp_id] = set()
        num_links_pers_workers[worker_id][exp_id].add(x[1])
    return num_links_pers_workers
