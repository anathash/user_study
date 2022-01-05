import csv

import requests

from db_analysis.utils import connect_to_db, get_time_diff, string_to_datetime, filter_user, get_server_url
from user_behaviour_table import BEHAVIOUR_FILE

MIN_MINUTES_IN_SERP = 2


def get_answers_per_url(dbcursor):
    exp_data_query_string = "SELECT user_id, query, test_url,treatment,problem, knowledge, start, end, reason FROM serp.exp_data"
    dbcursor.execute(exp_data_query_string)
    exp_data = dbcursor.fetchall()
    answers = {}
    for x in exp_data:
        user_id = x[0]
        query = x[1]
        url = x[2]
        if url not in answers:
            answers[url] = 0

        treatment_answer = x[3]
        condition_answer = x[4]
        prev_know = x[5]
        start = x[6]
        end = x[7]
     #   sdt = string_to_datetime(start)
    #    edt = string_to_datetime(end)
       # time_spent = get_time_diff(start, end)

        if not filter_user(user_id, query, treatment_answer, condition_answer, start, end, prev_know, x[8]):
            answers[url] += 1
    return answers


def set_needed_answers_for_ads(seq_prefix, ad_prefix, local, exclude = 'Does Melatonin  treat jetlag'):
    db = connect_to_db(local)
    server_url = get_server_url(local)
    mycursor = db.cursor()
    answer_seq_dict = {}
    query = "SELECT URL FROM serp.config_data  where query != '" + exclude + "';"
    mycursor.execute(query)
    myresult = mycursor.fetchall()
    for x in myresult:
        db_url = x[0]
        db_seq = db_url.split('SERP/')[1]
        answer_seq_dict[db_seq] = 0

    total = 0
    with open(BEHAVIOUR_FILE, newline='', encoding='utf8') as csvf:
        reader = csv.DictReader(csvf)
        for row in reader:
            url = row['url']
            if exclude in url:
                continue

            if url not in answer_seq_dict:
                answer_seq_dict[url] = 0
            answer_seq_dict[url] += 1

        for url, count in answer_seq_dict.items():

            qs = url.split('-')
            q = qs[0]
            s = qs[1]
            if s.startswith(seq_prefix):
                total += count
                new_seq = ad_prefix + s
                new_url = server_url + q+'-' +new_seq
                #print('UPDATE serp.config_data set used =0, answered = 0, needed_answers=' + str(count) + " where URL='"+new_url+"';")
                print('UPDATE serp.config_data set  needed_answers=' + str(count) + " where URL='"+new_url+"';")
    print(total)


def set_need_answers_for_ads():
    with open('../resources/reports//answers_per_query_seq.csv', newline='', encoding='utf8') as csvf:
        reader = csv.DictReader(csvf)
        for row in reader:
            url = row['url']


def update_config_table(dbcursor):
    answers = get_answers_per_url(dbcursor)
    for url, answered in answers.items():
        msg = 'update url ' + url + ' to ' + str(answered) + ' answers'
        #print(msg)
        query = "UPDATE serp.config_data SET answered = " +str(answered) +" WHERE URL = '"+url+"'"
        print(query+';')
     #   dbcursor.execute(query)



if __name__ == "__main__":
#    db = connect_to_db(test=False)
#    dbcursor = db.cursor()
#    update_config_table(dbcursor)
    set_needed_answers_for_ads(local = False, seq_prefix='M', ad_prefix='S')



