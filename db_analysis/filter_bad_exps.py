import requests

from db_analysis.utils import connect_to_db, get_time_diff, string_to_datetime, filter_user

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


def update_config_table(dbcursor):
    answers = get_answers_per_url(dbcursor)
    for url, answered in answers.items():
        msg = 'update url ' + url + ' to ' + str(answered) + ' answers'
        #print(msg)
        query = "UPDATE serp.config_data SET answered = " +str(answered) +" WHERE URL = '"+url+"'"
        print(query+';')
     #   dbcursor.execute(query)



if __name__ == "__main__":
    db = connect_to_db(test=False)
    dbcursor = db.cursor()
    update_config_table(dbcursor)



