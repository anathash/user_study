
from db_analysis.utils import connect_to_db, get_time_diff

MIN_MINUTES_IN_SERP = 2




def process_exp_quality(dbcursor):
    exp_data_query_string = "SELECT exp_id, test_url, start, end, knowledge FROM serp.exp_data"
    dbcursor.execute(exp_data_query_string)
    exp_data = dbcursor.fetchall()
    omitted_exps = []
    answers = []
    for x in exp_data:
        exp_id = x[0]
        url = x[1]
        start = x[2]
        end = x[3]
        knowledge = x[4]
        time_spent = get_time_diff(start, end)
        if time_spent >= MIN_MINUTES_IN_SERP or knowledge == 'yes':
            answers[url] += 1
        else:
            omitted_exps.append(exp_id)
    return answers, omitted_exps

def update_config_table(answers, dbcursor):
    config_data_query_string = "SELECT URL,answered  FROM serp.config_data"
    dbcursor.execute(config_data_query_string)
    db_recorded_answers = {}
    config_data = dbcursor.fetchall()
    for x in config_data:
        db_recorded_answers[x[0]] = x[1]
        answers[x[0]] = 0

    for url, answers in db_recorded_answers:
        if answers[url] != db_recorded_answers:
            print('update url '  + url + ' to ' + str(answers) + ' answers')
            #TODO : update DB
    #update ommited urls table
    # "UPDATE serp.config_data SET answered = answered + 1 WHERE URL = ?"



if __name__ == "__main__":
    db = connect_to_db()
    dbcursor = db.cursor()
    answers, omitted_exps = process_exp_quality(dbcursor)
    update_config_table(answers, dbcursor)



