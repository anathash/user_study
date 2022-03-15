import csv

import requests

from db_analysis.utils import connect_to_db, get_time_diff, string_to_datetime, filter_user, SERVER_URL

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


def set_needed_answers_for_ads(seq_prefix, ad_prefix, db_name, exclude = 'Does Melatonin  treat jetlag'):
    db = connect_to_db(db_name)
    server_url = SERVER_URL[db_name]
    mycursor = db.cursor()
    answer_seq_dict = {}
    query = "SELECT URL FROM serp.config_data  where query != '" + exclude + "';"
    mycursor.execute(query)
    myresult = mycursor.fetchall()
    for x in myresult:
        db_url = x[0]
        db_seq = db_url.split('serp/')[1]
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


def stringefy(str):
    return "'"+str+"'"

def add_files_to_config(from_id, seq_prefix, ad_prefix, db_name, exclude = 'Does Melatonin  treat jetlag',answered=0,needed_answeres=3):
    db = connect_to_db(db_name)
    server_url = SERVER_URL[db_name]
    mycursor = db.cursor()
    answer_seq_dict = {}
    query = "SELECT * FROM serp.config_data  where query != '" + exclude + "';"
    mycursor.execute(query)
    myresult = mycursor.fetchall()
    id = from_id
    for x in myresult:
        query = x[1]
        entry_file = x[2]
        sequence = x[3]
        db_url = x[4]
        topic0 = x[5]
        topic1 = x[6]
        #used = x[7]
        #answered = x[8]
        #needed_answers = x[9]
        if 'db_name' == 'biu':
            db_url_split = db_url.split('serp/')
            url_prefix = db_url_split[0]+'serp/'
        else:
            db_url_split = db_url.split('SERP/')
            url_prefix = db_url_split[0]+'SERP/'

        orig_page = db_url_split[1]
        qs = orig_page.split('-')
        q = qs[0]
        s = qs[1]
        if not s.startswith(seq_prefix):
            continue
        new_sequence = ad_prefix + sequence
        new_seq = ad_prefix + s
        new_url = url_prefix + q + '-' + new_seq
        # print('UPDATE serp.config_data set used =0, answered = 0, needed_answers=' + str(count) + " where URL='"+new_url+"';")
        #print('UPDATE serp.config_data set  needed_answers=' + str(needed_answeres) + " where URL='" + new_url + "';")
        #print('UPDATE serp.config_data set  answered=' + str(answered) + " where URL='" + new_url + "';")
        id += 1
#        print("insert into  serp.config_data (id,query,entry_file,sequence,URL,topic0,topic1,used,answered,needed_answers) VALUES  (" +
#              stringefy(str(id)) +stringefy(query) + "," + stringefy(entry_file) + "," + stringefy(sequence) +"," + stringefy(db_url)
#              + "," + stringefy(topic0) + "," + stringefy(topic1) + "," + stringefy(used) + "," + stringefy(answered) + "," + stringefy(needed_answers)+
#              ");")

        print("insert into  serp.config_data (id,query,entry_file,sequence,URL,topic0,topic1,used,answered,needed_answers) VALUES  (" +
              str(id) + "," + stringefy(query) + "," + stringefy(entry_file) + "," + stringefy(new_sequence) +"," + stringefy(new_url)
              + "," + stringefy(topic0) +
              "," + stringefy(topic1) + "," + "0" + "," + str(answered) + "," + str(needed_answeres)+
              ");")


if __name__ == "__main__":
#    db = connect_to_db(test=False)
#    dbcursor = db.cursor()
#    update_config_table(dbcursor)
    add_files_to_config(from_id=264, seq_prefix='Y', ad_prefix='S', db_name='local')



