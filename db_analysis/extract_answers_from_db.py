import csv
from datetime import datetime
from db_analysis.utils import connect_to_db, get_time_spent, filter_user, string_to_datetime, get_links_per_worker

ANSWERS = ['Y','M','N','NS']

#TODO : filter by knowledge?? YES /NO/ ALL
def get_data_for_query( query = None, prefix= None):
    db = connect_to_db()
    mycursor = db.cursor()
    links = get_links_per_worker(mycursor)

    if query == None:
        sql_query_string = "SELECT user_id,query,treatment,problem, knowledge, start, end, sequence, feedback, reason FROM serp.exp_data"
        filename = "feedback_all"
    else:
        sql_query_string = "SELECT user_id,query,treatment,problem, knowledge, start, end, sequence, feedback, reason FROM serp.exp_data where query = '"+query +"'"
        filename = "feedback_"+ query
    if prefix:
        filename += '_prefix_'+str(prefix)+'.csv'
    else:
        filename += '.csv'
    mycursor.execute(sql_query_string)

    myresult = mycursor.fetchall()

    results = dict()
    possible_answers = set()
    for x in myresult:
        user = x[0]
        query = x[1]
        treatment_answer = x[2]
        condition_answer = x[3]
        prev_know = x[4]
        start = x[5]
        end = x[6]

        if not end:
            continue
        sdt = string_to_datetime(start)
        edt = string_to_datetime(end)
        time = get_time_spent(sdt, edt)
        reason = x[9]
        if filter_user(user, query, treatment_answer, condition_answer, time, prev_know, reason):
            continue
        if user not in links:
            continue

        config = x[7]
        if prefix:
            config = config[:prefix]
        answer = x[8]
        print(user)
        if config not in results:
            results[config] = {'Y':0,'N':0,'M':0,'NS':0}
            for i in range (1,11):
                results[config]['link'+str(i)] = 0
        results[config][answer] += 1
        user_links = links[user]
        for l in user_links:
            results[config]['link'+str(l)] += 1


    with open('../resources/output//'+filename, 'w', newline='') as csvfile:
        fieldnames = ['sequence'] + ANSWERS
        for i in range(1, 11):
            fieldnames.append('link'+str(i))

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for config, counters in results.items():
            row = {'sequence': config}
            row.update(counters)
            for answer in possible_answers:
                if answer not in row:
                    row[answer] = 0
            writer.writerow(row)


if __name__ == "__main__":
    #get_data_for_query('Does Omega Fatty Acids treat Adhd')
    get_data_for_query(prefix=1)
    #get_data_for_query()



