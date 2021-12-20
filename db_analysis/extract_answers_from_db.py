import csv

from db_analysis.utils import connect_to_db, get_time_spent, filter_user

ANSWERS = ['Y','M','N']

#TODO : filter by knowledge?? YES /NO/ ALL
def get_data_for_query( query = None):
    if query == None:
        sql_query_string = "SELECT * FROM serp.exp_data"
        filename = "feedback_all.csv"
    else:
        sql_query_string = "SELECT * FROM serp.exp_data where query = '"+query +"'"
        filename = "feedback_"+ query+'.csv'
    db = connect_to_db()
    mycursor = db.cursor()
    mycursor.execute(sql_query_string)

    myresult = mycursor.fetchall()

    results = dict()
    possible_answers = set()
    for x in myresult:
        user = x[1]
        query = [5]
        treatment_answer = x[12]
        condition_answer = x[13]
        prev_know = x[9]
        start = x[7]
        end = x[8]
        time = get_time_spent(start, end)
        if filter_user(user, query, treatment_answer, condition_answer, time, prev_know):
            continue

        config = x[6]
        answer = x[11]
        if config not in results:
            results[config] = dict()
        if answer not in results[config]:
            results[config][answer] = 0
        results[config][answer] += 1

    with open('../resources/output//'+filename, 'w', newline='') as csvfile:
        fieldnames = ['sequence'] + ANSWERS

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
    get_data_for_query()



