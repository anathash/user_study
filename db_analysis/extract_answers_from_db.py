import csv

from db_analysis.utils import connect_to_db

#TODO : filter by knowledge?? YES /NO/ ALL
def get_data_for_query( query = None):
    if query == None:
        sql_query_string = "SELECT sequence, feedback FROM serp.exp_data"
        filename = "feedback_all.csv"
    else:
        sql_query_string = "SELECT sequence, feedback FROM serp.exp_data where query = '"+query +"'"
        filename = "feedback_"+ query+'.csv'
    db = connect_to_db()
    mycursor = db.cursor()
    mycursor.execute(sql_query_string)

    myresult = mycursor.fetchall()

    results = dict()
    possible_answers = set()
    for x in myresult:
        config = x[0]
        answer = x[1]
        possible_answers.add(answer)
        if config not in results:
            results[config] = dict()
        if answer not in results[config]:
            results[config][answer] = 0
        results[config][answer] += 1

    with open('../resources/output//'+filename, 'w', newline='') as csvfile:
        fieldnames = ['sequence'] + list(possible_answers)

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


