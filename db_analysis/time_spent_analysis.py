import csv
import statistics

from datetime import datetime
from db_analysis.db_setup import connect_to_db


def get_time_diff(start, end):
    start_datetime = datetime.strptime(start, '%m/%d/%Y, %I:%M:%S %p')
    end_datetime = datetime.strptime(end, '%m/%d/%Y, %I:%M:%S %p')
    time_spent = (end_datetime - start_datetime).total_seconds() / 60
    return time_spent


def time_spent_in_serp_sequence_feedback( query = None):
    if query == None:
        sql_query_string = "SELECT sequence, feedback, start, end FROM serp.exp_data"
        filename = "time_spent_per_feedback_and_sequence" + '_all.csv'
    else:
        sql_query_string = "SELECT sequence, feedback, start, end FROM serp.exp_data where query = '" + query + "'"
        filename = 'time_spent_per_feedback_and_sequence' + '_' + query + '.csv'
    db = connect_to_db()
    mycursor = db.cursor()
    mycursor.execute(sql_query_string)
    myresult = mycursor.fetchall()

    results = dict()
    for x in myresult:
        sequence = x[0]
        feedback = x[1]
        start = x[2]
        end = x[3]
        #TODO zero padded?
        if end == 'N/A':
            continue
        if sequence not in results:
            results[sequence] = dict()
        if feedback not in results[sequence]:
            results[sequence][feedback] = []
        time_spent = get_time_diff(start, end)
        results[sequence][feedback].append(time_spent)

    with open('../output//' + filename, 'w', newline='') as csvfile:
        fieldnames = ['sequence', 'feedback', 'avg_minutes_spent']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for sequence, feedbacks in results.items():
            for feedback, times in feedbacks.items():
                avg_time = statistics.mean(times)
                avg_time_formatted = float("{:.2f}".format(avg_time))
                row = {'sequence': sequence, 'feedback': feedback, 'avg_minutes_spent':avg_time_formatted}
                writer.writerow(row)


def time_spent_in_serp(field, query = None):
    if query == None:
        sql_query_string = "SELECT " + field + ", start, end FROM serp.exp_data"
        filename = "time_spent_per_" + field + '_all.csv'
    else:
        sql_query_string = "SELECT " + field +", start, end FROM serp.exp_data where query = '" + query + "'"
        filename = 'time_spent_per_' + field + '_'+ query + '.csv'
    db = connect_to_db()
    mycursor = db.cursor()
    mycursor.execute(sql_query_string)
    myresult = mycursor.fetchall()

    results = dict()
    for x in myresult:
        grouped_field = x[0]
        start = x[1]
        end = x[2]
        #TODO zero padded?
        if end == 'N/A':
            continue
        if grouped_field not in results:
            results[grouped_field] = []
        time_spent = get_time_diff(start, end)
        results[grouped_field].append(time_spent)

    with open('../output//' + filename, 'w', newline='') as csvfile:
        fieldnames = [field, 'avg_minutes_spent']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for ft, times in results.items():
            avg_time = statistics.mean(times)
            avg_time_formatted = float("{:.2f}".format(avg_time))
            row = {field: ft, 'avg_minutes_spent':avg_time_formatted}
            writer.writerow(row)

if __name__ == "__main__":
    #time_spent_in_serp('sequence', 'Does Omega Fatty Acids treat Adhd')
    #time_spent_in_serp('sequence')
    #time_spent_in_serp('feedback')
    #time_spent_in_serp('feedback', 'Does Omega Fatty Acids treat Adhd')
    #time_spent_in_serp('feedback')
    #time_spent_in_serp('sequence')
    time_spent_in_serp_sequence_feedback('Does Omega Fatty Acids treat Adhd')



