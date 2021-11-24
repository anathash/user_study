import csv
from db_analysis.db_setup import connect_to_db


def get_exp_clicks(exp_ids= None):
    if exp_ids:
        ids_sql_tring = '('
        for id in exp_ids:
            ids_sql_tring += "'" + id + "',"
        ids_sql_tring = ids_sql_tring[:-1]
        ids_sql_tring += ')'
        sql_query_string = "SELECT exp_id, link_id FROM serp.user_action WHERE action = 'click link' and exp_id in " + ids_sql_tring
    else:
        sql_query_string = "SELECT exp_id, link_id FROM serp.user_action WHERE action = 'click link'"
    db = connect_to_db()
    mycursor = db.cursor()
    mycursor.execute(sql_query_string)
    results = mycursor.fetchall()
    expid_to_clicks = {}
    for x in results:
        exp_id = x[0]
        link_id = x[1]
        if exp_id not in expid_to_clicks:
            expid_to_clicks[exp_id] = []
        expid_to_clicks[exp_id].append(link_id)
    return expid_to_clicks


def link_clicked_per_rank(field, query = None):
    if query == None:
        sql_query_string = "SELECT exp_id, end," + field + " FROM serp.exp_data"
        filename = "ranking_bias_per_" + field + '_all.csv'
    else:
        sql_query_string = "SELECT exp_id, end," + field +" FROM serp.exp_data where query = '" + query + "'"
        filename = 'ranking_bias_per_' + field + '_'+ query + '.csv'
    db = connect_to_db()
    mycursor = db.cursor()
    mycursor.execute(sql_query_string)
    myresult = mycursor.fetchall()

    if query:
        exp_ids = []
        for x in myresult:
            exp_ids.append(x[0])

        expid_to_clicks = get_exp_clicks(exp_ids)
    else:
        expid_to_clicks = get_exp_clicks()

    results = dict()
    for x in myresult:
        exp_id = x[0]
        end = x[1]
        grouped_field = x[2]
        if end == 'N/A': #user did not finish experiment
            continue
        if grouped_field not in results:
            results[grouped_field] = {"num_exps":0}
            results[grouped_field].update({x:0 for x in range(1,11)})
        results[grouped_field]['num_exps'] += 1
        clicked_ranks = expid_to_clicks[exp_id]
        for rank in clicked_ranks:
            results[grouped_field][rank] += 1


    with open('../output//' + filename, 'w', newline='') as csvfile:
        fieldnames = ['sequence']
        fieldnames .extend([x for x in range(1,11)])
        fieldnames .append('num_exps')
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for config, stats in results.items():
            for i in range(1,11):
                stats[i] /= stats['num_exps']
            row = {'sequence': config}
            row.update(stats)
            writer.writerow(row)


def link_clicked_per_rank_feedback_and_sequence(query = None):
    if query == None:
        sql_query_string = "SELECT exp_id, sequence, feedback, end FROM serp.exp_data"
        filename = "ranking_bias_per_feedback_and_sequence_all.csv"
    else:
        sql_query_string = "SELECT exp_id, sequence, feedback, end FROM serp.exp_data where query = '" + query + "'"
        filename = 'ranking_bias_per_feedback_and_sequence_' + query + '.csv'
    db = connect_to_db()
    mycursor = db.cursor()
    mycursor.execute(sql_query_string)
    myresult = mycursor.fetchall()

    if query:
        exp_ids = []
        for x in myresult:
            exp_ids.append(x[0])

        expid_to_clicks = get_exp_clicks(exp_ids)
    else:
        expid_to_clicks = get_exp_clicks()

    results = dict()
    for x in myresult:
        exp_id = x[0]
        sequence = x[1]
        feedback = x[2]
        end = x[3]
        if end == 'N/A': #user did not finish experiment
            continue
        if sequence not in results:
            results[sequence] = dict()
        if feedback not in results[sequence]:
            results[sequence][feedback] = {"num_exps":0}
            results[sequence][feedback].update({x:0 for x in range(1,11)})
        results[sequence][feedback]['num_exps'] += 1
        clicked_ranks = expid_to_clicks[exp_id]
        for rank in clicked_ranks:
            results[sequence][feedback][rank] += 1


    with open('../output//' + filename, 'w', newline='') as csvfile:
        fieldnames = ['sequence', 'feedback']
        fieldnames .extend([x for x in range(1,11)])
        fieldnames .append('num_exps')
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for config, feedbacks in results.items():
            for feedback, stats in feedbacks.items():
                for i in range(1,11):
                    stats[i] /= stats['num_exps']
                row = {'sequence': config, 'feedback':feedback}
                row.update(stats)
                writer.writerow(row)


if __name__ == "__main__":
#    link_clicked_per_rank('feedback', 'Does Omega Fatty Acids treat Adhd')
    #link_clicked_per_rank('feedback', )
    link_clicked_per_rank_feedback_and_sequence()



