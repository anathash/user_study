import csv

from db_analysis.utils import connect_to_db,  get_time_diff, get_time_diff_from_actions, \
    TREATMENT_CORRECT_ANSWERS, CONDITION_CORRECT_ANSWERS, get_num_links_per_worker, get_links_entered_by_worker
from process_batch import get_worker_id_list, BATCH_FILE_PREFIX


#TODO : filter by knowledge?? YES /NO/ ALL
def generate_user_behaviour_table(batch_number=None):
    db = connect_to_db()
    mycursor = db.cursor()
    #links = get_links_entered_in_exps(mycursor)

    if batch_number:
        amazon_results, ids_sql_string = get_worker_id_list(batch_number)
        exp_data_query_string = "SELECT * FROM serp.exp_data where user_id in " + ids_sql_string
        links = get_links_entered_by_worker(mycursor, ids_sql_string)

    else:
        exp_data_query_string = "SELECT * FROM serp.exp_data"
        links = get_links_entered_by_worker(mycursor)

    mycursor.execute(exp_data_query_string)
    myresult = mycursor.fetchall()
    exp_data = []
    for x in myresult:
        url = x[2][35:]
        start = x[7]
        end = x[8]
        exp_id = x[0]
        time_diff_actions = get_time_diff_from_actions(mycursor, x[0])
        time_diff_actions = "%.2f" % round(time_diff_actions, 2)
        if end:
            time_diff_exp = get_time_diff(start, end)
            time_diff_exp = "%.2f" % round(time_diff_exp, 2)
        else:
            time_diff_exp = time_diff_actions
        answer_treatment = x[12].strip()
        answer_condition = x[13].strip()
        query = x[5]
        treatment_answer_correct = 1 if answer_treatment == TREATMENT_CORRECT_ANSWERS[query] else 0
        condition_answer_correct = 1 if answer_condition == CONDITION_CORRECT_ANSWERS[query] else 0
        worker_id = x[1]
        entry = {'exp_id': exp_id, 'WorkerId':worker_id, 'url':url,'start_time':start, 'search_time_exp': time_diff_exp,'search_time_actions': time_diff_actions,
                 'knowledge':x[9] ,'feedback':x[10],'reason':x[11],'treatment_answer_correct':treatment_answer_correct,'condition_answer_correct':condition_answer_correct,'comments':x[14] }
        num_links = 0
        for i in range(1, 11):
            #link_pressed = links[exp_id][i]
            if worker_id in links:
                link_pressed = links[worker_id][i]
                entry['link'+str(i)] =link_pressed
                num_links += link_pressed

        entry['links_pressed'] = num_links
        exp_data.append(entry)
    if not batch_number:
        filename = '../resources/output//user_behaviour.csv'
    else:

        filename = BATCH_FILE_PREFIX+str(batch_number)+'_user_behaviour.csv'
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['exp_id', 'WorkerId', 'url','start_time', 'search_time_exp','search_time_actions','links_pressed','knowledge','feedback','reason','treatment_answer_correct','condition_answer_correct','comments',
                      'link1', 'link2', 'link3', 'link4', 'link5', 'link6', 'link7', 'link8', 'link9', 'link10']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in exp_data:
            writer.writerow(row)


if __name__ == "__main__":
    generate_user_behaviour_table(12)

