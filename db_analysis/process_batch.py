import csv
import os
import requests
from datetime import date

from db_analysis.utils import connect_to_db, get_time_diff, get_time_diff_from_actions, \
    get_links_per_worker, get_workers_with_no_link

BATCH_FILE_PREFIX = '../resources/batch results/batch'
NUM_MNUTES_FOR_BONUS = 2
NUM_LINKS_FOR_BONUS = 1
BONUS = 1

payment_report_file = '../resources/batch results//user_study_payment_report.csv'

def get_worker_id_list(from_batch, to_batch=None):
    amazon_results = {}
    ids_sql_string = '('
    if not to_batch:
        to_batch = from_batch
    for batch_number in range(from_batch, to_batch+1):
        fname = BATCH_FILE_PREFIX + str(batch_number) + '_amazon.csv'
        if not os.path.isfile(fname):
            continue
        with open(fname, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            # get survey codes from db
            for row in reader:
                worker_id = row['WorkerId']
                if worker_id in amazon_results:
                    print('duplicate!!')
                    print(row)
                else:
                    amazon_results[worker_id] = row
                    ids_sql_string += "'" + worker_id + "',"

            ids_sql_string = ids_sql_string[:-1]
    ids_sql_string += ')'
    return amazon_results, ids_sql_string


#TODO: update payments csv
def process_bonus(batch_number, worker_ids):
    db = connect_to_db()
    dbcursor = db.cursor()
    num_links_pers_workers = get_links_per_worker(dbcursor, worker_ids)

    bonus_workers = []

    sql_exp_data_query_string = "SELECT exp_id, user_id, start, end  FROM serp.exp_data where user_id in " + worker_ids
    dbcursor.execute(sql_exp_data_query_string)
    exp_data = dbcursor.fetchall()
    for r in exp_data:
        exp_id = r[0]
        worker_id = r[1]
        start = r[2]
        end = r[3]
        if not end:
            time_spent = get_time_diff_from_actions(dbcursor, exp_id)
        else:
            time_spent = get_time_diff(start, end)

        if time_spent >= NUM_MNUTES_FOR_BONUS:
            if len(num_links_pers_workers[worker_id][exp_id]) >= NUM_LINKS_FOR_BONUS:
                bonus_workers.append(worker_id)

    with open(BATCH_FILE_PREFIX+str(batch_number)+'_bonus.csv', 'w', newline='') as bonus_csv:
        writer = csv.DictWriter(bonus_csv, fieldnames=['WorkerId'])
        writer.writeheader()
        for worker_id in bonus_workers:
            writer.writerow({'WorkerId': worker_id})

def get_bonuses_workers(dbcursor, worker_ids):
    bonus_workers = set()
    num_links_per_workers = get_links_per_worker(dbcursor, worker_ids)
    sql_exp_data_query_string = "SELECT exp_id, user_id, start, end  FROM serp.exp_data where user_id in " + worker_ids
    dbcursor.execute(sql_exp_data_query_string)
    exp_data = dbcursor.fetchall()
    for r in exp_data:
        exp_id = r[0]
        worker_id = r[1]
        start = r[2]
        end = r[3]
        if not end:
            time_spent = get_time_diff_from_actions(dbcursor, exp_id)
        else:
            time_spent = get_time_diff(start, end)

        if time_spent >= NUM_MNUTES_FOR_BONUS:
            #if worker_id in num_links_per_workers and len(num_links_per_workers[worker_id][exp_id]) >= NUM_LINKS_FOR_BONUS:
            if worker_id in num_links_per_workers and len(num_links_per_workers[worker_id]) >= NUM_LINKS_FOR_BONUS:
                bonus_workers.add(worker_id)
    return bonus_workers


def get_paid_workers():
    paid_workers = {}
    with open(payment_report_file,  newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            paid_workers[row['WorkerId']] = row['date']
    return paid_workers

def process_survery_code( amazon_results, ids_sql_string,  base_payment,from_batch, to_batch = None, bonus_payment = 0, update_paymets_file = True):

    today = date.today()

    workers_to_pay = []
    workers_not_finished = []
    db = connect_to_db()
    mycursor = db.cursor()
    bonus_dict = {}
    if bonus_payment:
        bonus_dict = get_bonuses_workers(mycursor,ids_sql_string)

    sql_query_string = "SELECT * FROM serp.user_config where amazon_id in " + ids_sql_string

    mycursor.execute(sql_query_string)
    dbresult = mycursor.fetchall()
    for r in dbresult:
        amazon_id_db = r[1]
        survey_code_db = r[2]
        worker_supplied_survey_code = amazon_results[amazon_id_db]['Answer.surveycode']
        if worker_supplied_survey_code == survey_code_db:
            workers_to_pay.append(amazon_id_db)
        else:
            workers_not_finished.append(amazon_id_db)


    if to_batch:
        amazon_report_csv_filename = BATCH_FILE_PREFIX + str(from_batch) + '_to_' + str(to_batch) + '_approve_reject_report.csv'
    else:
        amazon_report_csv_filename = BATCH_FILE_PREFIX + str(from_batch) + '_approve_reject_report.csv'

    with open(amazon_report_csv_filename, 'w', newline='') as amazon_report_csv,\
            open(payment_report_file, 'a', newline='') as payment_report_csv:

        if update_paymets_file:
            payment_report_file_writer = csv.DictWriter(payment_report_csv, fieldnames=['WorkerId','payment','date'])

        amazon_report_writer = csv.DictWriter(amazon_report_csv, fieldnames=['WorkerId', 'AssignmentId','HITId','Approve', 'Reject', 'Bonus'])
        amazon_report_writer.writeheader()
        paid_workers = get_paid_workers()
        for worker_id, row in amazon_results.items():
            assignment_id = row['AssignmentId']
            hit_id = row['HITId']
            if worker_id in workers_to_pay:
                if worker_id not in paid_workers:

                    if worker_id in bonus_dict:
                        worker_payment = base_payment +  bonus_payment
                        bonus = 'X'
                    else:
                        worker_payment = base_payment
                        bonus = ''

                    amazon_report_writer.writerow({'WorkerId': worker_id, 'AssignmentId': assignment_id,
                                                         'HITId': hit_id, 'Approve':'x', 'Bonus':bonus})

                    if update_paymets_file:
                        payment_report_file_writer.writerow({'WorkerId': worker_id, 'payment':str(worker_payment)+'$','date': today})
                else:
                    amazon_report_writer.writerow({'WorkerId': worker_id, 'AssignmentId': assignment_id,
                                                   'HITId': hit_id,
                                                   'Reject': 'Worker already paid for assignment'})
            elif worker_id in workers_not_finished:
                amazon_report_writer.writerow({'WorkerId': worker_id,'AssignmentId': assignment_id,
                                                     'HITId': hit_id, 'Reject':'The survey code entered does not match our records'})


def process_batch(from_batch, to_batch=None):
    payment = 1.4
    bonus = 1.4
    amazon_results, ids_sql_string = get_worker_id_list(from_batch,to_batch )
    process_survery_code(amazon_results, ids_sql_string, payment, from_batch,to_batch,   bonus, True)
  #  process_bonus(batch_number, ids_sql_string)


def print_workers_with_no_links(from_batch=None, to_batch=None):
    db = connect_to_db()
    dbcursor = db.cursor()
    if from_batch:
        amazon_results, ids_sql_string = get_worker_id_list(from_batch, to_batch)
        ignore = get_workers_with_no_link(dbcursor, ids_sql_string)
    else:
        ignore = get_workers_with_no_link(dbcursor)
    if len(ignore) == 0:
        print('All workers entered at least one link')
    else:
        print(ignore)
#

def add_assignemnts(num):
    url = 'http://mturk-requester.us-east-1.amazonaws.com'
    myobj = {'HITId': '3VO4XFFP15NS7CMT4E5RXTRG6SXQ7Q', 'HITNumberOfAdditionalAssignments':str(num)}

    x = requests.post(url, data=myobj)

    print(x.text)


if __name__ == "__main__":
    #add_assignemnts(5)
    #print_workers_with_no_links()
    process_batch(18,21)
    #get_data_for_query('Does Omega Fatty Acids treat Adhd')



