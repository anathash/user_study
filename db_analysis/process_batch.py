import csv
import os
import xml.etree.ElementTree as ET
import requests
from datetime import date

import xmltodict

from db_analysis.utils import connect_to_db, get_time_diff, get_time_diff_from_actions, \
    get_links_per_worker, get_workers_with_no_link,  filter_user_new, \
    get_time_spent
from turk_api import AMT_api, get_done_workers

BATCH_FILE_PREFIX = '../resources/batch results/batch'
NUM_MNUTES_FOR_BONUS = 2
NUM_LINKS_FOR_BONUS = 1
BONUS = 1

payment_report_file = '../resources/reports//user_study_payment_report.csv'

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




def process_workers(dbcursor, worker_ids, start_date):
    bonus_workers = set()
    num_links_per_workers = get_links_per_worker(dbcursor, worker_ids)
    sql_exp_data_query_string = "SELECT *  FROM serp.exp_data where user_id in " + worker_ids + " and start >= '" + start_date + "'"
    print(sql_exp_data_query_string)
    dbcursor.execute(sql_exp_data_query_string)
    exp_data = dbcursor.fetchall()
    black_list = []
    worker_to_exp = {}
    for r in exp_data:
        exp_id = r[0]
        worker_id = r[1]
        worker_to_exp[worker_id] = exp_id
        start = r[7]
        end = r[8]
        answer_treatment = r[12].strip()
        answer_condition = r[13].strip()
        query = r[5]
        if filter_user_new(worker_id, query, answer_treatment, answer_condition, start, end, r[9], r[11], filter_prev_know= False):
            black_list.append(worker_id)
        else:
            if not end:
                continue
            else:
                time_spent = get_time_spent(start, end)

            if time_spent >= NUM_MNUTES_FOR_BONUS:
                #if worker_id in num_links_per_workers and len(num_links_per_workers[worker_id][exp_id]) >= NUM_LINKS_FOR_BONUS:
                if worker_id in num_links_per_workers and len(num_links_per_workers[worker_id]) >= NUM_LINKS_FOR_BONUS:
                    bonus_workers.add(worker_id)
    done_workers = get_done_workers(worker_ids)
    return worker_to_exp, done_workers, black_list, bonus_workers


def get_paid_workers():
    paid_workers = {}
    with open(payment_report_file,  newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            paid_workers[row['WorkerId']] = row['date']
    return paid_workers


def process_survery_code(api, amazon_results, ids_sql_string, start_date, base_payment, from_batch, to_batch = None, bonus_payment = 0, update_files = False):

    today = date.today()

    workers_to_pay = []
    workers_not_finished = []
    db = connect_to_db('biu')
    mycursor = db.cursor()

    worker_to_exp, done_workers, black_list, bonus_dict = process_workers(mycursor,ids_sql_string, start_date)
    for u in black_list:
        api.assign_serp_qualification(u, 'SERP black list', 1)
    for u in done_workers:
        api.assign_serp_qualification(u, 'SERP done', 1)
#    if bonus_payment:
#        bonus_dict = get_bonuses_workers(mycursor,ids_sql_string)
    print(str(len(bonus_dict)) + ' bonuses')
    exp_id_string = str(list(worker_to_exp.values()))
    exp_id_string = '(' + exp_id_string[1:-1] + ')'
    #sql_query_string = "SELECT * FROM serp.user_config where amazon_id in " + ids_sql_string
    #sql_query_string = "SELECT * FROM serp.exp_verification_codes where amazon_id in " + ids_sql_string + " and exp_id in " + exp_id_string
    sql_query_string = "SELECT * FROM serp.exp_verification_codes where exp_id in " + exp_id_string
    print(sql_query_string)

    mycursor.execute(sql_query_string)
    dbresult = mycursor.fetchall()
    for r in dbresult:
        exp_id = r[0]
        amazon_id_db = r[1]
        assert(worker_to_exp[amazon_id_db] == exp_id)
        survey_code_db = r[2]
        answer = amazon_results[amazon_id_db]['Answer']
        answer_dict = xmltodict.parse(answer)
        worker_supplied_survey_code = answer_dict['QuestionFormAnswers']['Answer']['FreeText']
#        worker_supplied_survey_code = amazon_results[amazon_id_db]['Answer.surveycode']
        assignment_id = amazon_results[amazon_id_db]['AssignmentId']
        api.assign_serp_qualification(amazon_id_db, 'SERP', 1)
        if worker_supplied_survey_code == survey_code_db:
            workers_to_pay.append(amazon_id_db)
            api.accept_assignment(assignment_id)
            if amazon_id_db in bonus_dict:
                api.send_bonus(amazon_id_db, assignment_id, assignment_id)
        else:
            workers_not_finished.append(amazon_id_db)
            api.reject_assignment(assignment_id, 'The survey code entered does not match our records')
            api.assign_serp_qualification(amazon_id_db, 'SERP black list', 1)

    if to_batch:
        amazon_report_csv_filename = BATCH_FILE_PREFIX + str(from_batch) + '_to_' + str(to_batch) + '_approve_reject_report.csv'
    else:
        amazon_report_csv_filename = BATCH_FILE_PREFIX + str(from_batch) + '_approve_reject_report.csv'

    if not update_files:
        return

    with open(amazon_report_csv_filename, 'w', newline='') as amazon_report_csv,\
            open(payment_report_file, 'a', newline='') as payment_report_csv:

#        if update_paymets_file:
        if update_files:
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

                    if update_files:
                        payment_report_file_writer.writerow({'WorkerId': worker_id, 'payment':str(worker_payment)+'$','date': today})
                else:
                    amazon_report_writer.writerow({'WorkerId': worker_id, 'AssignmentId': assignment_id,
                                                   'HITId': hit_id,
                                                   'Reject': 'Worker already paid for assignment'})
            elif worker_id in workers_not_finished:
                amazon_report_writer.writerow({'WorkerId': worker_id,'AssignmentId': assignment_id,
                                                     'HITId': hit_id, 'Reject':'The survey code entered does not match our records'})


def get_worker_id_list_for_hit(api, hit_id):
    hit_results = api.get_hit_results(hit_id)
    amazon_results = {}
    ids_sql_string = '('
    for row in hit_results:
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


def process_hit(hit_id, start_date):
    api = AMT_api()
    payment = 1.4
    bonus = 1.4
    amazon_results, ids_sql_string = get_worker_id_list_for_hit(api, hit_id)
    process_survery_code(api, amazon_results, ids_sql_string, start_date, payment, hit_id, bonus, True)

def assign_qual():
    hit_id = '3JHB4BPSFKAWLVAMJB3BBPIM1RUQ9T'
    api = AMT_api()
    response = api.client.list_assignments_for_hit(
        HITId=hit_id,
        #            NextToken='string',
        MaxResults=100,
        AssignmentStatuses=['Approved']
    )
    for r in response['Assignments']:
        api.assign_serp_qualification(r['WorkerId'])


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
        print('All workers entered at least one link', )
    else:
        print(ignore)
#

def add_assignemnts(num):
    url = 'http://mturk-requester.us-east-1.amazonaws.com'
    myobj = {'HITId': '3VO4XFFP15NS7CMT4E5RXTRG6SXQ7Q', 'HITNumberOfAdditionalAssignments':str(num)}

    x = requests.post(url, data=myobj)


    print(x.text)


if __name__ == "__main__":
    #assign_qual()
    #add_assignemnts(5)
    #print_workers_with_no_links()
    #process_batch(18,21)
    #process_batch(22,26)
    #get_data_for_query('Does Omega Fatty Acids treat Adhd')
    process_hit('39O6Z4JLX2YERZO18Q3ZXF3CYJSXV8','2022-11-30 00:00:00')

    #db = connect_to_db('biu')
    #dbcursor = db.cursor()
    #done_workers, black_list, bonus_dict = process_workers(dbcursor,"('A1H48DL2CMYH7U','A1I7H6RDJS4EKN','A1OZPLHNIU1519','A1PTH9KTRO06EG','A21W69Z63UQP69','A29IBNSV45WIMQ','A2Z94V60G6K26P','AQXSE8R0MDUWN','ARJ7XWSK0Y8FJ')", '2022-03-08 00:58:07')
    #print('bonus = ' + str(bonus_dict))
    #print(black_list)

    #api = AMT_api()
    #assignemnts = ['3TPWUS5F8925P9B5X6HAUNX8IXRWC8','3ZV9H2YQQD8HC9FM4D691KTECKF3WI','3MMN5BL1WZ5L7XL80B0MSMRNEI0M39','3QY5DC2MXRLZ0H6AT8SAK5XEUUDFU8','3NG53N1RLVKDTXOR48NA07TTZ988PO','324G5B4FB39652FODIGE76WQGHS07O','3Y9N9SS8LYCI33FVNI1J9W4T54FD3P','3UNH76FOCS6MN0IWPWTCIGN0NJZYMV','3GFK2QRXX9IKQO2QIWU2GHRYQGA5WE']
    #api.send_bonus('AQXSE8R0MDUWN', '3UNH76FOCS6MN0IWPWTCIGN0NJZYMV', '3UNH76FOCS6MN0IWPWTCIGN0NJZYMV')
    #api.send_bonus('A1PTH9KTRO06EG', '3QY5DC2MXRLZ0H6AT8SAK5XEUUDFU8', '3QY5DC2MXRLZ0H6AT8SAK5XEUUDFU8')
    #api.send_bonus('A1OZPLHNIU1519', '3MMN5BL1WZ5L7XL80B0MSMRNEI0M39', '3MMN5BL1WZ5L7XL80B0MSMRNEI0M39')

    #api.send_bonus('A1I7H6RDJS4EKN', '3ZV9H2YQQD8HC9FM4D691KTECKF3WI', '3ZV9H2YQQD8HC9FM4D691KTECKF3WI')
    #for a in assignemnts:
    #    api.accept_assignment('3ZV9H2YQQD8HC9FM4D691KTECKF3WI', override_rejection = True)

 #   api.send_bonus('A1I7H6RDJS4EKN', '3ZV9H2YQQD8HC9FM4D691KTECKF3WI', '3ZV9H2YQQD8HC9FM4D691KTECKF3WI')


