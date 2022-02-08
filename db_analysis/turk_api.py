import csv
import datetime

import boto3
import botocore
import xmltodict

from utils import connect_to_db, string_to_datetime, get_time_spent, test_users_names_prefix, unsatisfactory

BLACK_LIST_QUAL = '3FZJH8GF5LMVIHAO0K827A4HTNUZVQ'
DONE_WORKERS_QUAL = '31IGUM6XYQD8HNHQ7FFNCMVHNGKLLA'
SERP = '3VG0EHWASUFRE500CC3BLPY5SRAOQH'

class AMT_api:
    def __init__(self):
        create_hits_in_production = True
        environments = {
            "production": {
                "endpoint": "https://mturk-requester.us-east-1.amazonaws.com",
                "preview": "https://www.mturk.com/mturk/preview"
            },
            "sandbox": {
                "endpoint":
                    "https://mturk-requester-sandbox.us-east-1.amazonaws.com",
                "preview": "https://workersandbox.mturk.com/mturk/preview"
            },
        }
        mturk_environment = environments["production"] if create_hits_in_production else environments["sandbox"]
        session = boto3.Session(profile_name='mturk')
        self.client = session.client(
            service_name='mturk',
            region_name='us-east-1',
            endpoint_url=mturk_environment['endpoint'],
        )

        print(self.client.get_account_balance()['AvailableBalance'])

    @staticmethod
    def check_response(response):
        code = response['ResponseMetadata']['HTTPStatusCode']
        if code != 200:
            print(response)
            return False
        return True

    def is_hit_completed(self, hit_id):
        response = self.client.get_hit(
            HITId=hit_id
        )
        if not self.check_response(response):
            print('get_hit_status failed for hit: ' + hit_id)
        return response['HIT']['NumberOfAssignmentsPending'] == 0



    def get_hit_results(self, hit_id):
        response = self.client.list_assignments_for_hit(
            HITId=hit_id,
#            NextToken='string',
            MaxResults=100,
            AssignmentStatuses= ['Submitted']
        )
        if not self.check_response(response):
            print('get_hit_results failed for hit: ' + hit_id)
        return response['Assignments']

    def assign_serp_qualification(self, worker_id, qual_type_id,  value):
        try:
            response = self.client.associate_qualification_with_worker(
                QualificationTypeId=qual_type_id,#'3VG0EHWASUFRE500CC3BLPY5SRAOQH',
                WorkerId=worker_id,
                IntegerValue=value,
                SendNotification=False
            )
            if self.check_response(response):
                print('Assigned SERP qualification ' + str(value)+ ' to worker: ' + worker_id)

            else:
                print('assign_serp_qualification failed for worker: ' + worker_id)
        except botocore.exceptions.ClientError:
            print('assign qualification failed for worker: ' + worker_id)

    def send_bonus(self, worker_id, assignment_id, unique_token):
        try:
            response = self.client.send_bonus(
                WorkerId=worker_id,
                BonusAmount='1',
                AssignmentId= assignment_id,
                Reason='Qualified answer for health search results survey',
                UniqueRequestToken=unique_token
            )
            if self.check_response(response):
                print('Bonus granted to worker ' + worker_id)
            else:
                print('send_bonus failed for worker: ' + worker_id)
        except:
            print('send_bonus failed for worker: ' + worker_id)


    def reject_assignment(self, assignment_id, feedback):
        response = self.client.reject_assignment(
            AssignmentId=assignment_id,
            RequesterFeedback=feedback
        )
        if self.check_response(response):
            print('Rejected assignment' + assignment_id)
        else:
            print('reject_assignment failed for assignment_id: ' + assignment_id)

    def accept_assignment(self, assignment_id, override_rejection = False ):
        response = self.client.approve_assignment(
            AssignmentId=assignment_id,
            RequesterFeedback='',
            OverrideRejection= override_rejection
        )
        if self.check_response(response):
            print('Accepted assignment ' + assignment_id)
        else:
            print('accept_assignment failed for assignment_id: ' + assignment_id)

    def add_assignments_to_hit(self, hit_id, num_assignments ,token):
        response = self.client.create_additional_assignments_for_hit(
            HITId=hit_id,
            NumberOfAdditionalAssignments=num_assignments,
            UniqueRequestToken=token
        )
        if self.check_response(response):
            print('Added ' + str(num_assignments) + ' to hit ' + hit_id)
        else:
            print('add_assignments_to_hit failed for hit_id: ' + hit_id)



def assign_quals():
    api = AMT_api()
    with open('../resources/batch results/qualified_users.csv',
              newline='') as csvFile:
        reader = csv.DictReader(csvFile)
        for row in reader:
            id = row['workerID'].strip()
            try:
                api.assign_serp_qualification(id)
                print(id)
            except Exception as e:
                print(e)


def pay_bonuses(transaction_file, payemnt_file):
    bonus_paid = {}
    assignment_paid = {}
    with open(transaction_file, newline='', encoding='utf8') as trasnCSV:
        reader = csv.DictReader(trasnCSV)
        for row in reader:
            if row['Transaction Type'] == 'BonusPayment':
                bonus_paid[row['Recipient ID']] = row['Assignment ID']
            elif row['Transaction Type'] == 'AssignmentPayment':
                assignment_paid[row['Recipient ID']] = row['Assignment ID']

    with open(payemnt_file, newline='', encoding='utf8') as payCSV:
        reader = csv.DictReader(payCSV)
        for row in reader:
            worker_id = row['WorkerId']
            if row['payment'] =='2.8$' and worker_id not in bonus_paid:
                if worker_id not in assignment_paid:
                    print('worker ' + worker_id + ' not paid for assignment')
                print(row)
                #response = send_bonus(client=client, worker_id= worker_id, assignment_id =assignment_paid[worker_id], unique_token = worker_id)
                #print(response)


def set_blacklist_qual_for_all(db_name):
    db = connect_to_db(db_name)
    dbcursor = db.cursor()
    #user, query, treatment_answer, condition_answer, start,end
    query = "SELECT user_id, query, treatment,problem, start, end FROM serp.exp_data"
    dbcursor.execute(query)
    results = dbcursor.fetchall()
    black_list = []
    for x in results:
        user = x[0]
        query = x[1]
        treatment_answer = x[2]
        condition_answer = x[3]
        start = x[4]
        end = x[5]
        if not start or not end:
             continue
        time = get_time_spent(start, end)

        if not end or not start:
            continue

        is_test_user = False
        for test_user in test_users_names_prefix:
            if user.lower().startswith(test_user):
                is_test_user = True
                break

        if is_test_user:
            continue

        bad_work_msg = unsatisfactory(query, treatment_answer, condition_answer, time, False)
        if bad_work_msg:
            black_list.append(user)
    api = AMT_api()

    for u in black_list:
        api.assign_serp_qualification(u,BLACK_LIST_QUAL, 1)




if __name__ == "__main__":
    set_blacklist_qual_for_all('local')
    #api = AMT_api()
    #api.add_assignments_to_hit('3JHB4BPSFKAWLVAMJB3BBPIM1RUQ9T', 7, ' 3JHB4BPSFKAWLVAMJB3BBPIM1RUQ9T3')


    #assign_quals()
#api = AMT_api()
#api.get_hit_results('3BPP3MA3TCL2PULQZHB1MHK3IDPELU')
#print(api.is_hit_completed('3BPP3MA3TCL2PULQZHB1MHK3IDPELU'))
#pay_bonuses(client, 'C:\\Users\\User\\PycharmProjects\\user_study\\resources\\reports\\Transactions_2021-12-25_to_2021-12-26.csv',
#            'C:\\Users\\User\\PycharmProjects\\user_study\\resources\\reports\\user_study_payment_report.csv')

