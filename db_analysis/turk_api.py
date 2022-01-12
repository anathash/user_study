import csv

import boto3
import xmltodict

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

    def assign_serp_qualification(self, worker_id):
        response = self.client.associate_qualification_with_worker(
            QualificationTypeId='3VG0EHWASUFRE500CC3BLPY5SRAOQH',
            WorkerId=worker_id,
            IntegerValue=1,
            SendNotification=False
        )
        if self.check_response(response):
            print('Assigned SERP qualification to worker: ' + worker_id)

        else:
            print('assign_serp_qualification failed for worker: ' + worker_id)

    def send_bonus(self, worker_id, assignment_id, unique_token):
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


if __name__ == "__main__":
    api = AMT_api()#19+15+22+7=63
    api.add_assignments_to_hit('3JHB4BPSFKAWLVAMJB3BBPIM1RUQ9T', 7, ' 3JHB4BPSFKAWLVAMJB3BBPIM1RUQ9T3')
    #assign_quals()
#api = AMT_api()
#api.get_hit_results('3BPP3MA3TCL2PULQZHB1MHK3IDPELU')
#print(api.is_hit_completed('3BPP3MA3TCL2PULQZHB1MHK3IDPELU'))
#pay_bonuses(client, 'C:\\Users\\User\\PycharmProjects\\user_study\\resources\\reports\\Transactions_2021-12-25_to_2021-12-26.csv',
#            'C:\\Users\\User\\PycharmProjects\\user_study\\resources\\reports\\user_study_payment_report.csv')

