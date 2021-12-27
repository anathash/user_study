import csv

from utils import string_to_datetime


def compare_files(tamar, anat):
    tamar_users = set()
    anat_users = set()
    with open(tamar, newline='', encoding='utf8') as tamarfile:
        reader = csv.DictReader(tamarfile)
        # get survey codes from db
        for row in reader:
            amazon_id = row['amazon_id']
            tamar_users.add(amazon_id)

    with open(anat, newline='', encoding='utf8') as anatfile:
        reader = csv.DictReader(anatfile)
        # get survey codes from db
        for row in reader:
            amazon_id = row['amazon_id']
            anat_users.add(amazon_id)

    inter = anat_users.intersection(tamar_users)
    if not inter:
        print("intersection empty")
    else:
        print(inter)

def process_double_users(tamar, anat):
    tamar_users = {}
    anat_users = {}
    with open(tamar, newline='', encoding='utf8') as tamarfile:
        reader = csv.DictReader(tamarfile)
        # get survey codes from db
        for row in reader:
            amazon_id = row['user_id']
            tamar_users[amazon_id] = row
    with open(anat, newline='', encoding='utf8') as anatfile:
        reader = csv.DictReader(anatfile)
        # get survey codes from db
        for row in reader:
            amazon_id = row['user_id']
            anat_users[amazon_id] = row

    for amazon_id, exp_data in anat_users.items():
        if amazon_id not in tamar_users:
            print(amazon_id +' not in Tamars users')
            continue

        tamar_query = tamar_users[amazon_id]['query']
        anat_query = anat_users[amazon_id]['query']
        if tamar_query == anat_query:
            tamar_datesting = tamar_users[amazon_id]['start']
            anat_datesting = anat_users[amazon_id]['start']
            tamar_date = string_to_datetime(tamar_datesting)
            anat_date = string_to_datetime(anat_datesting)
            print('duplicate user :' + amazon_id + ' Tamar: ' + tamar_datesting + ' Anat: ' + anat_datesting)
            if tamar_date < anat_date:
                print('Anat should remove')
            elif anat_date < tamar_date:
                print('Tamar should remove')
            else:
                print('Both should remove')



#compare_files('C:\\Users\\User\\Documents\\dumps\\user_config_dump_tamar_2612.csv',
#              'C:\\Users\\User\\Documents\\dumps\\user_config_dump_2612.csv')

process_double_users('C:\\Users\\User\\Documents\\dumps\\double_users_exp_data_tamar.csv',
              'C:\\Users\\User\\Documents\\dumps\\double_users_exp_data_anat.csv')