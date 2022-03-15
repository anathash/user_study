import csv
from datetime import datetime

from utils import string_to_datetime, connect_to_db


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


def double_single_db(dn_name):
    db = connect_to_db(dn_name)
    cursor = db.cursor()
    q = "SELECT  user_id, query FROM serp.exp_data"
    users = {}
    cursor.execute(q)
    my_results = cursor.fetchall()
    for r in my_results:
        user_id = r[0]
        query = r[1]
        if user_id not in users:
            users[user_id] = set()
        else:
            if query in users[user_id]:
                print('duplication for user_id ' + user_id)
        users[user_id].add(query)
    print('done')


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

double_single_db('biu')

def get_queries_dict(db_name):
    db = connect_to_db(db_name)
    cursor = db.cursor()
    q = "SELECT  user_id, query FROM serp.exp_data"
    users = {}
    cursor.execute(q)
    my_results = cursor.fetchall()
    for r in my_results:
        user_id = r[0]
        query = r[1]
        if user_id not in users:
            users[user_id] = set()
        else:
            if query in users[user_id]:
                print('duplication for user_id ' + user_id)
        users[user_id].add(query)
    return  users


def get_double_users():
    tamar =get_queries_dict('tamar')
    anat =get_queries_dict('biu')
    users = set(anat.keys())
    users_t = set(tamar.keys())
    users.update(users_t)
    duplications = {}
    for u in users:
        if u not in anat or u not in tamar:
            continue
        for q in anat[u]:
            if q in tamar[u]:
                print('duplication for user ' + u)
                if u not in duplications:
                    duplications[u] = []
                duplications[u].append(q)
    return duplications


def get_exp_entries(user, q, db_name):
    db = connect_to_db(db_name)
    cursor = db.cursor()
    q = "SELECT  exp_id, start FROM serp.exp_data where user_id='"+user+"' and query='"+q+"'"
    cursor.execute(q)
    my_results = cursor.fetchall()
    assert(len(my_results) == 1)
    return my_results[0]


def process_double_users():
    duplications = get_double_users()
    tamar = get_queries_dict('tamar')
    anat = get_queries_dict('biu')
    users = set(anat.keys())
    users_t = set(tamar.keys())
    users.update(users_t)
    anat_remove = []
    anat = []
    for u,qs in duplications.items():
        for q in qs:
            r_anat = get_exp_entries(u, q, 'biu')
            date_anat = r_anat[1]
            r_tamar = get_exp_entries(u, q, 'tamar')
            date_tamar = string_to_datetime(r_tamar[1])

            tamar = []
            if date_anat > date_tamar :
                anat.append(r_anat)
                anat_remove.append(r_anat[0])
                print('remove exp ' + r_anat[0] + ' at date ' + date_anat.strftime('%m/%d/%Y, %I:%M:%S %p') +' from Anat')
            else:
                tamar.append(r_tamar)
                print('remove exp ' + r_tamar[0] + ' at date ' + r_tamar[1] + ' from Tamar')

    return anat



process_double_users()
#compare_files('C:\\Users\\User\\Documents\\dumps\\user_config_dump_tamar_2612.csv',
#              'C:\\Users\\User\\Documents\\dumps\\user_config_dump_2612.csv')

#process_double_users('C:\\Users\\User\\Documents\\dumps\\double_users_exp_data_tamar.csv',
#              'C:\\Users\\User\\Documents\\dumps\\double_users_exp_data_anat.csv')