from turk_api import AMT_api, is_worker_done
from utils import connect_to_db


def get_user_queries(cursor):
    exp_data_query_string = 'SELECT user_id, query FROM serp.exp_data'
    user_queries = {}
    cursor.execute(exp_data_query_string)
    results = cursor.fetchall()
    for r in results:
        user_id = r[0]
        query = r[1]
        if user_id not in user_queries:
            user_queries[user_id] = set()
        user_queries[user_id].add(query)
    return user_queries


def gen_params(user_queries, add_hyphen = True):
    params = []
    for user_id, queries in user_queries.items():
        q_str = "("
        for q in queries:
            if add_hyphen:
                q_str += "'"+q+"'"
            else:
                q_str += q
            q_str += ','
        q_str = q_str[:-1] + ")"
        params.append([q_str, user_id])
    return params


def add_user_queries_to_user_data_table(db_name):
    db = connect_to_db(db_name)
    mycursor = db.cursor()
#    UPDATE `serp`.`user_data` SET `queries` = '9\')' WHERE (`user_id` = 'test1514');
    user_queries = get_user_queries(mycursor)
    params = gen_params(user_queries)
    q = "UPDATE `serp`.`user_data` SET `queries`=%s WHERE (`user_id`=%s);"
    mycursor.executemany(q,params)
    db.commit()


def import_user_queries(db_name, all_qs = {'Does Omega Fatty Acids treat Adhd', 'Does Ginkgo Biloba treat tinnitus'}):
    to_db = connect_to_db(db_name)
    to_cursor = to_db.cursor()
    from_db = connect_to_db('tamar')
    biu_cursor = from_db.cursor()
    biu_users_q = "SELECT * FROM serp.user_data"
    biu_cursor.execute(biu_users_q)
    biu_results = biu_cursor.fetchall()
    update_user_queries = {}
    insert_user_queries = []

    for br in biu_results:
        user_id = br[0]
        q = "SELECT * FROM serp_shared.user_data where user_id = '" + user_id + "'"
        to_cursor.execute(q)
        my_results = to_cursor.fetchall()
        if my_results:
            for r in my_results:
                my_queries = r[5]
                biu_queries = br[5]
                if biu_queries and my_queries:
                    my_set = set(my_queries[1:-1].split(","))
                    biu_set = set(biu_queries[1:-1].split(","))
                    if not my_set.symmetric_difference(biu_set):
                        continue
                    q_set = my_set.union(biu_set)
                elif biu_queries:
                    q_set = set(biu_queries[1:-1].split(","))
                update_user_queries[user_id] = q_set
        else:
            insert_user_queries.append(br)

    update_params = gen_params(update_user_queries, add_hyphen = False)
    print('update params')
    print(update_params)
    q = "UPDATE `serp_shared`.`user_data` SET `queries`=%s WHERE (`user_id`=%s);"
    to_cursor.executemany(q, update_params)
    to_db.commit()
    api = AMT_api()
    for user_id, queries in update_user_queries.items():
         if is_worker_done(queries, all_qs):
             api.assign_serp_qualification(user_id, 'SERP done', 1)






#INSERT INTO `serp`.`user_data` (`user_id`, `age`, `gender`, `education_level`, `education_field`, `queries`) VALUES ('d', '20', 'male', 'ere', 'ere', 'ere');
    print('insert_user_queries')
    print(insert_user_queries)
    q = "INSERT INTO `serp_shared`.`user_data` (`user_id`, `age`, `gender`,`education_level`, `education_field`, `queries`) VALUES (%s, %s, %s, %s, %s, %s)"
    to_cursor.executemany(q, insert_user_queries)

    to_db.commit()


def find_duplicate_users():
    to_db = connect_to_db('shared')
    to_cursor = to_db.cursor()
    q = "SELECT * FROM serp_shared.user_data"
    users = {}
    to_cursor.execute(q)
    my_results = to_cursor.fetchall()
    for r in my_results:
        user_id = r[0]
        age = r[1]
        gender = r[2]
        eduacation_level = r[3]
        education_field = r[4]
        queries = r[5]
        if user_id not in users:
            users[user_id] = r
        else:
            print(user_id)
            e = users[user_id]
            for i in range(0,6):
                if r[i] != e[i]:
                    print(r)
                    print(e)

if __name__ == "__main__":
    #print()
    import_user_queries('shared')
    find_duplicate_users()
    #add_user_queries_to_user_data_table('local')

