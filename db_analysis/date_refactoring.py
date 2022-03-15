
#TODO : filter by knowledge?? YES /NO/ ALL
from time import strftime

from utils import connect_to_db, string_to_datetime


def refactor_exp(db_name):
    db = connect_to_db(db_name)
    dbcursor = db.cursor()
    query = "SELECT exp_id, start, end FROM serp.exp_data"
    dbcursor.execute(query)
    results = dbcursor.fetchall()
    start_update = []
    end_update = []
    for x in results:
        exp_id = x[0]
        if x[1]:
            start_t = string_to_datetime(x[1])
            start_update.append((start_t.strftime('%Y-%m-%d %H:%M:%S'), exp_id))
        if x[2]:
            end_t = string_to_datetime(x[2])
            end_update.append((end_t.strftime('%Y-%m-%d %H:%M:%S'), exp_id))

    q = "UPDATE `serp`.`exp_data` SET `start_t`=%s WHERE (`exp_id`=%s);"
    dbcursor.executemany(q, start_update)
    db.commit()

    q = "UPDATE `serp`.`exp_data` SET `end_t`=%s WHERE (`exp_id`=%s);"
    dbcursor.executemany(q, end_update)
    db.commit()


def refactor_actions(db_name):
    db = connect_to_db(db_name)
    dbcursor = db.cursor()
    query = "SELECT data_id, date FROM serp.user_action"
    dbcursor.execute(query)
    results = dbcursor.fetchall()
    date_update = []
    for x in results:
        data_id = x[0]
        date_t = string_to_datetime(x[1])
        date_update.append((date_t.strftime('%Y-%m-%d %H:%M:%S'), data_id))

    q = "UPDATE `serp`.`user_action` SET `date_dt`=%s WHERE (`data_id`=%s);"
    dbcursor.executemany(q, date_update)
    db.commit()


if __name__ == "__main__":
    refactor_actions('local')



