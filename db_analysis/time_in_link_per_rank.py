import csv

from utils import connect_to_db, string_to_datetime, get_time_spent


def get_ids_from_csv(filename):
    ids = []
    with open(filename,  newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            ids.append(row['WorkerId'])
    return ids


def get_time_in_link_per_rank(filename):
    ids = get_ids_from_csv(filename)
    db = connect_to_db(test=False)
    dbcursor = db.cursor()
    query = "SELECT user_id, link_id, date FROM serp.user_action"
    dbcursor.execute(query)
    results = dbcursor.fetchall()
    links = {}
    times = {x: 0 for x in range(1, 11)}
    for l in results:
        user_id = l[0]
        link_id = l[1]
        t_press = l[2]
        if user_id not in ids:
            continue
        if user_id not in links:
            links[user_id] = []

        links[user_id].append({'link_id':link_id,'date':string_to_datetime(t_press)})
    link_times = {}
    for user, l in links.items():
        links_by_date = sorted(l, key = lambda i: i['date'])
        if user not in link_times:
            link_times[user] = {x: 0 for x in range(0, 11)}
        for i in range (0, len(links_by_date)-1):
            rank = links_by_date[i]['link_id']
            date = links_by_date[i]['date']
            next_action = links_by_date[i+1]['date']
            time_diff = get_time_spent(date,next_action, False)
            link_times[user][rank] += time_diff
    return link_times


if __name__ == "__main__":
    #get_data_for_query('Does Omega Fatty Acids treat Adhd')
    get_time_in_link_per_rank('../resources/output//user_behaviour.csv')





