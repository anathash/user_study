import mysql

from db_analysis.utils import connect_to_db, get_links_entered_in_exps, get_time_diff


def update_config_data_table():
    db = mysql.connector.connect(
        host="experiments2.biu-ai.com",
        user="serp_user",
        passwd="YpkYNyMA4qWtJA",
        database="serp",
        port=15136,
        auth_plugin='mysql_native_password'
    )  # database connection



if __name__ == "__main__":
    update_config_data_table()

