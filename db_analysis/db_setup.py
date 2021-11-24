import mysql.connector

def connect_to_db():
    db = mysql.connector.connect(
        host="hcdm3.cs.virginia.edu",
        user="zw3hk",
        passwd="Fall2021!!",
        database="serp"
    )  # database connection
    return db

