import psycopg2
from decouple import config

class Database:
    def __init__(self):
        self.connection = psycopg2.connect(
            host=config("dbHost"),
            user=config("dbUser"),
            port=config("dbPort"),
            password=config("dbPassword"),
            database=config("dbDatabase")
        )

        self.cur=self.connection.cursor()

    def insert(self,queryString):
        self.cur.execute(queryString)
        self.connection.commit()

    def query(self,queryString):
        self.cur.execute(queryString)
        self.connection.commit()
        return self.cur.fetchall()
