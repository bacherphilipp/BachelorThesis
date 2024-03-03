import psycopg2
from configparser import ConfigParser

def get_db_info(section):
        parser = ConfigParser()
        try:
            parser.read('src/config/database.ini')
        except Exception as e:
            print("Error while reading database.ini:", e)
            return {}
        db_info = {}
        if parser.has_section(section):
            key_val_tuple = parser.items(section) 
            for item in key_val_tuple:
             db_info[item[0]]=item[1]
        return db_info

def get_connection():
        conn_info = get_db_info('beymsdb')
        try:
            conn = psycopg2.connect(**conn_info)
            print("Connected to db!")
            return conn.cursor()
        except psycopg2.OperationalError:
            print("Error connecting to the database :/")
            exit()

