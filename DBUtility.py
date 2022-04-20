import psycopg2
from urllib.parse import urlparse
import psycopg2.extras as extras
import os
from dotenv import load_dotenv

local_development = True


def init_connection():
    if local_development:
        load_dotenv()
        DATABASE_URL = urlparse(os.getenv('DATABASE_URL'))
    else:
        DATABASE_URL = urlparse(os.environ.get('DATABASE_URL'))
    return psycopg2.connect(
        database=DATABASE_URL.path[1:],
        user=DATABASE_URL.username,
        password=DATABASE_URL.password,
        host=DATABASE_URL.hostname,
        port=DATABASE_URL.port
    )


def run_query(connection, query):
    with connection.cursor() as cur:
        cur.execute(query)
        return cur.fetchall()


def execute_values(conn, df, table):
    tuples = [tuple(x) for x in df.to_numpy()]

    cols = ','.join(list(df.columns))
    query = "INSERT INTO %s(%s) VALUES %%s" % (table, cols)
    cursor = conn.cursor()
    try:
        extras.execute_values(cursor, query, tuples)
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print("Error: %s" % error)
        conn.rollback()
        cursor.close()
        return 1
    cursor.close()

