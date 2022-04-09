import psycopg2
from urllib.parse import urlparse
import os


def init_connection():
    result = urlparse(os.environ.get('DATABASE_URL'))
    return psycopg2.connect(
        database=result.path[1:],
        user=result.username,
        password=result.password,
        host=result.hostname,
        port=result.port
    )


def run_query(connection, query):
    with connection.cursor() as cur:
        cur.execute(query)
        return cur.fetchall()
