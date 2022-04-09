import streamlit as st
import streamlit_authenticator as stauth
import psycopg2
from urllib.parse import urlparse
import os

def init_connection():
    result = urlparse(os.environ.get('DATABASE_URL'))
    return psycopg2.connect(
        database = result.path[1:],
        user = result.username,
        password = result.password,
        host = result.hostname,
        port = result.port
    )

def run_query(query):
    with conn.cursor() as cur:
        cur.execute(query)
        return cur.fetchall()

conn = init_connection()

user_credentials = run_query("SELECT * from user_credentials;")


id = [user[0] for user in user_credentials]
usernames = [user[1] for user in user_credentials]
passwords = [user[2] for user in user_credentials]

hashed_passwords = stauth.Hasher(passwords).generate()

authenticator = stauth.Authenticate(id,usernames,hashed_passwords,
    'some_cookie_name','some_signature_key',cookie_expiry_days=30)

name, authentication_status, username = authenticator.login('Login','main')

if authentication_status:
    st.write('Welcome *%s*' % (name))
    st.title('Some content')
elif authentication_status == False:
    st.error('Username/password is incorrect')
elif authentication_status == None:
    st.warning('Please enter your username and password')

