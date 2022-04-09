import streamlit as st
import streamlit_authenticator as stauth
import DBUtility


connection = DBUtility.init_connection()
user_credentials = DBUtility.run_query(connection,query="SELECT * from user_credentials;")

id = [user[0] for user in user_credentials]
usernames = [user[1] for user in user_credentials]
passwords = [user[2] for user in user_credentials]

hashed_passwords = stauth.Hasher(passwords).generate()

authenticator = stauth.Authenticate(id, usernames, hashed_passwords,
                                    'some_cookie_name', 'some_signature_key', cookie_expiry_days=30)

name, authentication_status, username = authenticator.login('Login', 'main')

if authentication_status:
    st.write('Welcome *%s*' % (name))
    st.title('Some content')
elif authentication_status == False:
    st.error('Username/password is incorrect')
elif authentication_status == None:
    st.warning('Please enter your username and password')
