import streamlit as st
import streamlit_authenticator as stauth
import DBUtility
import pandas as pd
import pickle
from predictive_models import SuperGenes
import matplotlib.pyplot as plt
import seaborn as sns

connection = DBUtility.init_connection()
user_credentials = DBUtility.run_query(connection,query="SELECT * from user_credentials;")

id = [user[0] for user in user_credentials]
usernames = [user[1] for user in user_credentials]
passwords = [user[2] for user in user_credentials]

hashed_passwords = stauth.Hasher(passwords).generate()

authenticator = stauth.Authenticate(id, usernames, hashed_passwords,
                                    'some_cookie_name', 'some_signature_key', cookie_expiry_days=1)

id, authentication_status, username = authenticator.login('Login', 'main')

authentication_status = True

if authentication_status:
    st.sidebar.subheader("File Upload")
    uploaded_file = st.sidebar.file_uploader(label="upload gene expression data CSV file",
                                             type=['csv'])

    ##TODO upload cell line data for patient to DB
    test_df = pd.read_csv(uploaded_file)
    super_genes_model = pickle.load(open('model.pkl', 'rb'))
    y_pred = super_genes_model.predict(test_df)



    #TODO add all treatments

    # st.title("IC50 prediction for mitomycin")
    # st.write(pd.DataFrame(y_pred).rename(columns={0: "mitomycin"}))
    #TODO store them in the database (PATIENT ID, TREATMENT, IC50 VALUE)

    #TODO visualization --- select db
    drug_rslts = pd.read_csv("data/Drug_sensitivity_IC50_(Sanger_GDSC1).csv")
    drug_rslts = drug_rslts.fillna(0)
    drug_rslts.set_index("Unnamed: 0", inplace=True)
    df = drug_rslts.stack().reset_index().rename(columns={0: 'value'})
    df.rename(columns={'Unnamed: 0': 'Patient', 'level_1': 'Drug',
                       'value': 'IC50'}, inplace=True)
    df = df.loc[(df != 0).all(axis=1)]

    # st.set_page_config(layout="wide")
    st.title("Interact with Drug_sensitivity_IC50_(Sanger_GDSC2)")

    drug_list = ['mitomycin-C (GDSC1:136)']

    with st.sidebar:
        st.subheader("Configure the plot")
        drug = st.selectbox(label="Choose a drug", options=drug_list)

    query_1 = f"Drug=='{drug}'"
    df_filtered_1 = df.query(query_1)


    sns.set(rc={'figure.figsize': (11.7, 8.27)})
    sns.displot(df_filtered_1, x="IC50", kind="kde", bw_adjust=2, fill=True)
    plt.axvline(x=df_filtered_1.IC50.mean(),
                color='blue',
                ls='--',
                lw=2.5)
    plt.axvline(x=y_pred[id],
                color='red',
                ls='-',
                lw=2.5)

    st.pyplot(plt.gcf())

elif authentication_status == False:
    st.error('Username/password is incorrect')
elif authentication_status == None:
    st.warning('Please enter your username and password')
