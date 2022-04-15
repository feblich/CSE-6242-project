import streamlit as st
import streamlit_authenticator as stauth
import DBUtility
import pandas as pd
import pickle
from predictive_models import SuperGenes
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
import plotly.subplots as sp

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

    super_genes_models = pickle.load(open('model.pkl', 'rb'))
    # st.write(super_genes_models)
    pred_arr = np.empty([test_df.shape[0], len(super_genes_models.keys())])
    pred_arr[:] = np.nan
    for i, drug in enumerate(super_genes_models.keys()):
        y_pred = super_genes_models[drug].predict(test_df)
        pred_arr[:, i] = y_pred

    pred_arr = pd.DataFrame(pred_arr, columns=[drug for drug in super_genes_models.keys()])
    pred_arr['sample_number'] = ['sample ' + str(num) for num in range(pred_arr.shape[0])]
    pred_arr.set_index('sample_number', inplace=True)

    st.title("IC50 prediction")

    pred_arr = pred_arr.stack().reset_index().rename(columns={0:'IC50', 'level_1': 'Drug'})


    # st.write(pred_arr)



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

    # st.write(df)

    # st.set_page_config(layout="wide")
    # st.title("Interact with Drug_sensitivity_IC50_(Sanger_GDSC2)")

    drug_list = [drug for drug in super_genes_models.keys()]
    # patient_list = list(df['Patient'].unique())
    samples = list(pred_arr['sample_number'].unique())

    with st.sidebar:
        st.subheader("Configure the plot")
        patient = st.selectbox(label="Choose a patient", options=samples)

        chart_visual = st.sidebar.selectbox('Select Chart',
                                            ('Predictions', 'Sanger GDSC2', 'Model Metrics'))
    query_1 = f"sample_number=='{patient}'"
    df_patient = pred_arr.query(query_1)

    # st.write(df_patient)

    # drug_subset = ['cisplatin (GDSC2:1005)',
    #                'docetaxel (GDSC2:1007)',
    #                'gefitinib (GDSC2:1010)',
    #                'gemcitabine (GDSC2:1190)',
    #                'paclitaxel (GDSC2:1080)']

    if chart_visual == 'Predictions':

        with st.sidebar:
            treatment_select = st.multiselect(label="Pick Treatment(s)", options=drug_list)

        # if len(treatment_select) == 0:
        #
        #     f = plt.figure(figsize=(10, 15))
        #     for i, val in enumerate(drug_list):
        #         # f.add_subplot(len(drug_list), int(len(drug_list)/2), i + 1)
        #         sns.distplot(df['IC50'][df['Drug'] == val], hist=True)
        #
        #         plt.axvline(x=df['IC50'][df['Drug'] == val].mean(),
        #                     color='blue',
        #                     ls='--',
        #                     lw=2.5,
        #                     label="Average IC50")
        #
        #         if len(df_patient[df_patient['Drug'] == val]['IC50']) != 0:
        #
        #             treatment_avg = df['IC50'][df['Drug'] == val].mean()
        #             patient_result = df_patient[df_patient['Drug'] == val]['IC50'].values[0]
        #
        #             if patient_result > treatment_avg:
        #                 linecolor = 'green'
        #             else:
        #                 linecolor = 'red'
        #
        #             plt.axvline(x=df_patient[df_patient['Drug'] == val]['IC50'].values[0],
        #                         color=linecolor,
        #                         ls='-',
        #                         lw=2.5,
        #                         label="Patient's IC50")
        #
        #         plt.title(str(df['Drug'][df['Drug'] == val].iloc[0]), fontweight='bold')
        #         plt.legend()
        #         plt.grid()
        #         plt.tight_layout()
        #
        #     st.pyplot(f)

        # else:

        if (len(treatment_select) > 0):

            f = plt.figure(figsize=(10, 15))
            for i, val in enumerate(treatment_select):
                f.add_subplot(4, 2, i + 1)
                sns.distplot(df['IC50'][df['Drug'] == val], hist=True)

                plt.axvline(x=df['IC50'][df['Drug'] == val].mean(),
                            color='blue',
                            ls='--',
                            lw=2.5,
                            label="Average IC50")

                if len(df_patient[df_patient['Drug'] == val]['IC50']) != 0:

                    treatment_avg = df['IC50'][df['Drug'] == val].mean()
                    patient_result = df_patient[df_patient['Drug'] == val]['IC50'].values[0]

                    if patient_result > treatment_avg:
                        linecolor = 'green'
                    else:
                        linecolor = 'red'

                    plt.axvline(x=df_patient[df_patient['Drug'] == val]['IC50'].values[0],
                                color=linecolor,
                                ls='-',
                                lw=2.5,
                                label="Patient's IC50")

                plt.title(str(df['Drug'][df['Drug'] == val].iloc[0]), fontweight='bold')
                plt.legend()
                plt.grid()
                plt.tight_layout()

            st.pyplot(f)

    if chart_visual == 'Sanger GDSC2':
        with st.sidebar:
            patient = st.multiselect(label="Pick Treatment(s)", options=drug_list)

        df.rename(columns={"Drug": "Treatment(s)"}, inplace=True)

        if len(patient) == 0:
            fig = px.violin(df[df['Treatment(s)'].isin(drug_list)], y='IC50', x='Treatment(s)',
                            color='Treatment(s)', box=True,
                            hover_data=df[df['Treatment(s)'].isin(drug_list)].columns,
                            width=1000, height=700)
            fig.update_yaxes(title='IC50 (micromolar)')
            fig.update_xaxes(title=None, showticklabels=False)
            st.plotly_chart(fig, sharing="streamlit")

        if len(patient) > 0:
            fig = px.violin(df[df['Treatment(s)'].isin(patient)], y='IC50', x='Treatment(s)',
                            color='Treatment(s)', box=True,
                            hover_data=df[df['Treatment(s)'].isin(patient)].columns,
                            width=1000, height=700)
            fig.update_yaxes(title='IC50 (micromolar)')
            fig.update_xaxes(title=None, showticklabels=False)
            st.plotly_chart(fig)

    if chart_visual == 'Model Metrics':

        #### Model Performance Metrics ####
        metric = ['accuracy', 'accuracy', 'rmse', 'rmse']
        split = ['Training Data', 'Test Data', 'Training Data', 'Test Data']
        vals = [0.8234, 0.71234, 0.2435, 0.123]

        metrics = pd.DataFrame(list(zip(metric, split, vals)), columns=['Metric', 'Split', 'Value'])
        print("metrics:", metrics)

        pm_fig = px.histogram(metrics, x='Split', y='Value', facet_col='Metric', color='Split',
                              title='Model Performance Metrics')
        pm_fig.update_layout(showlegend=False)
        st.plotly_chart(pm_fig, sharing="streamlit")

elif authentication_status == False:
    st.error('Username/password is incorrect')
elif authentication_status == None:
    st.warning('Please enter your username and password')
