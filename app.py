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
import datetime as dt


def generate_metrics_df(object, round_n):
    '''This function takes a default dict of SuperGenes models (labelled by corresponding drug name) and iterates through each dictionary key,
    extracting relevant model training parameters and metrics of the best-performing model. A pandas DataFrame containing these metrics is returned.
    '''
    n = round_n  # shorthand
    results = np.array([])
    for drug in object:
        model = object[drug]
        best_idx = model.mdl.best_index_
        cv_results = model.mdl.cv_results_
        results = np.append(results, [drug, model.mdl.estimator, model.mdl.n_splits_, cv_results['param_alpha'][best_idx],\
                            np.round(cv_results['mean_test_score'][best_idx], n), np.round(cv_results['std_test_score'][best_idx], n),\
                            np.round(cv_results['split0_test_score'][best_idx], n), np.round(cv_results['split1_test_score'][best_idx], n),\
                            np.round(cv_results['split2_test_score'][best_idx], n), np.round(cv_results['split3_test_score'][best_idx], n),\
                            np.round(cv_results['split4_test_score'][best_idx], n)])
    results = np.reshape(results, (len(object), -1))  # reshape array
    results = pd.DataFrame(results, columns= ['drug', 'estimator', 'n_splits', 'best alpha', 'mean_test_score', 'std_test_score',\
                                              'test split 0', 'test split 1', 'test split 2', 'test split 3', 'test split 4'])
    return results

col1, col2 = st.columns([11,2])

with col1:
    st.title('Cancer Treatment Predictor')

with col2:
    st.image("title_image.png", use_column_width='always')



connection = DBUtility.init_connection()

user_credentials = DBUtility.run_query(connection,query="SELECT * from user_credentials;")

id = [user[0] for user in user_credentials]
usernames = [user[1] for user in user_credentials]
passwords = [user[2] for user in user_credentials]

hashed_passwords = stauth.Hasher(passwords).generate()

authenticator = stauth.Authenticate(id, usernames, hashed_passwords,
                                    'some_cookie_name', 'some_signature_key', cookie_expiry_days=1)

id, authentication_status, username = authenticator.login('Login', 'main')

initial_run = True

if authentication_status:
    st.sidebar.subheader("File Upload")
    uploaded_file = st.sidebar.file_uploader(label="upload gene expression data CSV file",type=['csv'])
    super_genes_models = pickle.load(open('model.pkl', 'rb'))

    ##TODO upload cell line data for patient to DB

    if uploaded_file is not None:
        test_df = pd.read_csv(uploaded_file)
        pred_arr = np.empty([test_df.shape[0], len(super_genes_models.keys())])
        pred_arr[:] = np.nan
        for i, drug in enumerate(super_genes_models.keys()):
            y_pred = super_genes_models[drug].predict(test_df)
            pred_arr[:, i] = y_pred

        pred_arr = pd.DataFrame(pred_arr, columns=[drug for drug in super_genes_models.keys()])
        pred_arr['sample_id'] = pred_arr.sum(axis=1).map(hash)
        pred_arr.set_index('sample_id', inplace=True)

        pred_arr = pred_arr.stack().reset_index().rename(columns={0: 'IC50', 'level_1': 'Drug'})
        pred_arr['user_id'] = [id for num in range(pred_arr.shape[0])]
        DBUtility.execute_values(connection, pred_arr, 'results')

        uploaded_file = None


    results = DBUtility.run_query(connection, query=f"SELECT * from results where user_id = {id};")
    result_df = pd.DataFrame(results, columns=['sample_id', 'user_id', 'drug', 'ic50'])

    drug_rslts = pd.read_csv("data/Drug_sensitivity_IC50_(Sanger_GDSC1).csv")
    drug_rslts = drug_rslts.fillna(0)
    drug_rslts.set_index("Unnamed: 0", inplace=True)
    df = drug_rslts.stack().reset_index().rename(columns={0: 'value'})
    df.rename(columns={'Unnamed: 0': 'Patient', 'level_1': 'drug',
                       'value': 'ic50'}, inplace=True)
    df = df.loc[(df != 0).all(axis=1)]

    # st.write(df)

    # st.set_page_config(layout="wide")
    # st.title("Interact with Drug_sensitivity_IC50_(Sanger_GDSC2)")

    drug_list = list(result_df['drug'].unique())
    # patient_list = list(df['Patient'].unique())
    samples = list(result_df['sample_id'].unique())

    with st.sidebar:
        st.subheader("Configure the plot")
        patient = st.selectbox(label="Choose a patient", options=samples)

        chart_visual = st.sidebar.selectbox('Select View', options=['Edit Patient Data','Predictions', 'Sanger GDSC1', 'Model Metrics'])



    query_1 = f"sample_id=='{patient}'"
    df_patient = result_df.query(query_1)

    if chart_visual == 'Predictions':
        st.header("IC50 prediction")
        with st.sidebar:
            treatment_select = st.multiselect(label="Pick Treatment(s)", options=drug_list)

        if (len(treatment_select) > 0):

            f = plt.figure(figsize=(10, 15))
            for i, val in enumerate(treatment_select):
                f.add_subplot(4, 2, i + 1)
                sns.distplot(df['ic50'][df['drug'] == val], hist=True)

                plt.axvline(x=df['ic50'][df['drug'] == val].mean(),
                            color='blue',
                            ls='--',
                            lw=2.5,
                            label="Average IC50")

                if len(df_patient[df_patient['drug'] == val]['ic50']) != 0:

                    treatment_avg = df['ic50'][df['drug'] == val].mean()
                    patient_result = df_patient[df_patient['drug'] == val]['ic50'].values[0]

                    if patient_result > treatment_avg:
                        linecolor = 'green'
                    else:
                        linecolor = 'red'

                    plt.axvline(x=df_patient[df_patient['drug'] == val]['ic50'].values[0],
                                color=linecolor,
                                ls='-',
                                lw=2.5,
                                label="Patient's IC50")

                plt.title(str(df['drug'][df['drug'] == val].iloc[0]), fontweight='bold')
                plt.legend()
                plt.grid()
                plt.tight_layout()

            st.pyplot(f)

    if chart_visual == 'Sanger GDSC1':
        with st.sidebar:
            patient = st.multiselect(label="Pick Treatment(s)", options=drug_list)

        df.rename(columns={"drug": "Treatment(s)"}, inplace=True)

        if len(patient) == 0:
            fig = px.violin(df[df['Treatment(s)'].isin(drug_list)], y='ic50', x='Treatment(s)',
                            color='Treatment(s)', box=True,
                            hover_data=df[df['Treatment(s)'].isin(drug_list)].columns,
                            width=1000, height=700)
            fig.update_yaxes(title='IC50 (micromolar)')
            fig.update_xaxes(title=None, showticklabels=False)
            st.plotly_chart(fig, sharing="streamlit")

        if len(patient) > 0:
            fig = px.violin(df[df['Treatment(s)'].isin(patient)], y='ic50', x='Treatment(s)',
                            color='Treatment(s)', box=True,
                            hover_data=df[df['Treatment(s)'].isin(patient)].columns,
                            width=1000, height=700)
            fig.update_yaxes(title='IC50 (micromolar)')
            fig.update_xaxes(title=None, showticklabels=False)
            st.plotly_chart(fig)

    if chart_visual == 'Model Metrics':
        metrics = generate_metrics_df(super_genes_models, round_n=2)

        # Convert sign of std test score to match that of mean test score (for visual aesthetics)
        metrics['std_test_score'] = metrics['std_test_score'] * np.sign(metrics['mean_test_score'])

        # add drug filter
        model_names = sorted(metrics['drug'].unique())

        # drug, variable, value
        metrics = pd.melt(metrics, id_vars='drug')
        metrics['value'] = metrics['value'].astype(str)

        filter_drug = st.sidebar.selectbox('Select drug model:', options=model_names)
        metrics = metrics[metrics['drug'] == filter_drug]

        st.subheader(f"{filter_drug} model")

        hide_dataframe_row_index = """
                    <style>
                    .row_heading.level0 {display:none}
                    .blank {display:none}
                    </style>
                    """

        st.markdown(hide_dataframe_row_index, unsafe_allow_html=True)
        metrics_table = st.table(data=metrics.loc[:, ['variable', 'value']])

    if chart_visual == 'Edit Patient Data':
        # patient_results = DBUtility.run_query(DBUtility.init_connection(), query=f"SELECT * from patient_data where sample_id = \'{patient}\' order by date_entered desc limit 1;")
        # st.write(patient_results)
        patient_data = {}
        st.header("Patient Demographics")

        # add patient slider
        census_races = ['Other', 'White', 'Black or African American', 'American Indian or Alaska Native', 'Asian',
                        'Native Hawaiian or Other Pacific Islander', 'Hispanic/Latino']

        sexs = ['N/A', 'Female', 'Male', 'Intersex']

        # note: we can and SHOULD use default/auto-generated patient ID as default
        patient_data['sample_'] = patient
        patient_data['age'] = st.number_input("Age at Diagnosis", step=1, max_value=150, value=67)
        patient_data['sex'] = st.selectbox('Sex', sexs)
        patient_data['race'] = st.selectbox("Race or Ethnicity", options=census_races)
        patient_data['cancer_type'] = st.selectbox("Cancer Type", options=['Carcinoma', 'Sarcoma', 'Melanoma', 'Lymphoma', 'Leukemia'], index=4)
        patient_data['cancer_subtype'] = st.text_input("Cancer Subtype")
        patient_data['stage'] = st.selectbox("Stage:", options=['I', 'II', 'III', 'IV', 'Unstaged'])
        patient_data['prior_treatments'] = st.multiselect("Prior Therapies? List all that apply", options=drug_list)
        patient_data['recommended_treatment'] = st.selectbox("Which treatment will you recommend?", options=drug_list)
        patient_data['frequency'] = st.selectbox('Frequency Unit', options=['Weekly', 'Biweekly', 'Triweekly', 'Monthly'])
        patient_data['treatments_per_week'] = st.number_input(f"Treatments per unit", step=1)
        patient_data['dosage'] = st.number_input("Dosage (micrograms)")
        patient_data['anticipated_duration'] = st.number_input("Anticipated Treatment Duration (in weeks)")

        patient_data['date_entered'] = dt.date.today()


        st.write('Input date:', patient_data['date_entered'].strftime("%B %d, %Y"))

        # user = "test user"  # ideally we would sub in the physician's user id here.

        # print('test df iteration')

        if st.button("Update Patient Data"):
            DBUtility.execute_values(connection, pd.DataFrame.from_dict(patient_data), 'patient_data')



elif authentication_status == False:
    st.error('Username/password is incorrect')
elif authentication_status == None:
    st.warning('Please enter your username and password')
