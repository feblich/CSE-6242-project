import streamlit as st
import pandas as pd
import pickle
import numpy as np
from predictive_models import SuperGenes

st.sidebar.subheader("File Upload")
uploaded_file = st.sidebar.file_uploader(label="upload gene expression data CSV file",
                         type=['csv'])

if uploaded_file is not None:
    try:
        test_df = pd.read_csv(uploaded_file)

    except Exception as e:
        print(e)

super_genes_models = pickle.load(open('model.pkl', 'rb'))
pred_arr = np.empty([test_df.shape[0], len(super_genes_models.keys())])
pred_arr[:] = np.nan
for i, drug in enumerate(super_genes_models.keys()):
    y_pred = super_genes_models[drug].predict(test_df)
    pred_arr[:, i] = y_pred

pred_arr = pd.DataFrame(pred_arr, columns=[drug for drug in super_genes_models.keys()])
pred_arr['sample_number'] = ['sample ' + str(num) for num in range(pred_arr.shape[0])]
pred_arr.set_index('sample_number', inplace=True)

st.title("IC50 prediction")
st.write(pred_arr)