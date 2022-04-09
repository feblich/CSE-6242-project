import streamlit as st
import pandas as pd
import pickle
from predictive_models import SuperGenes

st.sidebar.subheader("File Upload")
uploaded_file = st.sidebar.file_uploader(label="upload gene expression data CSV file",
                         type=['csv'])

if uploaded_file is not None:
    try:
        test_df = pd.read_csv(uploaded_file)

    except Exception as e:
        print(e)

super_genes_model = pickle.load(open('model.pkl', 'rb'))
y_pred = super_genes_model.predict(test_df)
st.title("IC50 prediction for mitomycin")
st.write(pd.DataFrame(y_pred))
