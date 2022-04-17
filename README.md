# CSE-6242-project
###Cancer Treatment Predictor

This package contains our Cancer Treatment Predictor [Streamlit](https://streamlit.io/) app. This app deploys our own 
predictive models which calculates the reaction of a gene sample to a cancer treatment (implementation in `predictive_model.py`). 
This model is built from gene expression data found in `Expression 22Q1 Public.csv` and `Drug_sensitivity_IC50_Sanger_GDSC1.csv` from
[DepMap](https://depmap.org/portal/download/custom/). The model is then stored in a pickle file (`model.pkl`) and consumed
by the application. Models were built
for the following cancer drugs: 
- LDN-193189 (GDSC1:478)
- Fingolimod hydrochloride; Gilenya; TDI-132; Imusera; Gilenia (GDSC1:546)
- CD532 (GDSC1:449)
- PF-00299804 (GDSC1:363)
- Anchusin (GDSC1:170)
- tenovin-6 (GDSC1:342)
- dacinostat (GDSC1:200)
- panobinostat (GDSC1:438)
- AST-1306 (GDSC1:381)
- Trichostatin A (GDSC1:437)
- vorinostat (GDSC1:1012)
- foretinib (GDSC1:308)
- PDK1 inhibitor AR-12 (GDSC1:167)
- Dimethyloxalylglcine (GDSC1:165)
- CAY10603 (GDSC1:276)
- mitomycin-C (GDSC1:136)
- KIN001-204 (GDSC1:157)
- Piplartine (GDSC1:1243)
- IMD-0354 (GDSC1:442)
- CPI-613 (GDSC1:415)

This app provides a portal for users to upload gene samples for cancer patients and then see the predicted IC50 reaction 
value for the 20 cancer treatments. Along with drug reaction predictions, you can view model metrics in order
to more intuitively understand the results. 

This app is remotely deployed through Heroku and saves data in a Postgres database. You can access our deployed app
at https://cse-6242.herokuapp.com/.


###Local Development

####Connect to database
In order to deploy the app locally there should be an `.env` file at the root of the directory 
containing the following. This will allow you to connect to the Heroku Postgres database 

```.text
DATABASE_URL=<redacted>
```

Note: The `local_development` variable in `DBUtility.py` needs to be
`True` for local development (line 7). 


####Install all required packages
Install all required python packages specified in the `requirement.txt` file. Packages can be installed by using `pip`. 
- [How To Install PIP to Manage Python Packages On Windows](https://phoenixnap.com/kb/install-pip-windows)
- [How to Install Pip on Mac](https://phoenixnap.com/kb/install-pip-mac)

Once pip is working on your device, you can run the following command in your terminal: 
```text
pip install -r requirements.txt
```

####Deploy the app!
You are now ready to deploy the streamlit app using your terminal: 
```text
streamlit run app.py 
```

####Use App
Once the login page is visible you can log into the app using the following credentials: 
```text
username: local_user
password: cse-6242
```

You can now upload gene samples using `x_test.csv` or view existing samples. 


