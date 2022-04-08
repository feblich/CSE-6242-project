import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

class SuperGenesLasso:
    def __init__(self):
        pass


if __name__ == "__main__":



    ## read in gene expression data
    gene_exp = pd.read_csv("data\Expression_22Q1_Public.csv")
    gene_exp.dropna(inplace=True)

    ## read in drug IC50 data
    drug_IC50 = pd.read_csv("data\Drug_sensitivity_IC50_Sanger_GDSC1.csv")
    mitomycin = drug_IC50[["Unnamed: 0", "mitomycin-C (GDSC1:136)"]]
    mitomycin.dropna(inplace=True)
    trametinib = drug_IC50[["Unnamed: 0", "trametinib (GDSC1:1372)"]]
    trametinib.dropna(inplace=True)

    ## merge the gene epression and IC50 for drugs
    data_mitomycin = gene_exp.merge(mitomycin)
    data_trametinib = gene_exp.merge(trametinib)

    # X = data_mitomycin.loc[:, data_mitomycin.columns != 'mitomycin-C (GDSC1:136)']  # features
    # X = X.loc[:, X.columns != 'Unnamed: 0']  # features
    # y = data_mitomycin['mitomycin-C (GDSC1:136)']  # response Variable

    X = data_trametinib.loc[:, data_trametinib.columns != 'trametinib (GDSC1:1372)']  # features
    X = X.loc[:, X.columns != 'Unnamed: 0']  # features
    y = data_trametinib['trametinib (GDSC1:1372)']  # response Variable

    model = AgglomerativeClustering(n_clusters=500, compute_distances=True)
    X = X.T
    model = model.fit(X)
    X['cluster_labels'] = model.labels_
    X = X.groupby(by="cluster_labels").mean()


    ## try lasso first
    X = X.T
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=42)
    parameters = {'alpha': [.0000001, .0001, .1]}
    lasso = GridSearchCV(Lasso(fit_intercept=True), parameters)
    lasso.fit(X_train, Y_train)
    # plt.plot(ridge.best_estimator_.predict(X_test), color='k', marker='o')
    # plt.plot(Y_test.reset_index(drop=True), 'ro')
    # plt.show()

    ## ridge
    ridge = GridSearchCV(Ridge(fit_intercept=True), parameters)
    ridge.fit(X_train, Y_train)


    ## neural net regressor
    mlp = GridSearchCV(MLPRegressor(), parameters)
    mlp.fit(X_train, Y_train)


    a=2




