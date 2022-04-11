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
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import pickle

class SuperGenes:
    def __init__(self, clustering_method, model_config):
        self.clustering_method = clustering_method
        self.model_config = model_config
        self.pipe = None
        self.mdl = None
        self.scaler = None
        self.super_genes_labels = None

    def _cluster(self, X):
        if self.clustering_method == 'hierarchical_clustering':
            X = X.T
            # self.scaler = StandardScaler()
            # scaled_X = self.scaler.fit_transform(X)
            # X = pd.DataFrame(scaled_X, index=X.index, columns=X.columns)
            cluster_obj = AgglomerativeClustering(n_clusters=500, compute_distances=True)
            cluster_obj.fit(X)
            self.super_genes_labels = cluster_obj.labels_
            X['cluster_labels'] = self.super_genes_labels
            X = X.groupby(by="cluster_labels").mean()
            X = X.T
            return X

    def learning(self, X, y):
        X_low_dim = self._cluster(X)
        self.mdl = GridSearchCV(self.model_config['name'], self.model_config['parameters'])
        self.mdl.fit(X_low_dim, y)

    def predict(self, X):
        X = X.T
        # X = pd.DataFrame(self.scaler.transform(X), index=X.index, columns=X.columns)
        X['cluster_labels'] = self.super_genes_labels
        X = X.groupby(by="cluster_labels").mean()
        X = X.T
        return self.mdl.predict(X)

def create_training_set(gene_exp, drug_IC50, drug_name):
    drug_col_name = drug_IC50.columns[drug_IC50.columns.str.contains(drug_name)][0]
    drug_name_df = drug_IC50[["Unnamed: 0", drug_col_name]]
    drug_name_df.dropna(inplace=True)
    gene_exp_IC50 = gene_exp.merge(drug_name_df)
    drug_col_name = gene_exp_IC50.columns[gene_exp_IC50.columns.str.contains(drug_name)][0]
    X = gene_exp_IC50.loc[:, gene_exp_IC50.columns != drug_col_name]  # features
    X = X.loc[:, X.columns != 'Unnamed: 0']  # features
    y = gene_exp_IC50[drug_col_name]
    return X, y
    # drug_IC50_dataset.iloc[:, drug_IC50.columns.str.contains(drug_name)]

if __name__ == "__main__":

    # list of drug to the analysis
    drug_list = ['mitomycin', 'vorinostat', 'elesclomol', 'bortezomib']

    ## read in gene expression data
    gene_exp = pd.read_csv("data\Expression_22Q1_Public.csv")
    gene_exp.dropna(inplace=True)

    ## read in drug IC50 data
    all_drugs_IC50 = pd.read_csv("data\Drug_sensitivity_IC50_Sanger_GDSC1.csv")

    
    X, y = create_training_set(gene_exp, all_drugs_IC50, 'mitomycin')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=42)

    mdl = SuperGenes(clustering_method='hierarchical_clustering',
                     model_config={'name': Lasso(), 'parameters': {'alpha': [.0000001, .0001]}})

    mdl.learning(X_train, y_train)
    y_pred = mdl.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred)
    pickle.dump(mdl, open('model.pkl', 'wb'))
    X_test.to_csv('x_test.csv', index=False)
    y_test.to_csv('y_test.csv', index=False)






