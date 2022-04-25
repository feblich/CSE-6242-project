# viusalize training performance
import pickle
import numpy as np
import matplotlib.pyplot as plt
from predictive_models import SuperGenes



# def visualize_scores(model_1, model_2, model_name):
#
#     models_dict_1 = pickle.load(open(model_1, 'rb'))
#     x_500_model = np.array(models_dict_1['vorinostat (GDSC1:1012)'].mdl.cv_results_['param_alpha'])
#     y_500_model = models_dict_1['vorinostat (GDSC1:1012)'].mdl.cv_results_['mean_test_score']
#     err = models_dict_1['vorinostat (GDSC1:1012)'].mdl.cv_results_['std_test_score']
#
#     models_dict_2 = pickle.load(open(model_2, 'rb'))
#     x_1000_model = np.array(models_dict_2['vorinostat (GDSC1:1012)'].mdl.cv_results_['param_alpha'])
#     y_1000_model = models_dict_2['vorinostat (GDSC1:1012)'].mdl.cv_results_['mean_test_score']
#     err = models_dict_2['vorinostat (GDSC1:1012)'].mdl.cv_results_['std_test_score']
#
#     plt.errorbar(x_500_model, y_500_model, err, marker='s', mfc='red', linestyle='dashed',
#                  label='Lasso, 500 super genes')
#     plt.errorbar(x_1000_model, y_1000_model, err, marker='s', mfc='green', linestyle='dashed',
#                  label='Lasso, 1000 super genes')
#     plt.xlabel('alpha')
#     plt.ylabel('mean CV test score')
#     plt.legend(loc='lower right')
#     plt.title('{} super genes for Vorinostat'.format(model_name))
#     # plt.show()
#     plt.savefig("{}.png".format(model_name), format="png", dpi=1200)



if __name__ == "__main__":

    lasso_500_model = r'C:\Files\OMSA\Data and Visual Analytics\project\repos\CSE-6242-project\model_lasso_500_super_genes.pkl'
    lasso_1000_model = r'C:\Files\OMSA\Data and Visual Analytics\project\repos\CSE-6242-project\model_lasso_1000_super_gene.pkl'
    ridge_500_model = r'C:\Files\OMSA\Data and Visual Analytics\project\repos\CSE-6242-project\model_ridge_500_super_gene.pkl'
    ridge_1000_model = r'C:\Files\OMSA\Data and Visual Analytics\project\repos\CSE-6242-project\model_ridge_1000_super_gene.pkl'
    mlp_500_model = r'C:\Files\OMSA\Data and Visual Analytics\project\repos\CSE-6242-project\model_mlp_500_super_gene.pkl'
    mlp_1000_model = r'C:\Files\OMSA\Data and Visual Analytics\project\repos\CSE-6242-project\model_mlp_1000_super_gene.pkl'
    # visualize_scores(lasso_500_model, lasso_1000_model, model_name='Multi-layer perceptron')

    models_dict_lasso_500 = pickle.load(open(lasso_500_model, 'rb'))
    x_lasso_500_model = np.array(models_dict_lasso_500['vorinostat (GDSC1:1012)'].mdl.cv_results_['param_alpha'])
    y_lasso_500_model = models_dict_lasso_500['vorinostat (GDSC1:1012)'].mdl.cv_results_['mean_test_score']
    err_lasso_500 = models_dict_lasso_500['vorinostat (GDSC1:1012)'].mdl.cv_results_['std_test_score']

    models_dict_lasso_1000 = pickle.load(open(lasso_1000_model, 'rb'))
    x_lasso_1000_model = np.array(models_dict_lasso_1000['vorinostat (GDSC1:1012)'].mdl.cv_results_['param_alpha'])
    y_lasso_1000_model = models_dict_lasso_1000['vorinostat (GDSC1:1012)'].mdl.cv_results_['mean_test_score']
    err_lasso_1000 = models_dict_lasso_1000['vorinostat (GDSC1:1012)'].mdl.cv_results_['std_test_score']

    models_dict_ridge_500 = pickle.load(open(ridge_500_model, 'rb'))
    x_ridge_500_model = np.array(models_dict_ridge_500['vorinostat (GDSC1:1012)'].mdl.cv_results_['param_alpha'])
    y_ridge_500_model = models_dict_ridge_500['vorinostat (GDSC1:1012)'].mdl.cv_results_['mean_test_score']
    err_ridge_500 = models_dict_ridge_500['vorinostat (GDSC1:1012)'].mdl.cv_results_['std_test_score']

    models_dict_ridge_1000 = pickle.load(open(ridge_1000_model, 'rb'))
    x_ridge_1000_model = np.array(models_dict_ridge_1000['vorinostat (GDSC1:1012)'].mdl.cv_results_['param_alpha'])
    y_ridge_1000_model = models_dict_ridge_1000['vorinostat (GDSC1:1012)'].mdl.cv_results_['mean_test_score']
    err_ridge_1000 = models_dict_ridge_1000['vorinostat (GDSC1:1012)'].mdl.cv_results_['std_test_score']

    models_dict_mlp_500 = pickle.load(open(mlp_500_model, 'rb'))
    x_mlp_500_model = np.array(models_dict_mlp_500['vorinostat (GDSC1:1012)'].mdl.cv_results_['param_alpha'])
    y_mlp_500_model = models_dict_mlp_500['vorinostat (GDSC1:1012)'].mdl.cv_results_['mean_test_score']
    err_mlp_500 = models_dict_mlp_500['vorinostat (GDSC1:1012)'].mdl.cv_results_['std_test_score']

    models_dict_mlp_1000 = pickle.load(open(mlp_1000_model, 'rb'))
    x_mlp_1000_model = np.array(models_dict_mlp_1000['vorinostat (GDSC1:1012)'].mdl.cv_results_['param_alpha'])
    y_mlp_1000_model = models_dict_mlp_1000['vorinostat (GDSC1:1012)'].mdl.cv_results_['mean_test_score']
    err_mlp_1000 = models_dict_mlp_1000['vorinostat (GDSC1:1012)'].mdl.cv_results_['std_test_score']



    plt.errorbar(x_lasso_500_model, y_lasso_500_model, err_lasso_500, marker='s', linestyle='dashed',
                 label='Lasso, 500 super genes')
    plt.errorbar(x_lasso_1000_model, y_lasso_1000_model, err_lasso_1000, marker='s', linestyle='dashed',
                 label='Lasso, 1000 super genes')
    plt.errorbar(x_ridge_500_model, y_ridge_500_model, err_ridge_500, marker='s', linestyle='dashed',
                 label='Ridge, 500 super genes')
    plt.errorbar(x_ridge_1000_model, y_ridge_1000_model, err_ridge_1000, marker='s', linestyle='dashed',
                 label='Ridge, 1000 super genes')
    plt.errorbar(x_mlp_500_model, y_mlp_500_model, err_mlp_500, marker='s', linestyle='dashed',
                 label='Multi-layer perceptron, 500 super genes')
    plt.errorbar(x_mlp_1000_model, y_mlp_1000_model, err_mlp_1000, marker='s', linestyle='dashed',
                 label='Multi-layer perceptron, 1000 super genes')
    plt.ylim(-5,1)

    plt.xlabel('alpha', fontweight='bold')
    plt.ylabel('mean CV test score', fontweight='bold')
    plt.legend(loc='lower right')
    plt.title('super genes models for Vorinostat')
    # plt.show()
    plt.savefig("model_name.png", format="png", dpi=1200)