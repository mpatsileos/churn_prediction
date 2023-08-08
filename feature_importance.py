import pickle
from data_preprocessing import data_preprocessing_trees
import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import os

import argparse

parser = argparse.ArgumentParser(description='Feature Importance')
parser.add_argument('--model_load_path', default = 'model\extraTreesClassifier.pickle', type=str, help='path to save model')

def main():
    args = parser.parse_args()
    # load model
    load_path = args.model_load_path
    loaded_model = pickle.load(open(load_path, "rb"))

    # Normalizing the individual importances
    feature_importance_normalized = np.std([tree.feature_importances_ for tree in
                                        loaded_model.estimators_],
                                        axis = 0)
    
    columns = ['Geography', 'Age_Band', 'Gender', 'TenureYears', 'Income', 'Balance', 'NoProducts', 'CreditCard', 'Loan', 'TRX_ratio']
    # Plotting a Bar Graph to compare the models
    plt.bar(columns, feature_importance_normalized)
    plt.xlabel('Feature Labels')
    plt.ylabel('Feature Importances')
    plt.title('Comparison of different Feature Importances')
    plt.show()
    plt.savefig(fig_save_path)

    
if __name__ == "__main__":
    main()