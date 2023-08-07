import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, classification_report
import os

def main():
    tree_preds = os.path.join('probs_labels','extratree_probs.csv')
    labels_csv = os.path.join('probs_labels', 'labels.csv')

    df_tree = pd.read_csv(tree_preds)
    df_labels = pd.read_csv(labels_csv)

    tree_probs = df_tree['pred'].to_numpy()
    labels = df_labels['label'].to_numpy()
    predictions = np.zeros(tree_probs.shape)

    auc = roc_auc_score(labels, tree_probs)
    print('roc auc score: ', auc)

    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for threshold in thresholds:
        predictions[tree_probs>threshold] = 1
        print(" ")
        print("Classification Report for threshold: ", threshold)
        print(classification_report(labels, predictions))
        print(" ")
        predictions = np.zeros(tree_probs.shape)





if __name__ == "__main__":
    main()