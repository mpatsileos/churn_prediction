import pickle
from data_preprocessing import data_preprocessing_trees
import numpy as np
from sklearn.metrics import roc_auc_score
import pandas as pd
import os

import argparse

parser = argparse.ArgumentParser(description='ExtraTrees Inference')
parser.add_argument('--data_path', default = 'data\churn_data.csv', type=str, help='load path of data csv file')
parser.add_argument('--save_preds', default = 'probs_labels\extratree_probs.csv', type=str, help='save path of extratree predictions csv file')
parser.add_argument('--save_labels', default = 'probs_labels\labels.csv', type=str, help='save path of labels csv file')
parser.add_argument('--model_load_path', default = 'model\extraTreesClassifier.pickle', type=str, help='path to save model')

def main():
    args = parser.parse_args()
    data_path = args.data_path
    save_preds = args.save_preds
    save_labels = args.save_labels
    load_path = args.model_load_path
    
    loaded_model = pickle.load(open(load_path, "rb"))

    _, test_data = data_preprocessing_trees(data_path, 0.2)
    x_test = np.concatenate((test_data.categorical_features,test_data.numerical_features), axis=-1)
    y_test = test_data.labels

    probs = loaded_model.predict_proba(x_test)
    print('roc auc score: ', roc_auc_score(y_test, probs[:,1]))
    
    prob_dict = {}
    for row_id, (prob, user_id) in enumerate(zip(probs, test_data.user_ids)):
        prob_dict[row_id] = [prob[1], user_id]
    
    prob_df = pd.DataFrame.from_dict(prob_dict, orient='index', columns=['pred', 'user_id'])
    prob_df.to_csv(save_preds)

    label_dict = {}
    for row_id, (label, user_id) in enumerate(zip(y_test, test_data.user_ids)):
        label_dict[row_id] = [label, user_id]
    
    label_df = pd.DataFrame.from_dict(label_dict, orient='index', columns=['label', 'user_id'])
    label_df.to_csv(save_labels)


if __name__ == "__main__":
    main()