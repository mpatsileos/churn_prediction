import pickle
from data_preprocessing import data_preprocessing_trees
import numpy as np
from sklearn.metrics import roc_auc_score
import pandas as pd
import os

def main():
    save_preds = os.path.join('probs_labels', 'extratree_probs.csv')
    labels = os.path.join('probs_labels', 'labels.csv')
    data_path = os.path.join('data','churn_data.csv')
    # load model
    load_path = os.path.join('model','extraTreesClassifier.pickle')
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
    label_df.to_csv(labels)


if __name__ == "__main__":
    main()