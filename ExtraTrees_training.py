import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from data_preprocessing import data_preprocessing_trees
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, KFold
import pickle

def main():
    # Extra Trees
    data_path = 'churn_data.csv'
    model_save_path = 'extraTreesClassifier.pickle'
    train_data, test_data = data_preprocessing_trees(data_path, 0.2)
    x_train = np.concatenate((train_data.categorical_features,train_data.numerical_features), axis=-1)
    y_train = train_data.labels
    x_test = np.concatenate((test_data.categorical_features,test_data.numerical_features), axis=-1)
    y_test = test_data.labels

    params = {
    'n_estimators': [100, 200, 500],
    'criterion': ['gini', 'entropy'],
    'min_samples_split': [1,2,4,5],
    'min_samples_leaf': [1,2,4,5],
    'max_leaf_nodes': [4,10,20,50,None]
    }

    gs3 = GridSearchCV(ExtraTreesClassifier(n_jobs=-1), params, n_jobs=-1, cv=KFold(n_splits=4), scoring='roc_auc')
    gs3.fit(x_train, y_train)

    print('Best score:', gs3.best_score_)
    print('Best score:', gs3.best_params_)

    model = ExtraTreesClassifier(random_state=42, criterion=gs3.best_params_['criterion'], 
                                 max_leaf_nodes= gs3.best_params_['max_leaf_nodes'], 
                                 min_samples_leaf= gs3.best_params_['min_samples_leaf'], 
                                 min_samples_split=gs3.best_params_['min_samples_split'], 
                                 n_estimators=gs3.best_params_['n_estimators'])
    
    model.fit(x_train, y_train)
    y_test_prob = model.predict_proba(x_test)
    print(y_test_prob[:,1])
    print('roc auc score: ', roc_auc_score(y_test, y_test_prob[:,1]))

    # save model
    pickle.dump(model, open(model_save_path, "wb"))
    

if __name__ == "__main__":
    main()