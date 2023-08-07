import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from sklearn import preprocessing


class All_data:
  def __init__(self, user_ids, categorical_features, numerical_fetures, labels):
    self.user_ids = user_ids
    self.categorical_features = categorical_features
    self.numerical_features = numerical_fetures
    self.labels = labels

def data_preprocessing_trees(data_path, test_ratio = 0.2):
    df = pd.read_csv(data_path)
    cond = df['NoProducts']>3
    df.loc[cond,'NoProducts'] = 3
    user_ids = df['CustomerID'].to_numpy()
    labels_np = df['Inactive'].to_numpy()
    cat_data = df[['Geography', 'Age_Band']].to_numpy()
    num_data = df[['Gender', 'TenureYears', 'EstimatedIncome', 'BalanceEuros', 'NoProducts', 'CreditCardholder', 'CustomerWithLoan', 'Digital_TRX_ratio']].to_numpy()
    num_data[num_data=='Female'] = 1
    num_data[num_data=='Male'] = 0
    num_data = num_data.astype(float)

    user_ids_train, user_ids_test, labels_train, labels_test, cat_data_train, cat_data_test, num_data_train, num_data_test = train_test_split(user_ids, 
                                                                                                                                              labels_np, 
                                                                                                                                              cat_data, 
                                                                                                                                              num_data, 
                                                                                                                                              test_size = test_ratio,
                                                                                                                                            stratify=labels_np, 
                                                                                                                                            random_state=42)
    unq_cat1 = np.unique(cat_data_train[:,0])
    unq_cat2 = np.unique(cat_data_train[:,1])
    
    le1 = preprocessing.LabelEncoder()
    le1.fit(unq_cat1)
    le2 = preprocessing.LabelEncoder()
    le2.fit(unq_cat2)

    cat_data_train[:,0] = le1.transform(cat_data_train[:,0])
    cat_data_train[:,1] = le2.transform(cat_data_train[:,1])
    cat_data_test[:,0] = le1.transform(cat_data_test[:,0])
    cat_data_test[:,1] = le2.transform(cat_data_test[:,1])
    
    cat_data_train = cat_data_train.astype(np.int32)
    cat_data_test = cat_data_test.astype(np.int32)
    
    train_data = All_data(user_ids_train, cat_data_train, num_data_train, labels_train)
    test_data = All_data(user_ids_test, cat_data_test, num_data_test, labels_test)
    return train_data, test_data   
