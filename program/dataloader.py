import json
import os
import numpy as np
import pandas as pd

def get_index_drop_some_label_3(df, drop_rate=0.6):
    not3idx = df[df != 3].index.tolist()
    is3idx = df[df == 3].index
    some3idx = pd.Series(is3idx).sample(frac=1-drop_rate, random_state=1).tolist()
    return pd.Index(not3idx + some3idx)
    
def get_index_without_label_3(df):
    return df[df != 3].index
    #return df.index



def read_npy(dataset='raw'):

    train_data = np.load('../dataset/'+dataset+'/X_train.npy')
    train_label = np.load('../dataset/'+dataset+'/y_train.npy').astype(int)
    test_data = np.load('../dataset/'+dataset+'/X_test.npy')
    test_label = np.load('../dataset/'+dataset+'/y_test.npy').astype(int)

    # increase number of channels
    #train_data = np.concatenate([train_data, train_data], axis=2)
    #test_data = np.concatenate([test_data, test_data], axis=2)

    if train_label.min() != 0:
        train_label = train_label - 1
        test_label = test_label -1
    train_data = np.transpose(np.expand_dims(train_data, axis=1), (0, 1, 3, 2))
    test_data = np.transpose(np.expand_dims(test_data, axis=1), (0, 1, 3, 2))

    mask = np.where(np.isnan(train_data))
    train_data[mask] = np.nanmean(train_data)

    mask = np.where(np.isnan(test_data))
    test_data[mask] = np.nanmean(test_data)

    print(train_data.shape, train_label.shape, test_data.shape, test_label.shape)

    return train_data, train_label, test_data, test_label


