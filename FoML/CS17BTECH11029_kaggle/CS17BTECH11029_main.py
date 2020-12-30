import os
import numpy as np
import pandas as pd
import os.path as osp

from lightgbm import LGBMClassifier
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler


if __name__ == '__main__':

    # Load Train and Test csv files
    train_df = pd.read_csv('train_input.csv')
    test_df = pd.read_csv('test_input.csv')

    np.random.seed(0) # set seed to reproduce results

    # Replace NaN values with 0
    train_df.fillna(value=0, inplace=True)
    test_df.fillna(value=0, inplace=True)

    # Convert data into numpy format
    x = train_df.to_numpy()[:, :-1]
    y = train_df.to_numpy()[:, -1].astype(np.int)
    x_test = test_df.to_numpy()
    
    # Normalize data in range 0 to 1 using MinMaxScalar
    scaler = MinMaxScaler()
    scaler.fit(x)
    x = scaler.transform(x)
    
    x_test = scaler.transform(test_df)

    print('dataset shapes:')
    print('x_train:', x.shape)
    print('y_train:', y.shape)
    print('x_test:', x_test.shape)
    
    # Set parameters for the LightGBM Model
    num_leaves= 4
    lr = 0.14
    # lr = 0.1353352832366127 
    n_estimators= 1024
    max_acc = 0
    # Train the Classifier
    clf = LGBMClassifier(learning_rate=lr, n_estimators=n_estimators, num_leaves=num_leaves, reg_alpha=0.5e-2, reg_lambda=1.5e-2)
    clf.fit(x, y)
    train_acc = accuracy_score(clf.predict(x), y)
    print('lr=', lr, 'n_est=', n_estimators, 'num_leaves=', num_leaves, 'train acc', train_acc)
    # Predict 
    y_pred = clf.predict(x_test).tolist()
    # Dump results to output file as csv
    submission_df = pd.DataFrame({
                    'Id': range(1, len(y_pred)+1),
                    'Category': y_pred
                    })
    submission_df.to_csv('test_output.csv', index=False)
