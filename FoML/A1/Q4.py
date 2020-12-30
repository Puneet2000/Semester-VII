#!/usr/bin/env python
# coding: utf-8

import os
import sys
import numpy as np
import pandas as pd
import os.path as osp
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer, Normalizer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_squared_error
from scipy.special import expit, logit
from scipy.optimize import minimize

import scipy

#
# class LogLink:
#     """The log link function g(x)=log(x)."""
#
#     def __call__(self, y_pred):
#         return np.log(y_pred)
#
#     def derivative(self, y_pred):
#         return 1 / y_pred
#
#     def inverse(self, lin_pred):
#         return np.exp(lin_pred)
#
#     def inverse_derivative(self, lin_pred):
#         return np.exp(lin_pred)
#
# def _y_pred_deviance_derivative(coef, X, y, weights, family, link):
#     """Compute y_pred and the derivative of the deviance w.r.t coef."""
#     lin_pred = _safe_lin_pred(X, coef)
#     y_pred = link.inverse(lin_pred)
#     d1 = link.inverse_derivative(lin_pred)
#     temp = d1 * family.deviance_derivative(y, y_pred, weights)
#     if coef.size == X.shape[1] + 1:
#         devp = np.concatenate(([temp.sum()], temp @ X))
#     else:
#         devp = temp @ X  # same as X.T @ temp
#     return y_pred, devp

# class PoissonRegressor:
#     def __init__(self, n_features):
#         self.n_params = n_features + 1
#         self.weight = np.random.randn(self.n_params)
#         # self.weight = np.zeros(n_features+1)

#     def fit(self, X, y, maxiter=1000000000):

#         def objective(w, X, y):
#             wx = X.dot(w[1:]) + w[0]
#             if np.isnan(wx).any():
#                 raise Exception()
#             return -(y * wx - np.exp(wx)).sum()

#         def objective_der(w, X, y):
#             wx = X.dot(w[1:]) + w[0]
#             der = np.zeros(self.n_params)
#             der[0] = -(y - np.exp(wx)).sum()
#             for j in range(self.n_params-1):
#                 X_j = X[:, j]
#                 der[j+1] = -(y * X_j - np.exp(wx) * X_j).sum()

#             return der

#         print('shapes', X.shape, y.shape)
#         print('init:', objective(self.weight, X, y))
#         res = minimize(
#             objective, self.weight, method="BFGS", jac=objective_der,
#             options={
#                 # 'xatol': 1e-4,
#                 'disp': True
#             },
#             args=(X, y)
#         )
#         print('final weights obtained from optimizer', res.x)
#         self.weight = res.x.copy()
#         print('final:', objective(self.weight, X, y))
#         return res

#     def predict(self, X):

#         return np.exp(X.dot(self.weight[1:]) + self.weight[0])

def train_validation_split(df, train_dates=[1, 16]):

    df['day'] = df['datetime'].dt.day
    training_idx = (train_dates[0] <= df['day']) & (df['day'] <= train_dates[1])
    del df['day']
    train_df = df[training_idx]
    validation_df = df[~training_idx]

    for attr in ['year', 'month', 'day', 'hour']:
        train_df[attr] = getattr(train_df['datetime'].dt, attr)
        validation_df[attr] = getattr(validation_df['datetime'].dt, attr)

    return train_df, validation_df

def evaluate(model, test_df):
    y_pred = model.predict(test_df)
    print('y_pred', y_pred[:10])
    print('test_df', test_df[:10]['count'].tolist())
    # print(scipy.stats.pearsonr(y_pred[:10], test_df[:10]['count'].tolist()))
    return np.sqrt(mean_squared_error(test_df['count'], y_pred))

if __name__ == '__main__':

    # np.random.seed(0)

    plt.style.use('ggplot')
    dataset_dir = 'bike-sharing-demand'
    train_df = pd.read_csv(osp.join(dataset_dir, 'train.csv'))
    test_df = pd.read_csv(osp.join(dataset_dir, 'test.csv'))

    if not osp.exists('imgs'):
        os.mkdir('imgs')

    print('train_df')
    print(train_df.head())
    print('test_df')
    print(test_df.head())

    # print('train_df columns:', train_df.columns)
    # print('test_df columns:', test_df.columns)

    # 80-20 split for dates in range 1-20 is
    # train: 1-16
    # validation: 17-20

    train_df['datetime'] = pd.to_datetime(train_df['datetime'], format='%Y-%m-%d %H:%M:%S')
    train_df, validation_df = train_validation_split(train_df)

    print('split train-validation')
    print('train_df len', len(train_df))
    print(train_df.head())
    print('validation_df len', len(validation_df))
    print(validation_df.head())

    # test df
    test_df = pd.read_csv(osp.join(dataset_dir, 'test.csv'))
    test_df['datetime'] = pd.to_datetime(test_df['datetime'], format='%Y-%m-%d %H:%M:%S')

    print('test_df len', len(test_df))
    print(test_df.head())

    # print cols
    print('train_df cols', train_df.columns)
    print('test_df cols', test_df.columns)

    # 4.2
    # mean counts
    print('mean per year:', train_df.groupby('year')[['count']].aggregate(np.sum).mean()['count'])
    print('mean per month:', train_df.groupby(['year', 'month'])[['count']].aggregate(np.sum).mean()['count'])
    print('mean per day:', train_df.groupby(['year', 'month', 'day'])[['count']].aggregate(np.sum).mean()['count'])
    print('mean per hour:', train_df.groupby(['year', 'month', 'day', 'hour'])[['count']].aggregate(np.sum).mean()['count'])

    print('max per year:', train_df.groupby('year')[['count']].aggregate(np.sum).max()['count'])
    print('max per month:', train_df.groupby(['year', 'month'])[['count']].aggregate(np.sum).max()['count'])
    print('max per day:', train_df.groupby(['year', 'month', 'day'])[['count']].aggregate(np.sum).max()['count'])
    print('max per hour:', train_df.groupby(['year', 'month', 'day', 'hour'])[['count']].aggregate(np.sum).max()['count'])

    print('counts in season', train_df.groupby('season').aggregate(np.sum)['count'])
    print('counts in holiday', train_df.groupby('holiday').aggregate(np.sum)['count'])
    print('counts in workingday', train_df.groupby('workingday').aggregate(np.sum)['count'])
    print('counts in weather', train_df.groupby('weather').aggregate(np.sum)['count'])

    # 4.3
    # plot against any 5 featrues
    # fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(8, 12))
    # 5 features:
    # season
    # holiday
    # workingday
    # weather
    # temp
    # atemp
    # humidity
    # windspeed

    for feature in ['weather', 'temp', 'atemp', 'humidity', 'windspeed']:

        features_df = train_df.groupby(feature).agg(np.sum)
        features_df[['count']].plot.bar()
        plt.title(feature)
        image_path = osp.join('imgs', 'feature_{}.jpg'.format(feature))
        # plt.savefig(image_path, dpi=40)
        print('plotting feature', feature, 'at', image_path)

    plt.close('all')

    features = ['year', 'month', 'workingday', 'hour', 'holiday', 'weather', 'atemp', 'humidity', 'windspeed', 'season']
    for f in ['holiday', 'atemp']:
        features.remove(f)
    linear_model_preprocessor = ColumnTransformer([
        # ('passthrough_numeric', 'passthrough', features)
        # ('passthrough_numeric', Normalizer(norm='l2'), features)
        ('passthrough_numeric', StandardScaler(), features)
    ], remainder='drop')

    # poisson_regressor = PoissonRegressor(len(features))
    poisson_regressor = PoissonRegressor(alpha=0)
    # poisson_regressor = Ridge()
    model = Pipeline([
        ('preprocessor', linear_model_preprocessor),
        ('regressor', poisson_regressor)
    ])
    print('fitting model...')
    model.fit(train_df, train_df['count'])
    print('evaluating model over train...')
    error = evaluate(model, train_df)
    print('error', error)
    print('evaluating model over test...')
    error = evaluate(model, validation_df)
    print('error', error)
    print('params:', poisson_regressor.coef_)
    print('intercept:', poisson_regressor.intercept_)
    print(dir(poisson_regressor))

