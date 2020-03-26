import gc
import os
from datetime import datetime
import functools
import json

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn import metrics
from typing import List, Callable


RAND_STATE = 20200219
DATA_DIR = 'data'
EXP_DIR = 'exp'
TEST_DIR = 'test'


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(
            end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


def load_df(test_set: bool = False, nrows: int = None, sample_ratio: float = None, reduce_mem: bool = False) -> pd.DataFrame:
    if test_set:
        tran = pd.read_csv(f'{DATA_DIR}/test_transaction.csv', nrows=nrows)
        ids = pd.read_csv(f'{DATA_DIR}/test_identity.csv', nrows=nrows)
    else:
        tran = pd.read_csv(f'{DATA_DIR}/train_transaction.csv', nrows=nrows)
        ids = pd.read_csv(f'{DATA_DIR}/train_identity.csv', nrows=nrows)

    if sample_ratio:
        size = int(len(tran) * sample_ratio)
        tran = tran.sample(n=size, random_state=RAND_STATE)
        ids = ids.sample(n=size, random_state=RAND_STATE)
    df = tran.merge(ids, how='left', on='TransactionID')
    if reduce_mem:
        reduce_mem_usage(df)
    return df


def data_split_v1(X: pd.DataFrame, Y: pd.Series):
    # Y = df['isFraud']
    # X = df.drop(columns=['isFraud'])

    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=0.25, shuffle=False, random_state=RAND_STATE)
    return X_train, X_val, Y_train, Y_val


def data_split_oversample_v1(X: pd.DataFrame, Y: pd.Series):
    from imblearn.over_sampling import RandomOverSampler

    # Y = df['isFraud']
    # X = df.drop(columns=['isFraud'])

    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=0.25, shuffle=False, random_state=RAND_STATE)

    ros = RandomOverSampler(random_state=RAND_STATE)
    X_train, Y_train = ros.fit_resample(X_train, Y_train)

    return X_train, X_val, Y_train, Y_val


class Experiment:
    def __init__(self, df_nrows: int = None, transform_pipe: Pipeline = None,
                 data_split: Callable = None, model=None, model_class=None,
                 model_param: dict = None):
        self.df_nrows = df_nrows
        self.pipe = transform_pipe

        if data_split is None:
            self.data_split = data_split_v1
        else:
            self.data_split = data_split

        if model_class:
            self.model = model_class(**model_param)
        else:
            self.model = model

        self.model_param = model_param

    def transform(self, X):
        self.X = self.pipe.fit_transform(X)
        return self.X

    def run(self, save_exp: bool = True) -> float:
        self.df = load_df(nrows=self.df_nrows)
        self.y = self.df['isFraud']
        self.X = self.df.drop(columns=['isFraud'])

        X = self.transform(self.X)

        # X = self.X
        Y = self.y
        X_train, X_val, Y_train, Y_val = self.data_split(X, Y)
        # print(np.where(np.isnan(X_train)))
        self.model.fit(X_train, Y_train)

        Y_pred = self.model.predict(X_val)
        self.last_roc_auc = metrics.roc_auc_score(Y_val, Y_pred)

        if save_exp:
            self.save_result()

        return self.last_roc_auc

    def save_result(self):
        save_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        result = {}
        result['roc_auc'] = self.last_roc_auc
        # result['transform'] = self.pipe.get_params(deep=False)
        result['transform'] = list(self.pipe.named_steps.keys())
        result['model'] = self.model.__class__.__name__
        result['model_param'] = self.model_param
        result['data_split'] = self.data_split.__name__
        result['num_sample_rows'] = self.df_nrows
        result['save_time'] = save_time
        if hasattr(self.model, 'feature_importances_'):
            result['feature_importances_'] = dict(
                zip(self.X.columns, self.model.feature_importances_.tolist()))
        if hasattr(self.model, 'feature_importance'):
            result['feature_importances_'] = dict(
                zip(self.df.columns, self.model.feature_importance.tolist()))

        import pprint
        # pp = pprint.PrettyPrinter(indent=4)
        pprint.pprint(result, indent=4)

        if not os.path.exists(EXP_DIR):
            os.makedirs(EXP_DIR)
        with open(f'{EXP_DIR}/exp_{save_time}_{self.last_roc_auc:.4f}.json', 'w') as f:
            json.dump(result, f, indent=4)
