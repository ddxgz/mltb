""" This module includes a few scikit-learn transformers that use pandas.DataFrame
as input and output a data frame retains the same (almost) structure.
"""
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
import torch
from transformers import AutoTokenizer, AutoModel, BertConfig
import nltk
from typing import List, Callable


def missing_ratio_col(df):
    df_na = (df.isna().sum() / len(df)) * 100
    if isinstance(df, pd.DataFrame):
        df_na = df_na.drop(
            df_na[df_na == 0].index).sort_values(ascending=False)
        missing_data = pd.DataFrame({'Missing Ratio %': df_na})
    else:
        missing_data = pd.DataFrame({'Missing Ratio %': df_na}, index=[0])

    return missing_data


class OneHotDfEncoder(TransformerMixin, BaseEstimator):
    """Encode categoricals in pd.DataFrame, and return the data frame with the 
    encoded categories. The `save()` and `load()` enables load from trained 
    encoder to use for new data.

    Parameters:
    ----------
    cols: the cloumns in the dataframe to be encoded

    save_to: the path + file name to persist the encoder after fit and
    transform, without 'joblib.gz' suffix

    load_from: the path + file name to load fitted encoder, without 'joblib.gz' suffix
    """

    def __init__(self, cols: List[str], load_from=None, save_to=None):
        self.cols = cols
        self.load_from = load_from
        if load_from:
            self.encoder = self.load()
        else:
            self.encoder = preprocessing.OneHotEncoder(handle_unknown='ignore')

        self.save_to = save_to

    def fit(self, X, y=None):
        return self

    def transform(self, dff):
        if not self.cols:
            return dff

        cats = dff.loc[:, self.cols]
        cats = cats.astype('category')

        if self.load_from:
            traned = self.encoder.transform(cats)
        else:
            traned = self.encoder.fit_transform(cats)

        if self.save_to:
            self.save()

        new_cols = [f'{self.cols[i]}_{cat}'
                    for i, cat_col in enumerate(self.encoder.categories_)
                    for cat in cat_col]
        cats_enc = pd.DataFrame.sparse.from_spmatrix(traned,
                                                     columns=new_cols,
                                                     index=dff.index)

        nums = dff.drop(columns=self.cols)
        return pd.concat([nums, cats_enc], axis='columns')

    def save(self):
        import joblib

        filename = f'{self.save_to}.joblib.gz'
        m = joblib.dump(self.encoder, filename, compress=3)

    def load(self):
        import joblib

        filename = f'{self.load_from}.joblib.gz'
        return joblib.load(filename)


class ColsNaMedianFiller(TransformerMixin, BaseEstimator):
    def __init__(self, cols: List[str] = []):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, df):
        for col in self.cols:
            median = df[col].median()
            df[col].fillna(median, inplace=True)

        return df


class ColsNaNegFiller(TransformerMixin, BaseEstimator):
    def __init__(self, cols: List[str] = []):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, df):
        for col in self.cols:
            df[col].fillna(-999, inplace=True)

        return df


class ColsNaStrFiller(TransformerMixin, BaseEstimator):
    def __init__(self, cols: List[str] = [], fill_str: str='missing'):
        self.cols = cols
        self.fill_str = fill_str

    def fit(self, X, y=None):
        return self

    def transform(self, df):
        for col in self.cols:
            df[col] = df[col].cat.add_categories(self.fill_str).fillna(self.fill_str)
            # df[col].fillna(self.fill_str, inplace=True)

        return df


class DropByNaRatioTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, cols, ratio: float = 0.5):
        self.cols = cols
        self.ratio = ratio

    def fit(self, X, y=None):
        return self

    def transform(self, df):
        # nums = df.drop(columns=cols_cat)
        # cols_num = df[~df[cols_cat]].columns
        cols_num = self.cols
        nums = df[cols_num]

        ratio = self.ratio * 100
        missings = missing_ratio_col(nums)
        # print(missings)
        inds = missings[missings['Missing Ratio %'] > ratio].index
        df = df.drop(columns=inds)
        return df


class ColsDropper(TransformerMixin, BaseEstimator):
    def __init__(self, cols: List[str]):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, df):
        return df.drop(columns=self.cols)


class DataFrameSelector(TransformerMixin, BaseEstimator):
    def __init__(self, col_names):
        self.attribute_names = col_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # print(X[self.attribute_names].columns)

        return X[self.attribute_names].values


class LabelDfEncoder(TransformerMixin, BaseEstimator):
    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, df):

        for col in self.cols:
            le = preprocessing.LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str).values)
        return df


class FrequencyEncoder(TransformerMixin, BaseEstimator):
    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, df):
        for col in self.cols:
            vc = df[col].value_counts(dropna=True, normalize=True).to_dict()
            vc[-1] = -1
            nm = col + '_FE'
            df[nm] = df[col].map(vc)
            df[nm] = df[nm].astype('float32')
        return df


class CombineEncoder(TransformerMixin, BaseEstimator):
    def __init__(self, cols_pairs: List[List[str]]):
        self.cols_pairs = cols_pairs

    def fit(self, X, y=None):
        return self

    def transform(self, df):
        for pair in self.cols_pairs:
            col1 = pair[0]
            col2 = pair[1]
            nm = col1 + '_' + col2
            df[nm] = df[col1].astype(str) + '_' + df[col2].astype(str)
            df[nm] = df[nm].astype('category')
            # print(nm, ', ', end='')
        return df


class AggregateEncoder(TransformerMixin, BaseEstimator):
    def __init__(self, main_cols: List[str], uids: List[str], aggr_types: List[str],
                 fill_na: bool = True, use_na: bool = False):
        self.main_cols = main_cols
        self.uids = uids
        self.aggr_types = aggr_types
        self.use_na = use_na
        self.fill_na = fill_na

    def fit(self, X, y=None):
        return self

    def transform(self, df):
        for col in self.main_cols:
            for uid in self.uids:
                for aggr_type in self.aggr_types:
                    col_new = f'{col}_{uid}_{aggr_type}'
                    tmp = df.groupby([uid])[col].agg([aggr_type]).reset_index().rename(
                        columns={aggr_type: col_new})
                    tmp.index = list(tmp[uid])
                    tmp = tmp[col_new].to_dict()
                    df[col_new] = df[uid].map(tmp).astype('float32')
                    if self.fill_na:
                        df[col_new].fillna(-1, inplace=True)
        return df
