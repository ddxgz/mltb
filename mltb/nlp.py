import random
from functools import partial

import numpy as np
import pandas as pd
import nltk
from nltk.tokenize.treebank import TreebankWordTokenizer, TreebankWordDetokenizer
# import nlpaug.flow as naf
import nlpaug.augmenter.word as naw
from typing import List, Callable


nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


def top_tfidf_terms(X, tfidf_param: dict, top_n: int = 50) -> pd.DataFrame:
    """Return as df has 2 cols of `term` and `tfidf`, sorted by descending tfidf.

    Parameters
    ----------
    X: text column to be transformed.

    tfidf_param: a dict contains TfidfVectorizer's parameters.

    top_n: the top n terms to return. If None, return all terms.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    vect = TfidfVectorizer(**tfidf_param)
    X_vector = vect.fit_transform(X)
    feature_names = vect.get_feature_names()

    tfidf_means = np.mean(X_vector.toarray(), axis=0)

    if top_n:
        # Returns the indices that would sort an array.
        top_idx = np.argsort(tfidf_means)[::-1][:top_n]
    else:
        top_idx = np.argsort(tfidf_means)[::-1]

    top_tokens = [(feature_names[i], tfidf_means[i]) for i in top_idx]
    return pd.DataFrame(top_tokens, columns=['term', 'tfidf'])


def text_random_crop(rec, crop_by: str = 'word', crop_ratio: float = 0.1):
    if crop_by == 'word':
        sents = nltk.word_tokenize(rec)
    elif crop_by == 'sentence':
        sents = nltk.sent_tokenize(rec)
    else:
        sents = rec
    size = len(sents)
    chop_size = size // (1 / crop_ratio)
    chop_offset = random.randint(0, int(chop_size))
    sents_chop = sents[chop_offset:size - chop_offset - 1]

    d = TreebankWordDetokenizer()
    return d.detokenize(sents_chop)


def word_substitution(text, aug_src='wordnet'):
    aug = naw.SynonymAug(aug_src=aug_src)
    augmented_text = aug.augment(text)
    return augmented_text


def count_sentence(text):
    sents = nltk.sent_tokenize(text)
    return len(sents)


def count_token(text):
    tokens = nltk.word_tokenize(text)
    return len(tokens)


def chop_text(text, i, level):
    sents = nltk.sent_tokenize(text)

    size = len(sents)

    chop_size = size // level

    if i + 1 == level:
        sents_chop = sents[chop_size * i:]
    else:
        sents_chop = sents[chop_size * i:chop_size * (i + 1)]

    return ' '.join(sents_chop)


def text_chop_augment(features: pd.DataFrame, labels, col: str = 'fulltext',
                      level: int = 3, bypass_limit: int = 50, shuffle: bool = True,
                      new_col_name: str = None, *args, **kwargs):
    """Used to augment the text col of the data set, the augmented copies will
    be chopped into multiple pieces.

    Parameters
    ----------
    col : the columns name of the fulltext columns to be augmented.

    level : how many copies to append to the dataset. 0 means no append.

    bypass_limit : the lower limit of tokens to bypass chopping for the short
    text

    shuffle : if need to shuffle the samples after chopping

    new_col_name : rename the `col` to `new_col_name`. It'll overwrite the col
    with the same name to `new_col_name`
    """
    labels = pd.DataFrame(labels, columns=[
        f'label_{i}' for i in range(labels.shape[1])])
    features = features.reset_index(drop=True)
    df = pd.concat([features, labels], axis='columns')

    # chop based on sentences
    df['n_tok'] = df[col].apply(count_token)

    bypass = df[df['n_tok'] < bypass_limit]

    chop = df[~df.index.isin(bypass.index)]
    len_ori = chop.shape[0]
    chop = pd.concat([chop] * int(level), ignore_index=True)

    for i in range(level):
        ind = i * len_ori
        offset = len_ori * (i + 1)
        chop_text_ = partial(chop_text, i=i, level=level)
        chop.iloc[ind:offset][col] = chop.iloc[ind:offset][col].apply(
            chop_text_)

    df = pd.concat([chop, bypass])

    df = df.drop(columns=['n_tok'])

    if shuffle:
        df = df.sample(frac=1)

    df = df.reset_index(drop=True)
    # df = df.drop(columns=['index'])

    label_cols = [col for col in df.columns if col.startswith('label')]
    labels = df[label_cols]
    features = df.drop(columns=label_cols)

    if new_col_name:
        # features = features.rename(columns={col: new_col_name})
        features[new_col_name] = features[col]
        features = df.drop(columns=[col])

    return features, labels


def text_augment(features: pd.DataFrame, labels, col: str = 'description', level: int = 0,
                 oversample_weight: int = None, crop_ratio: float = 0.1,
                 aug_method: Callable = None, *args, **kwargs):
    """Used to augment the text col of the data set, the augmented copies will
    be randomly transformed a little

    Parameters
    ----------
    col : the columns name of the text columns to be augmented.
    level : how many copies to append to the dataset. 0 means no append.
    aug_method : the method used to augment text, if not specified, use random
    word crop.

    crop_ratio : How much ratio of the text to be raondomly cropped from head or
    tail. It actually crops out about 1/ratio of the text.
    """
    len_ori = features.shape[0]

    features = pd.concat([features] * (int(level) + 1), ignore_index=True)
    labels = np.concatenate([labels] * (int(level) + 1), axis=0)

    if aug_method is not None:
        text_aug_method = aug_method
    else:
        text_aug_method = partial(
            text_random_crop, crop_by='word', crop_ratio=crop_ratio)
    # see
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#indexing-view-versus-copy
    features[col].iloc[len_ori:] = features[col].iloc[len_ori:].apply(
        text_aug_method)

    return features, labels
