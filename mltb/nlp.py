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
    """Return as df has 2 cols of terms and tfidf, sorted by descending tfidf.

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
    return pd.DataFrame(top_tokens, columns=['terms', 'tfidf'])


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
