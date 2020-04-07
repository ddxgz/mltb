import numpy as np
import pandas as pd
from sklearn import metrics
from typing import List


def classification_report_avg(y_true, y_pred, cols_avg: List[str] = None):
    if isinstance(y_true, list):
        y_true = np.array(y_true)
    if isinstance(y_pred, list):
        y_pred = np.array(y_pred)

    report = metrics.classification_report(
        y_true, y_pred, output_dict=True, zero_division=0)
    df_report = pd.DataFrame(report).transpose()

    if cols_avg is not None:
        cols = cols_avg
    else:
        cols = ['micro avg', 'macro avg', 'weighted avg', 'samples avg']

    return df_report.loc[cols, ]


def best_fbeta_score(true_labels, predictions, beta=1, average='micro', **kwargs):
    fbeta = 0
    thr_bst = 0
    for thr in range(0, 6):
        Y_predicted = (predictions > (thr * 0.1))

        f = metrics.fbeta_score(
            true_labels, Y_predicted, beta=beta, average=average, **kwargs)
        if f > fbeta:
            fbeta = f
            thr_bst = thr * 0.1

    return fbeta, thr


def best_prec_score(true_labels, predictions, average='micro', **kwargs):
    fbeta = 0
    thr_bst = 0
    for thr in range(0, 6):
        Y_predicted = (predictions > (thr * 0.1))

        f = metrics.average_precision_score(
            true_labels, Y_predicted, average='micro', **kwargs)
        if f > fbeta:
            fbeta = f
            thr_bst = thr * 0.1

    return fbeta, thr
