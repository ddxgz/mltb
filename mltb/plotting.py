import os
from os.path import join
import shutil
import logging
import json

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.decomposition import PCA
from typing import Tuple


def pca_plot(n_components, df, col_color: str = None,
             figsize: Tuple = (10, 8)):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(df)
    fig, ax = plt.subplots(figsize=figsize)

    from mpl_toolkits.mplot3d import Axes3D

    if col_color:
        col_c = df[col_color]
    else:
        col_c = None

    if n_components == 3:
        Axes3D(fig).scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2],
                            c=col_c, s=20)
    elif n_components == 2:
        sns.scatterplot(X_pca[:, 0], X_pca[:, 1], hue=col_c)
    else:
        print('Not supported n_components!')


def create_loss_plot(exp_dir, epochs, train_losses, test_losses, fig_name='loss'):
    """Plot losses and save.

    Args:
        exp_dir (str): experiment directory.
        epochs (list): list of epochs (x-axis of loss plot).
        train_losses (list): list with train loss during each epoch.
        test_losses (list): list with test loss during each epoch.

    """
    f = plt.figure()
    plt.title(f"{fig_name} plot")
    plt.xlabel("Epoch")
    plt.ylabel(f"{fig_name}")
    plt.grid(True)
    plt.plot(epochs, train_losses, 'b', marker='o', label=f'train {fig_name}')
    plt.plot(epochs, test_losses, 'r', marker='o', label=f'val {fig_name}')
    plt.legend()
    plt.savefig(join(exp_dir, f'{fig_name}.png'))
    plt.close(f)


def create_confusion_matrix_plot(exp_dir, y_true, y_pred, class_names):
    # f = plt.figure()
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot non-normalized confusion matrix
    ax1 = plot_confusion_matrix(ax1, y_true, y_pred, classes=class_names,
                                title='Confusion matrix')

    # Plot normalized confusion matrix
    ax2 = plot_confusion_matrix(ax2, y_true, y_pred, classes=class_names, normalize=True,
                                title='Normalized confusion matrix')
    fig.tight_layout()
    fig.legend()
    fig.savefig(join(exp_dir, 'confusion_matrix.png'))
    plt.close(fig)


def plot_confusion_matrix(ax, y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    From scikit-learn example: 
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # print("Normalized confusion matrix")
    # else:
        # print('Confusion matrix, without normalization')

    # print(cm)

    # fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    # fig.tight_layout()
    return ax
