# @Author  : Andrian Lee
# @Time    : 2022/3/11 12:44
# @File    : utils.py
import logging

import os

import pandas as pd
from pandas.core.base import PandasObject

import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, ConfusionMatrixDisplay

import xgboost as xgb

SEED = 42

SUPPORT_TABLE_SUFFIX = ['.xlsx', '.csv']

RENAME_MAP = {'H2O': "H$_{2}$O"}


def read_file(file_path: str, sheet_name=0):
    if os.path.splitext(file_path)[-1] in SUPPORT_TABLE_SUFFIX[0]:
        df_tmp = pd.read_excel(file_path, sheet_name=sheet_name)  # TODO: if None here, will get a dict for each sheet
        df = df_tmp.rename(columns=RENAME_MAP)
        return df
    elif os.path.splitext(file_path)[-1] in SUPPORT_TABLE_SUFFIX[1]:
        df_tmp = pd.read_csv(file_path)
        df = df_tmp.rename(columns=RENAME_MAP)
        return df
    else:
        raise TypeError('Other formats not supported yet.')


def out(df: pd.DataFrame, out_path):
    """output data to disk"""
    print('Output to {}...'.format(out_path))
    if os.path.splitext(out_path)[-1] in SUPPORT_TABLE_SUFFIX[0]:
        df.to_excel(out_path, index=False)
    elif os.path.splitext(out_path)[-1] in SUPPORT_TABLE_SUFFIX[1]:
        df.to_csv(out_path, index=False)


PandasObject.out = out


def sampling(data_all, label_all):
    # define pipeline
    over = SMOTE(random_state=SEED, sampling_strategy=0.5)  # sampling_strategy=.2 | minor:major
    under = RandomUnderSampler(random_state=SEED, sampling_strategy=0.8)  # sampling_strategy=.5 | minor:major
    # steps = [('o', over), ('u', under)]  # ('o', over),
    # pipeline = Pipeline(steps=steps)
    # # transform the dataset
    # data_all, label_all = pipeline.fit_resample(data_all, label_all)
    data_all, label_all = over.fit_resample(data_all, label_all)
    data_all, label_all = under.fit_resample(data_all, label_all)
    print(Counter(label_all))

    return data_all, label_all


def classification_metrics(Y_test, Y_test_pred):
    f1 = f1_score(Y_test, Y_test_pred)
    roc_auc = roc_auc_score(Y_test, Y_test_pred)
    acc = accuracy_score(Y_test, Y_test_pred, )
    print('f1:{0:.4f} ROC_AUC:{1:.4f} ACC:{2:.4f}'.format(f1, roc_auc, acc))
    logging.info('f1:{0:.4f} ROC_AUC:{1:.4f} ACC:{2:.4f}'.format(f1, roc_auc, acc))


def plot_feat_imp(model):
    xgb.plot_importance(model)
    # sns.set(rc={'figure.figsize': (5, 5)})
    plt.show()


def plot_kde(x, show=False):
    sns.kdeplot(data=x, hue='label')
    if show:
        plt.show()


def judge_path(_path):
    if not os.path.exists(_path):
        os.makedirs(_path)
