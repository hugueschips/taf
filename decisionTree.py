# -*- coding: utf-8 -*-
'''
Created on 16th September 2016
@author: Daniel Durrenberger
daniel.durrenberger@amaris.com

Python 2.7.12
Pandas 0.18.1
Numpy 1.11.1
Scipy 0.18.0
'''

import random
import numpy as np
import pandas as pd
import manage_data as md
import preprocessing as pp
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import metrics

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(2)
    labels = ['Sticking', 'Fine']
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def concat_df(coil_list):
    big_df = md.importPeak(coil_list[0]).dropna()
    for coil in coil_list[1:]:
        df = md.importPeak(coil).dropna()
        big_df = pd.concat([big_df, df], ignore_index=True)
    return big_df

def create_DB(df):
    feature_col = [
                    'Ximf0', 'Ximf1',
                    'Ximf2', 'Yimf0',
                    'Yimf1', 'Yimf2',
                    'thickness',
                    'speed'
                    ]
    X = df[feature_col].values
    Y = df[['sticking']]
    Y = np.array(Y.sticking.values[:], dtype=bool)
    return X, Y

def random_sample(coil_list):
    df = concat_df(coil_list)
    sticking_index = list(df[df.sticking==True].index)
    non_sticking_index = list(df[df.sticking==False].index)
    group_of_items = non_sticking_index               # a sequence or set will work here.
    num_to_select = len(df[df.sticking==True])        # set the number to select here.
    list_of_random_items = random.sample(group_of_items, num_to_select)
    list_of_random_items
    ind = sticking_index+list_of_random_items
    return df.loc[ind]

def result_per_coil(coil_list):
    sum_true = []
    sum_predict = []
    for coil in coil_list:
        df_test = concat_df([coil])
        X_test, Y_true = create_DB(df_test)
        Y_predict = clf.predict(X_test)
        sum_true.append(sum(Y_true))
        sum_predict.append(sum(Y_predict))
    d = {'truth':sum_true, 'prediction':sum_predict}
    result = pd.DataFrame(d, index=coil_list)
    return result, np.array(sum_true), np.array(sum_predict)
