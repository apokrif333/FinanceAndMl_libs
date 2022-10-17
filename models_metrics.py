from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
from plotly.offline import iplot
from termcolor import colored
from pprint import pprint

import datetime
import os
import matplotlib
import seaborn as sns
import plotly.figure_factory as ff
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def show_metrics(model, valid, y_valid, test, y_test) -> tuple:
    """
    Вывести метрики от модели. Accuracy, Precision, Recall, F1, Roc_Auc

    Returns
    -------
    tuple: tuple({}, {})
        Возвращает два словаря, где в первом данные по valid, а во втором данные по test

    Parameters
    ----------
    :param model: модель, которая имеет функцию predict()
    :param valid: pd.DataFrame или многомерный numpy.array
    :param y_valid: pd.Series или numpy.array
    :param test: pd.DataFrame или многомерный numpy.array. Установить None, если test не нужен.
    :param y_test: pd.Series или numpy.array. Установить None, если test не нужен.

    """
    dict_valid = {}
    dict_test = {}
    
    valid_predict = model.predict(valid)
    dict_valid['Accuracy'] = accuracy_score(y_valid, valid_predict)
    dict_valid['Precision'] = precision_score(y_valid, valid_predict)
    dict_valid['Recall'] = recall_score(y_valid, valid_predict)
    dict_valid['F1'] = f1_score(y_valid, valid_predict)
    dict_valid['Roc_Auc'] = roc_auc_score(y_valid, valid_predict)
    
    if (test is not None) and (y_test is not None):
        test_predict = model.predict(test)
        dict_test['Accuracy'] = accuracy_score(y_valid, test_predict)
        dict_test['Precision'] = precision_score(y_valid, test_predict)
        dict_test['Recall'] = recall_score(y_valid, test_predict)
        dict_test['F1'] = f1_score(y_valid, test_predict)
        dict_test['Roc_Auc'] = roc_auc_score(y_valid, test_predict)
        
    return dict_valid, dict_test


def cat_feature_importance(cat_model, figsize=(15, 15), max_features=50):
    """
    Отрисуем важность фичей для CatBoostClassifier

    Returns
    -------
    bar_plot: sns.barplot
        Изображение от библиотеки seaborn.

    Parameters
    ----------
    :param cat_model: обученный CatBoostClassifier
    :param figsize: размер рисунка (ширина, высота)
    :param max_features: максимальное количество фичей для отображения

    """
    plt.style.use('ggplot')
    sns.set_color_codes("pastel")

    features_weight = {}
    for key, value in zip(list(cat_model.feature_names_), [[x] for x in list(cat_model.get_feature_importance())]):
        features_weight[key] = value
    features_weight = pd.DataFrame.from_dict(features_weight).sort_values(by=0, axis=1, ascending=0)
    features_weight = features_weight.iloc[:, :max_features]

    plt.figure(figsize=figsize)
    bar_plot = sns.barplot(x=features_weight.values[0], y=list(features_weight.columns))
    for p in bar_plot.patches:
        x = p.get_x() + p.get_width()
        y = p.get_y() + p.get_height()
        value = round(p.get_width(), 2)
        bar_plot.text(x, y, value, ha="left")

    return bar_plot


def plotly_conf_matrix_multi(model, X, y, target_names: list, normalize: bool):
    """
    Confusion matrix, поддерживающая мультикласс, отрисованная на plotly

    Returns
    -------
    iplot:
        Матрица в ввиде рисунка plotly

    Parameters
    ----------
    :param model: модель имеющая функцию predict()
    :param X: Пространство для предсказания. pd.DataFrame или многомерный np.array
    :param y: Целевой вектор. pd.Series или np.array
    :param target_names: названия для целевых классов. Например для бинарной задачи [0, 1].
    :param normalize: отображать значения в матрице в долях или в абсолюте

    """
    cm = confusion_matrix(y, model.predict(X))

    accuracy = np.round(np.trace(cm) / float(np.sum(cm)), 2)
    misclass = 1 - accuracy
    print(f"Predicted label \nAccuracy={accuracy}; Misclass={misclass}")

    dict_metrics = {'Precision': [], 'Recall': [], 'F1': []}
    for i in range(len(target_names)):
        dict_metrics['Precision'].append(cm[i, i] / cm[:, i].sum())
        dict_metrics['Recall'].append(cm[i, i] / cm[i].sum())
        dict_metrics['F1'].append(2 / (dict_metrics['Precision'][-1] ** (-1) + dict_metrics['Recall'][-1] ** (-1)))
    df_metrics = pd.DataFrame.from_dict(dict_metrics)
    print(df_metrics.set_index(pd.Series(target_names)))

    title = "Confusion Matrix"
    if normalize:
        cm = np.round(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], 2)
        title = "Confusion Matrix (share)"

    fig = go.Figure(
        ff.create_annotated_heatmap(
            z=cm,
            x=target_names,
            y=target_names,
            colorscale='GnBu',  # PuBu GnBu
            showscale=True
        )
    )
    fig['layout'].update(
        title=title,
        width=700,
        height=700,
        autosize=False,
        xaxis=go.layout.XAxis(
            title=go.layout.xaxis.Title(text="Predicted lable"),
            color='black'
        ),
        yaxis=go.layout.YAxis(
            title=go.layout.yaxis.Title(text="True lable"),
            color='black'
        ),
    )

    return iplot(fig, show_link=False)


if __name__ == '__main__':
    pass
