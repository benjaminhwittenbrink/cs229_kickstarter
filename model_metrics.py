import numpy as np
import pandas as pd

from sklearn import metrics
import sklearn.linear_model as lm
import sklearn.ensemble as em
from sklearn.svm import SVC


def format_results(models, X_test, y_test, model_names=None):
    """
    Calculate performance statistics for models on test data.
    Also return dataframe with test predictions.
    """
    pred_df = pd.DataFrame()
    column_header = ["model", "accuracy", "f1_score", "precision_1", "precision_0", "recall_1", "recall_0", "roc_auc", "brier"]
    rows = []
    if model_names is None: model_names = [type(m).__name__ for m in models]
    for i, m in enumerate(models):
        # get model type
        model = model_names[i]
        # calculate predictions
        if model == "LinearRegression":
            y_pred = m.predict(X_test.drop(columns=['dummy_cat_id_1','dummy_cat_parent_id_1.0']))
        else: y_pred = m.predict(X_test)
        pred_df[model + "_pred"] = y_pred
        # make sure values are binary 0-1
        y_pred = np.where(np.round(y_pred)>0, 1, 0)
        # calculate performance
        perf = calculate_performance(y_test, np.round(y_pred))
        rows.append( [model]+perf )
    # to df
    stats_df = pd.DataFrame(rows, columns=column_header)
    return stats_df, pred_df


def calculate_performance(y, y_pred):
    """
    Calculate scores for performance of a prediction.
    """
    acc = metrics.accuracy_score(y, y_pred)
    brier = metrics.brier_score_loss(y, y_pred)
    f1 = metrics.f1_score(y, y_pred)
    p1 = metrics.precision_score(y, y_pred, pos_label=1)
    p0 = metrics.precision_score(y, y_pred, pos_label=0)
    r1 = metrics.recall_score(y, y_pred, pos_label=1)
    r0 = metrics.recall_score(y, y_pred, pos_label=0)
    roc = metrics.roc_auc_score(y, y_pred)
    return [acc, f1, p1, p0, r1, r0, roc, brier]
