import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.preprocessing import MaxAbsScaler 
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, auc

import optuna
from functools import partial


def train_and_predict_toxicity(model, X_train, X_val, X_test, y_train):
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)

    if hasattr(model, 'predict_proba'):
        # Вероятности для положительного класса
        y_pred_proba_train = model.predict_proba(X_train)[:, 1]
        y_pred_proba_val = model.predict_proba(X_val)[:, 1]
        y_pred_proba_test = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, 'decision_function'):
        y_pred_proba_train = model.decision_function(X_train)
        y_pred_proba_val = model.decision_function(X_val)
        y_pred_proba_test = model.decision_function(X_test)
    else:
        y_pred_proba_train = None
        y_pred_proba_val = None
        y_pred_proba_test = None

    return y_pred_train, y_pred_val, y_pred_test, y_pred_proba_train, y_pred_proba_val, y_pred_proba_test


def show_metrics(y_train, y_val, y_test, y_pred_train, y_pred_val, y_pred_test, y_pred_proba_train, y_pred_proba_val, y_pred_proba_test, title):
    precision_train = precision_score(y_train, y_pred_train)
    precision_val = precision_score(y_val, y_pred_val)
    precision_test = precision_score(y_test, y_pred_test)

    recall_train = recall_score(y_train, y_pred_train)
    recall_val = recall_score(y_val, y_pred_val)
    recall_test = recall_score(y_test, y_pred_test)

    f1_train = f1_score(y_train, y_pred_train)
    f1_val = f1_score(y_val, y_pred_val)
    f1_test = f1_score(y_test, y_pred_test)

    print(title + '\n')

    print('Precision\t\tRecall\t\t\tF1\n')
    print(f'Train: {precision_train:.2f}\t\tTrain: {recall_train:.2f}\t\tTrain: {f1_train:.2f}')
    print(f'Val: {precision_val:.2f}\t\tVal: {recall_val:.2f}\t\tVal: {f1_val:.2f}')
    print(f'Test: {precision_test:.2f}\t\tTest: {recall_test:.2f}\t\tTest: {f1_test:.2f}')

    def show_confusion_matrix(y_true, y_pred, sample_name, ax):
        cm = confusion_matrix(y_true, y_pred, normalize='all')
        labels = ['Non-Toxic (0)', 'Toxic (1)']
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)
        
        sns.heatmap(cm_df, annot=True, fmt=".2%", cmap='twilight_shifted', ax=ax)
        ax.set_title(f'Confusion matrix ({sample_name})')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')

    def show_roc_curve(y_true, y_pred_proba, model_name, sample_name, ax):
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc = roc_auc_score(y_true, y_pred_proba)

        ax.plot(fpr, tpr)
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        ax.set_title(f'{model_name} ({sample_name}) ROC (AUC = {auc:.2f})')
        ax.grid(True)
    
    def show_pr_curve(y_true, y_pred_proba, model_name, sample_name, ax):
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        auc_pr = auc(recall, precision)

        ax.plot(recall, precision)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'{model_name} ({sample_name}) Precision-Recall Curve (AUC = {auc_pr:.2f})')
        ax.grid(True)

    if y_pred_proba_train is not None and y_pred_proba_test is not None:
        _, axs = plt.subplots(3, 3, figsize=(18, 16))
        
        show_confusion_matrix(y_train, y_pred_train, 'Train', axs[0, 0])
        show_confusion_matrix(y_val, y_pred_val, 'Val', axs[0, 1])
        show_confusion_matrix(y_test, y_pred_test, 'Test', axs[0, 2])

        show_roc_curve(y_train, y_pred_proba_train, title, 'Train', axs[1, 0])
        show_roc_curve(y_val, y_pred_proba_val, title, 'Val', axs[1, 1])
        show_roc_curve(y_test, y_pred_proba_test, title, 'Test', axs[1, 2])

        show_pr_curve(y_train, y_pred_proba_train, title, 'Train', axs[2, 0])
        show_pr_curve(y_val, y_pred_proba_val, title, 'Val', axs[2, 1])
        show_pr_curve(y_test, y_pred_proba_test, title, 'Test', axs[2, 2])
    else:
        _, axs = plt.subplots(1, 3, figsize=(14, 6))
        
        show_confusion_matrix(y_train, y_pred_train, 'Train', axs[0])
        show_confusion_matrix(y_val, y_pred_val, 'Val', axs[1])
        show_confusion_matrix(y_test, y_pred_test, 'Test', axs[2])

    plt.tight_layout()
    plt.show()

def get_bows(X_train, X_val, X_test, ngram_range=(1, 1)):
    vec = CountVectorizer(ngram_range=ngram_range)
    vec.fit(X_train)
    X_train_bow = vec.transform(X_train)
    X_val_bow = vec.transform(X_val)
    X_test_bow = vec.transform(X_test)
    
    scaler = MaxAbsScaler()
    X_train_bow = scaler.fit_transform(X_train_bow)
    X_val_bow = scaler.transform(X_val_bow)
    X_test_bow = scaler.transform(X_test_bow)
    
    return X_train_bow, X_val_bow, X_test_bow

def get_tfidf(X_train, X_val, X_test, ngram_range=(1, 1)):
    vec = TfidfVectorizer(ngram_range=ngram_range, min_df=0.0003, dtype=np.float32)
    X_train_tfidf = vec.fit_transform(X_train)
    X_val_tfidf = vec.transform(X_val)
    X_test_tfidf = vec.transform(X_test)

    scaler = MaxAbsScaler()
    X_train_tfidf = scaler.fit_transform(X_train_tfidf)
    X_val_tfidf = scaler.transform(X_val_tfidf)
    X_test_tfidf = scaler.transform(X_test_tfidf)
    
    return X_train_tfidf, X_val_tfidf, X_test_tfidf

def get_hv(X_train, X_val, X_test, ngram_range=(1, 1)):
    vec = HashingVectorizer(n_features=2**16, alternate_sign=False, norm='l2', ngram_range=ngram_range,
                           dtype=np.float32)
    X_train_hv = vec.fit_transform(X_train)
    X_val_hv = vec.transform(X_val)
    X_test_hv = vec.transform(X_test)

    scaler = MaxAbsScaler()
    X_train_hv = scaler.fit_transform(X_train_hv)
    X_val_hv = scaler.transform(X_val_hv)
    X_test_hv = scaler.transform(X_test_hv)

    return X_train_hv, X_val_hv, X_test_hv

def get_hyperparams(X_train, X_val, y_train, y_val, objective_func, n_trials=50):
    study = optuna.create_study(direction='maximize')
    study.optimize(partial(
        objective_func,
        X_train=X_train, X_val=X_val,
        y_train=y_train, y_val=y_val
    ), n_trials=n_trials)
    
    return study.best_params

def get_threshold(y_true, y_pred_proba):
    metrics_data = pd.DataFrame(columns = ['threshold','precision', 'recall', 'f-1'])
    for i in np.arange(0.01, 1.01, 0.01):
        p = precision_score(y_true, np.where(y_pred_proba > i, 1, 0))
        r = recall_score(y_true, np.where(y_pred_proba > i, 1, 0))
        f = f1_score(y_true, np.where(y_pred_proba > i, 1, 0))
        metrics_data.loc[len(metrics_data)] = [i, p, r, f]
    
    return metrics_data[metrics_data['f-1'] == metrics_data['f-1'].max()].threshold.values[0]

