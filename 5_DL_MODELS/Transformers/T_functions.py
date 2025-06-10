import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, auc, average_precision_score, accuracy_score

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EvalPrediction
from sklearn.model_selection import train_test_split
from pandarallel import pandarallel



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

class TextDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.texts = dataframe['comment_text'].values
        self.labels = dataframe['toxicity_b'].values
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def compute_train_metrics(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    probs = torch.nn.functional.softmax(torch.tensor(p.predictions), dim=-1).numpy()[:, 1]
    
    return {
        'roc_auc': roc_auc_score(p.label_ids, probs),
        'pr_auc': average_precision_score(p.label_ids, probs),
        'f1': f1_score(p.label_ids, preds),
        'accuracy': accuracy_score(p.label_ids, preds),
        'precision': precision_score(p.label_ids, preds),
        'recall': recall_score(p.label_ids, preds)
    }

def freeze_layers(model, num_frozen_layers=18):
    for param in model.roberta.embeddings.parameters():
        param.requires_grad = False
    
    for i in range(num_frozen_layers):
        for param in model.roberta.encoder.layer[i].parameters():
            param.requires_grad = False
    
    for param in model.classifier.parameters():
        param.requires_grad = True
        
    return model

def evaluate_checkpoint(path, dataset, dataset_name, tokenizer, compute_metrics):
    model = AutoModelForSequenceClassification.from_pretrained(path)
    trainer = Trainer(model=model, tokenizer=tokenizer, compute_metrics=compute_metrics)
    metrics = trainer.evaluate(eval_dataset=dataset)
    metrics['checkpoint'] = path.split('/')[-1]
    metrics['dataset'] = dataset_name
    return metrics

def plot_dil(md, model_name, metric = 'eval_loss'):
    plt.figure(figsize = (10,7))
    sns.pointplot(data = md[md.dataset == 'train'], x = 'epoch', y = metric, color = 'powderblue', label = 'Train')
    sns.pointplot(data = md[md.dataset == 'val'], x = 'epoch', y = metric, color = 'crimson', label = 'Val')
    plt.title(f'Difference in {metric[5:]} between train and val by epoch for {model_name}', fontsize = 18)
    plt.xlabel('Epoch', fontsize = 14)
    plt.ylabel('Loss', fontsize = 14)
    plt.grid(True, which='both', linestyle='-', linewidth=0.5, color='gray')
    plt.show()

def find_best_t(y_prob, df_true):
    y_t_np = df_true['toxicity_b'].to_numpy()
    best_threshold = 0
    best_f1 = 0
    for threshold in np.arange(0.01, 1.01, 0.01):
        binary_preds = (y_prob > threshold).astype(int)
        f1 = f1_score(y_t_np, binary_preds)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print(f"Лучший порог: {best_threshold:.4f}")
    print(f"Лучший F1: {best_f1:.4f}")
    return best_threshold

url_re = re.compile(r'(https?://\S+|www\.\S+)', re.IGNORECASE)
spaces_re = re.compile(r'\s+')

def clean_text_ft(text):
    text = url_re.sub('', text)
    text = spaces_re.sub(' ', text).strip()
    return text