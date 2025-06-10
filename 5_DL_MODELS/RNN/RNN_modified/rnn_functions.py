import copy
import gc
import spacy
from tqdm.notebook import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, auc, average_precision_score, accuracy_score
from collections import Counter
from IPython.display import clear_output
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import Vocab
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer

class ToxicDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

def collate_fn(batch):
    texts, labels = zip(*batch)
    padded = pad_sequence(texts, batch_first=True, padding_value=vocab_obj["<pad>"])
    return padded, torch.tensor(labels)

def plot_losses(train_losses, train_metrics, val_losses, val_metrics):

    clear_output()
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(range(1, len(train_losses) + 1), train_losses, label="train")
    axs[0].plot(range(1, len(val_losses) + 1), val_losses, label="val")
    axs[1].plot(range(1, len(train_metrics) + 1), train_metrics, label="train")
    axs[1].plot(range(1, len(val_metrics) + 1), val_metrics, label="val")

    if max(train_losses) / min(train_losses) > 10:
        axs[0].set_yscale("log")

    if max(train_metrics) / min(train_metrics) > 10:
        axs[0].set_yscale("log")

    for ax in axs:
        ax.set_xlabel("epoch")
        ax.legend()

    axs[0].set_ylabel("loss")
    axs[1].set_ylabel("F-1")
    plt.show()
    

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


def train_and_validate(
    model,
    optimizer,
    criterion,
    metric,
    train_loader,
    val_loader,
    num_epochs,
    verbose=True,
    save_path='best_model.pth',
):
    best_val_loss = float("inf")
    best_model_state = None

    train_losses, val_losses = [], []
    train_metrics, val_metrics = [], []

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss, running_metric = 0, 0
        pbar = tqdm(train_loader, desc=f"Training {epoch}/{num_epochs}") if verbose else train_loader

        for i, (X_batch, y_batch) in enumerate(pbar, 1):
            X = X_batch.to(device)
            y = y_batch.to(device)

            optimizer.zero_grad(set_to_none=True)
            predictions = model(X)
            loss = criterion(predictions, y)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                metric_value = metric(predictions, y)
                metric_value = metric_value.item() if isinstance(metric_value, torch.Tensor) else metric_value
                running_loss += loss.item() * X.shape[0]
                running_metric += metric_value * X.shape[0]

            if verbose and i % 100 == 0:
                pbar.set_postfix({"loss": loss.item(), "F-1": metric_value})

        train_losses.append(running_loss / len(train_loader.dataset))
        train_metrics.append(running_metric / len(train_loader.dataset))

        # Validation
        model.eval()
        running_loss, running_metric = 0, 0
        pbar = tqdm(val_loader, desc=f"Validating {epoch}/{num_epochs}") if verbose else val_loader

        for i, (X_batch, y_batch) in enumerate(pbar, 1):
            with torch.no_grad():
                X = X_batch.to(device)
                y = y_batch.to(device)
                predictions = model(X)
                loss = criterion(predictions, y)

                metric_value = metric(predictions, y)
                metric_value = metric_value.item() if isinstance(metric_value, torch.Tensor) else metric_value
                running_loss += loss.item() * X.shape[0]
                running_metric += metric_value * X.shape[0]

            if verbose and i % 100 == 0:
                pbar.set_postfix({"loss": loss.item(), "F-1": metric_value})

        val_loss = running_loss / len(val_loader.dataset)
        val_metric = running_metric / len(val_loader.dataset)
        val_losses.append(val_loss)
        val_metrics.append(val_metric)

        # Сохраняем по минимальному val_loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(best_model_state, save_path)
            if verbose:
                print(f"📌 Model improved (val_loss = {val_loss:.4f}). Saved to '{save_path}'")

        if verbose:
            plot_losses(train_losses, train_metrics, val_losses, val_metrics)

    if verbose:
        print(f"Best Validation Loss: {best_val_loss:.4f} (model saved to {save_path})")

    return train_metrics[-1], val_metrics[-1]


def get_predictions(model, data_loader, t='Reg', proba = 0.5):
    model.eval()
    predictions = []

    with torch.no_grad():
        for X_batch, _ in data_loader:
            X = X_batch.to(device)
            logits = model(X)
            probs = torch.sigmoid(logits)  # <- Преобразуем логиты в вероятности
            if t == 'Class':
                preds = (probs > proba).int().cpu().numpy()
                predictions.append(preds)
            else:
                predictions.append(probs.cpu())

    if t == 'Class':
        return np.concatenate(predictions)
    else:
        return torch.cat(predictions, dim=0).numpy()

def f1_metric(preds, targets):
    probs = torch.sigmoid(preds)
    preds_cls = (probs > 0.5).int().cpu().numpy()
    targets_cls = targets.int().cpu().numpy()
    return f1_score(targets_cls, preds_cls, average="binary", zero_division=0)


def find_best_t(y_prob, y_true):
    best_threshold = 0
    best_f1 = 0
    for threshold in np.arange(0.01, 1.01, 0.01):
        binary_preds = (y_prob > threshold).astype(int)
        f1 = f1_score(y_true, binary_preds)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print(f"Лучший порог: {best_threshold:.4f}")
    print(f"Лучший F1: {best_f1:.4f}")
    return best_threshold

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