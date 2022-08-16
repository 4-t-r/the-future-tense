#!/usr/bin/env python
# coding: utf-8

import pdb
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scikitplot as skplt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
# from sklearn.metrics import roc_curve
from transformers import DistilBertTokenizerFast, \
    TFAutoModelForSequenceClassification


OKBLUE = '\033[94m'
OKGREEN = '\033[92m'
ENDC = '\033[0m'


def load_data():
    print(f'{OKBLUE}Loading data...{ENDC}')
    # Load data as series
    _X_train = \
        pd.read_csv(
            '../../datasets/future_statements_dataset/X_train.csv'
        )['statement']
    _y_train = \
        pd.read_csv(
            '../../datasets/future_statements_dataset/y_train.csv'
        )['future']

    # Create train/test split
    _X_train, _X_test, _y_train, _y_test = \
        train_test_split(_X_train, _y_train, test_size=0.2)

    # Create train/validation split
    _X_train, _X_valid, _y_train, _y_valid = \
        train_test_split(_X_train, _y_train, test_size=0.2)

    # Sort index
    _X_train.sort_index(inplace=True)
    _X_valid.sort_index(inplace=True)
    _X_test.sort_index(inplace=True)
    _y_train.sort_index(inplace=True)
    _y_valid.sort_index(inplace=True)
    _y_test.sort_index(inplace=True)

    # Reset index
    _X_train.reset_index(drop=True, inplace=True)
    _X_valid.reset_index(drop=True, inplace=True)
    _X_test.reset_index(drop=True, inplace=True)
    _y_train.reset_index(drop=True, inplace=True)
    _y_valid.reset_index(drop=True, inplace=True)
    _y_test.reset_index(drop=True, inplace=True)

    print('Training data:   ', len(_X_train.index), ' rows.')
    print('Validation data: ', len(_X_valid.index), ' rows.')
    print('Test data:       ', len(_X_test.index), ' rows.')

    print(f'{OKGREEN}Loading done...{ENDC}')
    return _X_train.tolist(), _X_valid.tolist(), _X_test.tolist(), \
        _y_train.tolist(), _y_valid.tolist(), _y_test.tolist()


def encode_data(dataset, _tokenizer):
    print(f'{OKBLUE}Encoding data...{ENDC}')
    inputs = _tokenizer(dataset,
                        max_length=128,
                        padding='max_length',
                        truncation=True,
                        return_token_type_ids=False)
    ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    print(f'{OKGREEN}Encoding done...{ENDC}')
    return tf.convert_to_tensor(ids), tf.convert_to_tensor(attention_mask)


def evaluate_model(_X_test_ids, _X_test_attention, _y_test, threshold):
    print(f'{OKBLUE}Evaluating model...{ENDC}')
    prediction = model([_X_test_ids, _X_test_attention], training=False)
    logits = prediction['logits']
    _probabilities = tf.nn.softmax(logits)
    #pdb.set_trace()
    _probabilities = _probabilities[:, 1]
    _y_pred_thresh = np.where(_probabilities >= threshold, 1, 0)
    accuracy = accuracy_score(_y_test, _y_pred_thresh)
    auc_roc = roc_auc_score(_y_test, _probabilities)
    # fpr, tpr, thresholds = roc_curve(_y_test.to_numpy(), _y_pred)
    print('Accuracy:  ', accuracy)
    print('ROC-AUC:   ', auc_roc)
    print(f'{OKGREEN}Evaluating done...{ENDC}')
    return _probabilities, _y_pred_thresh


def plot_training_and_val_loss(_train_history):
    _train_history_df = pd.DataFrame(_train_history.history)
    _train_history_df.loc[:, ['loss', 'val_loss']].plot()
    plt.title(label='Training + Validation Loss Over Time',
              fontsize=17,
              pad=19)
    plt.xlabel('Epoch', labelpad=14, fontsize=14)
    plt.ylabel('Focal Loss', labelpad=16, fontsize=14)
    print('Minimum Validation Loss: {:0.4f}'
          .format(_train_history_df['val_loss'].min()))
    plt.savefig('../../figures/future_statements_trainvalloss.png',
                dpi=300.0,
                transparent=False)


def plot_confusion_matrix(_y_test, _y_pred_thresh):
    skplt.metrics.plot_confusion_matrix(_y_test,
                                        _y_pred_thresh.tolist(),
                                        figsize=(6, 6),
                                        text_fontsize=14)
    plt.title(label='Test Confusion Matrix', fontsize=20, pad=17)
    plt.xlabel('Predicted Label', labelpad=14)
    plt.ylabel('True Label', labelpad=14)
    plt.savefig('../../figures/future_statements_confusionmatrix.png',
                dpi=300.0,
                transparent=False)


if __name__ == '__main__':
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_data()

    model = TFAutoModelForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=2
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=tf.metrics.SparseCategoricalAccuracy(),
    )
    tokenizer = DistilBertTokenizerFast.from_pretrained(
        'distilbert-base-uncased'
    )

    # Get train_ids and attention_mask
    X_train_ids, X_train_attention_mask = encode_data(X_train, tokenizer)
    X_valid_ids, X_valid_attention_mask = encode_data(X_valid, tokenizer)
    X_test_ids, X_test_attention_mask = encode_data(X_test, tokenizer)

    # Callback
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      mode='min',
                                                      min_delta=0,
                                                      patience=0,
                                                      restore_best_weights=True)

    print(f'{OKBLUE}Training model...{ENDC}')
    train_history = model.fit(
        x=[X_train_ids, X_train_attention_mask],
        y=np.asarray(y_train),
        epochs=6,
        batch_size=64,
        steps_per_epoch=len(X_train) // 64,
        validation_data=([X_valid_ids, X_valid_attention_mask],
                         np.asarray(y_valid)),
        callbacks=[early_stopping],
        verbose=1
    )
    print(f'{OKGREEN}Training done...{ENDC}')

    plot_training_and_val_loss(train_history)

    _, y_pred_thresh = evaluate_model(X_test_ids,
                                      X_test_attention_mask,
                                      y_test,
                                      0.5)

    plot_confusion_matrix(y_test, y_pred_thresh)

    model.save_pretrained('../../models/ftr_mdl')
