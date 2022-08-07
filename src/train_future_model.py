#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import scikitplot as skplt
import tensorflow as tf

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from tensorflow import keras
from transformers import DistilBertTokenizerFast
from transformers import TFDistilBertModel, DistilBertConfig

print('Start train_future_model')

pd.plotting.register_matplotlib_converters()
# Don't truncate text
pd.set_option('display.max_colwidth', None)

# Load data as series
X_train = pd.read_csv('../datasets/future_statements_dataset/X_train.csv')["statement"]
y_train = pd.read_csv('../datasets/future_statements_dataset/y_train.csv')["future"]

# Create train/test split
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)

# Create train/validation split
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2)

# Sort index
X_train.sort_index(inplace=True)
X_valid.sort_index(inplace=True)
X_test.sort_index(inplace=True)
y_train.sort_index(inplace=True)
y_valid.sort_index(inplace=True)
y_test.sort_index(inplace=True)

# Reset index
X_train.reset_index(drop=True, inplace=True)
X_valid.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
y_valid.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)

print('Training data:   ', len(X_train.index), ' rows. Negatives:', (y_train==0).sum(), 'Positives:', (y_train==1).sum())
print('Validation data: ', len(X_valid.index), ' rows. Negatives:', (y_valid==0).sum(), 'Positives:', (y_valid==1).sum())
print('Test data:       ', len(X_test.index), ' rows. Negatives:', (y_test==0).sum(), 'Positives:', (y_test==1).sum())

params = {'MAX_LENGTH': 128,
          'EPOCHS': 10,
          #learningrate
          'LEARNING_RATE': 5e-5,
          'FT_EPOCHS': 2,
          'OPTIMIZER': 'adam',
          'FL_GAMMA': 2.0,
          'FL_ALPHA': 0.2,
          'BATCH_SIZE': 64,
          'NUM_STEPS': len(X_train.index) // 64,
          #dropouts:
          'DISTILBERT_DROPOUT': 0.2,
          'DISTILBERT_ATT_DROPOUT': 0.2,
          'LAYER_DROPOUT': 0.2,
          'KERNEL_INITIALIZER': 'GlorotNormal',
          'BIAS_INITIALIZER': 'zeros',
          'POS_PROBA_THRESHOLD': 0.5,
          'ADDED_LAYERS': 'Dense 256, Dense 32, Dropout 0.2',
          'LR_SCHEDULE': '5e-5 for 6 epochs, Fine-tune w/ adam for 2 epochs @2e-5',
          'FREEZING': 'All DistilBERT layers frozen for 6 epochs, then unfrozen for 2',
          'CALLBACKS': '[early_stopping w/ patience=0]',
          'RANDOM_STATE': 42
          }

# Ensure reproducibility
##############################################################################################
os.environ['PYTHONHASHSEED'] = str(params['RANDOM_STATE'])
random.seed(params['RANDOM_STATE'])
np.random.seed(params['RANDOM_STATE'])
tf.random.set_seed(seed=params['RANDOM_STATE'])
##############################################################################################


# Helper functions
##############################################################################################
def batch_encode(_tokenizer, texts, batch_size=256, max_length=params['MAX_LENGTH']):
    """""""""
    A function that encodes a batch of texts and returns the texts'
    corresponding encodings and attention masks that are ready to be fed
    into a pre-trained transformer model.

    Input:
        - _tokenizer:   Tokenizer object from the PreTrainedTokenizer Class
        - texts:       List of strings where each string represents a text
        - batch_size:  Integer controlling number of texts in a batch
        - max_length:  Integer controlling max number of words to tokenize in a given text
    Output:
        - input_ids:       sequence of texts encoded as a tf.Tensor object
        - attention_mask:  the texts' attention mask encoded as a tf.Tensor object
    """""""""

    input_ids = []
    attention_mask = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = _tokenizer.batch_encode_plus(batch,
                                              max_length=max_length,
                                              padding='max_length',
                                              truncation=True,
                                              return_attention_mask=True,
                                              return_token_type_ids=False
                                              )
        input_ids.extend(inputs['input_ids'])
        attention_mask.extend(inputs['attention_mask'])

    return tf.convert_to_tensor(input_ids), tf.convert_to_tensor(attention_mask)


def focal_loss(gamma=params['FL_GAMMA'], alpha=params['FL_ALPHA']):
    """""""""
    Function that computes the focal loss.

    Code adapted from https://gist.github.com/mkocabas/62dcd2f14ad21f3b25eac2d39ec2cc95
    """""""""

    def focal_loss_fixed(y_true, _y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), _y_pred, tf.ones_like(_y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), _y_pred, tf.zeros_like(_y_pred))
        return -keras.backend.mean(alpha * keras.backend.pow(1. - pt_1, gamma) * keras.backend.log(pt_1))\
            - keras.backend.mean((1 - alpha) * keras.backend.pow(pt_0, gamma) * keras.backend.log(1. - pt_0))

    return focal_loss_fixed


def build_model(transformer, max_length=params['MAX_LENGTH']):
    """""""""
    Template for building a model off of the BERT or DistilBERT architecture
    for a binary classification task.

    Input:
      - transformer:  a base Hugging Face transformer model object (BERT or DistilBERT)
                      with no added classification head attached.
      - max_length:   integer controlling the maximum number of encoded tokens
                      in a given sequence.

    Output:
      - model:        a compiled tf.keras.Model with added classification layers
                      on top of the base pre-trained model architecture.
    """""""""

    # Define weight initializer with a random seed to ensure reproducibility
    weight_initializer = tf.keras.initializers.GlorotNormal(seed=params['RANDOM_STATE'])

    # Define input layers
    input_ids_layer = tf.keras.layers.Input(shape=(max_length,),
                                            name='input_ids',
                                            dtype='int32')
    input_attention_layer = tf.keras.layers.Input(shape=(max_length,),
                                                  name='input_attention',
                                                  dtype='int32')

    # DistilBERT outputs a tuple where the first element at index 0
    # represents the hidden-state at the output of the model's last layer.
    # It is a tf.Tensor of shape (batch_size, sequence_length, hidden_size=768).
    
    #last_hidden_state = transformer([input_ids_layer, input_attention_layer])[0]
    last_hidden_state = transformer.distilbert([input_ids_layer, input_attention_layer])[0]

    # We only care about DistilBERT's output for the [CLS] token, which is located
    # at index 0.  Splicing out the [CLS] tokens gives us 2D data.
    cls_token = last_hidden_state[:, 0, :]

    D1 = tf.keras.layers.Dropout(params['LAYER_DROPOUT'],
                                 seed=params['RANDOM_STATE']
                                 )(cls_token)

    X = tf.keras.layers.Dense(256,
                              activation='relu',
                              kernel_initializer=weight_initializer,
                              bias_initializer='zeros'
                              )(D1)

    D2 = tf.keras.layers.Dropout(params['LAYER_DROPOUT'],
                                 seed=params['RANDOM_STATE']
                                 )(X)

    X = tf.keras.layers.Dense(32,
                              activation='relu',
                              kernel_initializer=weight_initializer,
                              bias_initializer='zeros'
                              )(D2)

    D3 = tf.keras.layers.Dropout(params['LAYER_DROPOUT'],
                                 seed=params['RANDOM_STATE']
                                 )(X)

    # Define a single node that makes up the output layer (for binary classification)
    output = tf.keras.layers.Dense(1,
                                   activation='sigmoid',
                                   kernel_initializer=weight_initializer,  # CONSIDER USING CONSTRAINT
                                   bias_initializer='zeros'
                                   )(D3)

    # Define the model
    _model = tf.keras.Model([input_ids_layer, input_attention_layer], output)

    # Compile the model
    _model.compile(tf.keras.optimizers.Adam(lr=params['LEARNING_RATE']),
                   loss=focal_loss(),
                   metrics=['accuracy'])

    return _model
##############################################################################################
# End of helper functions


# Tokenize text
##############################################################################################
# Instantiate DistilBERT tokenizer...we use the Fast version to optimize runtime
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

# Encode X_train
X_train_ids, X_train_attention = batch_encode(tokenizer, X_train.tolist())

# Encode X_valid
X_valid_ids, X_valid_attention = batch_encode(tokenizer, X_valid.tolist())

# Encode X_test
X_test_ids, X_test_attention = batch_encode(tokenizer, X_test.tolist())
##############################################################################################


# The bare, pre-trained DistilBERT transformer model outputting raw hidden-states
# and without any specific head on top.
config = DistilBertConfig(dropout=params['DISTILBERT_DROPOUT'],
                          attention_dropout=params['DISTILBERT_ATT_DROPOUT'],
                          output_hidden_states=True)
distilBERT = TFDistilBertModel.from_pretrained('distilbert-base-uncased', config=config)

# Freeze DistilBERT layers to preserve pre-trained weights
for layer in distilBERT.layers:
    layer.trainable = False

# Build model
model = build_model(distilBERT)


# Train Weights of Added Layers and Classification Head
##############################################################################################
# Define callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  mode='min',
                                                  min_delta=0,
                                                  patience=0,
                                                  restore_best_weights=True)

# Train the model
train_history1 = model.fit(
    x=[X_train_ids, X_train_attention],
    y=y_train.to_numpy(),
    epochs=params['EPOCHS'],
    batch_size=params['BATCH_SIZE'],
    steps_per_epoch=params['NUM_STEPS'],
    validation_data=([X_valid_ids, X_valid_attention], y_valid.to_numpy()),
    callbacks=[early_stopping],
    verbose=2
)
##############################################################################################


# Unfreeze DistilBERT and Fine-tune All Weights
##############################################################################################
# Unfreeze DistilBERT weights to enable fine-tuning
for layer in distilBERT.layers:
    layer.trainable = True

# Lower the learning rate to prevent destruction of pre-trained weights
optimizer = tf.keras.optimizers.Adam(lr=2e-5)

# Recompile model after unfreezing
model.compile(optimizer=optimizer,
              loss=focal_loss(),
              metrics=['accuracy'])

# Train the model
train_history2 = model.fit(
    x=[X_train_ids, X_train_attention],
    y=y_train.to_numpy(),
    epochs=params['FT_EPOCHS'],
    batch_size=params['BATCH_SIZE'],
    steps_per_epoch=params['NUM_STEPS'],
    validation_data=([X_valid_ids, X_valid_attention], y_valid.to_numpy()),
    callbacks=[early_stopping],
    verbose=2
)
##############################################################################################


# Evaluate Model Predictions
##############################################################################################
# Generate predictions
y_pred = model.predict([X_test_ids, X_test_attention])
#print('y_pred',y_pred)
#print('---------------------')
y_pred_thresh = np.where(y_pred >= params['POS_PROBA_THRESHOLD'], 1, 0)
#print('y_pred_thresh',y_pred_thresh)

# Get evaluation results
accuracy = accuracy_score(y_test, y_pred_thresh)
auc_roc = roc_auc_score(y_test, y_pred)

#test_pred = (y_pred != y_test)
#print('test_pred: ',test_pred)

pred_df = pd.DataFrame(zip(y_test, y_pred_thresh, y_pred, X_test), columns=['test', 'pred', 'pred_prob', 'statement'])
print(pred_df)
pred_df.to_csv('test_predict.csv', sep='|')

# Log the ROC curve
fpr, tpr, thresholds = roc_curve(y_test.to_numpy(), y_pred)

print('Accuracy:  ', accuracy)   # 0.9218
print('ROC-AUC:   ', auc_roc)    # 0.9691
##############################################################################################


# Plot Training and Validation Loss
##############################################################################################
# Build train_history
history_df1 = pd.DataFrame(train_history1.history)
history_df2 = pd.DataFrame(train_history2.history)
history_df = history_df1.append(history_df2, ignore_index=True)

# Plot training and validation loss over each epoch
history_df.loc[:, ['loss', 'val_loss']].plot()
plt.title(label='Training + Validation Loss Over Time', fontsize=17, pad=19)
plt.xlabel('Epoch', labelpad=14, fontsize=14)
plt.ylabel('Focal Loss', labelpad=16, fontsize=14)
print("Minimum Validation Loss: {:0.4f}".format(history_df['val_loss'].min()))

# Save figure
plt.savefig('../figures/future_statements_trainvalloss.png', dpi=300.0, transparent=False)
##############################################################################################

# Plot the Confusion Matrix
##############################################################################################
# Plot confusion matrix
skplt.metrics.plot_confusion_matrix(y_test.to_list(),
                                    y_pred_thresh.tolist(),
                                    figsize=(6, 6),
                                    text_fontsize=14)
plt.title(label='Test Confusion Matrix', fontsize=20, pad=17)
plt.xlabel('Predicted Label', labelpad=14)
plt.ylabel('True Label', labelpad=14)

# Save the figure
plt.savefig('../figures/future_statements_confusionmatrix.png', dpi=300.0, transparent=False)
##############################################################################################

# Save model
#tf.saved_model.save(model, '../models/future_statements_model')
#tf.saved_model.save(model, '../models/future_statements_model/future_model.h5')
model.save('../models/future_statements_model/future_model.h5', save_format='h5')
#model.save('../models/future_statements_model/future_model.tf', save_format='tf')