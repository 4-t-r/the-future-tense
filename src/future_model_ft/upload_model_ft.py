# !huggingface-cli login
from huggingface_hub import notebook_login

notebook_login()
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split

from transformers import (
    #AutoTokenizer,
    DistilBertTokenizerFast,
    TFAutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    PushToHubCallback
)
def load_data():
    print(f'Loading data...')
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

    print(f'Loading done...')
    return _X_train.tolist(), _X_valid.tolist(), _X_test.tolist(), \
        _y_train.tolist(), _y_valid.tolist(), _y_test.tolist()

def encode_data(dataset, _tokenizer):
    print(f'Encoding data...')
    inputs = _tokenizer(dataset,
                        max_length=128,
                        padding='max_length',
                        truncation=True,
                        return_token_type_ids=False)
    ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    print(f'Encoding done...')
    return tf.convert_to_tensor(ids), tf.convert_to_tensor(attention_mask)
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
push_cb = [PushToHubCallback("model_output/", 
                               tokenizer=tokenizer,
                               hub_model_id="bert-fine-tuned-cola")]
train_history = model.fit(
    x=[X_train_ids, X_train_attention_mask],
    y=np.asarray(y_train),
    epochs=6,
    batch_size=64,
    steps_per_epoch=len(X_train) // 64,
    validation_data=([X_valid_ids, X_valid_attention_mask],
                     np.asarray(y_valid)),
    callbacks=[early_stopping,
              push_cb
              ],
    verbose=1
)
model.push_to_hub("distilbert-base-future", commit_message="End of training")