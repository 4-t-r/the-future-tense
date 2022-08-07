import tensorflow as tf
import numpy as np
#from tensorflow import keras
import keras.models
from transformers import DistilBertTokenizerFast, AutoTokenizer
from transformers import TFDistilBertModel, DistilBertConfig, AutoModelForSequenceClassification, TFDistilBertForSequenceClassification
import os
#from keras import load_model

def tokenize_statement(statement):
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    #tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    
    #other model-work
    #tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-mnli')
    #tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')

    statement_tokenized = tokenizer.encode(statement,
                                            truncation=True,
                                            padding=True,
                                            return_tensors="tf")
    return statement_tokenized

'''
load the already finetuned model
'''
def load_model():
    #path = os.path.abspath('../models/future_statements_model/saved_model.pb')
    
    #loaded_model.load_weights('./distillbert_tf.h5')
    #loaded_model.load_weights('../models/future_statements_model/future_model.h5')

    loaded_model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
    loaded_model.load_weights('./topic_model/checkpoints/my_checkpoint')
    #loaded_model.load_weights('../models/future_statements_model/future_model.h5')
    #loaded_model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased')

    #other model-work
    #loaded_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')
    #loaded_model = AutoModelForSequenceClassification.from_pretrained('facebook/bart-large-mnli')
    #loaded_model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')

    return loaded_model

'''
predict a statement
'''
def pred_model(model, statement_tokenized):
    prediction = model(statement_tokenized)
    #pred = model.predict(statement)

    return prediction

if __name__ == "__main__":
    tf.config.run_functions_eagerly(True)
    #np_config.enable_numpy_behavior()
    statement = 'We are going to be well in our car.'
    #statement = 'We are in our car.'
    #statement = ['We were in our car.', 'Water levels will rise.']
    tokenized_statement = tokenize_statement(statement)
    model = load_model()
    pred = pred_model(model, tokenized_statement)
    prediction_logits = pred[0]
    prediction_probs = tf.nn.softmax(prediction_logits,axis=1).numpy()
    print('----')
    print(f'Prediction logits: {prediction_logits}')
    print(f'Prediction probs: {prediction_probs}')
    if prediction_probs[0][1] >= 0.5:
        print('\"',statement,'\"','is future statement!')
    else:
        print('\"',statement,'\"','is not future statement!')
    #print(pred_model(model, statement))