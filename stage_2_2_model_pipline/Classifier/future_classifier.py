import tensorflow as tf
import numpy as np
from transformers import DistilBertTokenizerFast, AutoTokenizer
from transformers import TFDistilBertModel, DistilBertConfig, AutoModelForSequenceClassification, \
    TFDistilBertForSequenceClassification, TFAutoModelForSequenceClassification
import os


class FutureClassifier:
    """
        class containing a model, that predicts if a given statement is a future statement or not.
    """

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("fidsinn/distilbert-base-future")
        #self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        #print(os.path.abspath('../../models/future_statements_model/checkpoints/my_checkpoint'))
        #self.model_weights_path = '../../models/future_statements_model/checkpoints/my_checkpoint'
        self.model = TFAutoModelForSequenceClassification.from_pretrained("fidsinn/distilbert-base-future")
        #self.model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
        #self.model.load_weights(self.model_weights_path)

    def tokenize_statement(self, statements):
        """
            text corresponds to the statement, given for tokenization
        """
        statement_tokenized = self.tokenizer.encode(statements,
                                                    truncation=True,
                                                    padding=True,
                                                    return_tensors="tf")
        return statement_tokenized

    def batch_encode(self, texts, batch_size=128, max_length=256):
        """""""""
        A function that encodes a batch of texts and returns the texts'
        corresponding encodings and attention masks that are ready to be fed
        into a pre-trained transformer model.
        Input:
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
            inputs = self.tokenizer.batch_encode_plus(batch,
                                                      max_length=max_length,
                                                      padding='max_length',
                                                      truncation=True,
                                                      return_attention_mask=True,
                                                      return_token_type_ids=False
                                                      )
            input_ids.extend(inputs['input_ids'])
            attention_mask.extend(inputs['attention_mask'])

        return tf.convert_to_tensor(input_ids), tf.convert_to_tensor(attention_mask)

    def load_model(self):
        """
            load finetuned future model
        """

        loaded_model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
        loaded_model.load_weights(self.model_weights_path).expect_partial()

        return loaded_model

    def predict_if_future_statement(self, statements_tokenized):
        """
            returns the probability if the tokenized statement is a future statement.
        """
        predictions = self.model.predict(statements_tokenized, verbose=1)
        prediction_logits = predictions[0]
        prediction_probs = tf.nn.softmax(prediction_logits, axis=1).numpy()
        prediction_probs = prediction_probs[:, 1]
        y_pred = np.where(prediction_probs >= 0.5, 1, 0)

        return y_pred
