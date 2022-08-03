import tensorflow as tf
#from tensorflow import keras
import keras.models
import os
#from keras import load_model

'''
load the already finetuned model
'''
def load_model():
    #path = os.path.abspath('../models/future_statements_model/saved_model.pb')
    
    #model = tf.saved_model.load(path)
    #model = load_model('../models/future_statements_model/saved_model.pb')
    model = keras.models.load_model('../models/future_statements_model/future_model.h5')

    return model

'''
predict a statement
'''
def pred_model(model, statement):
    pred = model.predict(statement)

    return pred

statement = 'We are going to be well.'
model = load_model()
print(pred_model(model, statement))