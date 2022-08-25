import os
import time
import pandas as pd

import transformers
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel

t_minus_one = time.time()
print('topic model starts...')

path = '../../datasets/test_dataset_model_pipeline/collected_statements_v0.5.csv'
df = pd.read_csv(path, sep='|', error_bad_lines=False)

## Topics
#set topic label
candidate_label = ['Robot Design', 'Autonomous Robotics', 'Forex trading']

#topic classifier
classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")

### Sample (only for exploration - delete in production)
df = df.sample(n=200).reset_index()

### Run topic classification

#topic classifier - labeling
def topics_classification(row, t0):
    cl = classifier(row['statement'], candidate_label, multi_label=False)
    cl = cl['labels'][0]
    if int(row.name) % 10 == 0:
         print(row.name, '/', df.shape[0], 'rows labeled in', int(round((time.time()-t0), 0)), 'seconds.')
    return cl

t0 = time.time()
df['topic'] = df.apply(lambda row: topics_classification(row, t0), axis=1)

# get distribution of topics
df.groupby('topic').size()

# chop substring at the end of string
def rchop(s, suffix):
    if suffix and s.endswith(suffix):
        return s[:-len(suffix)]
    return s

path = rchop(path, '.csv')
path_plus_topic = path + 'topic_labeled' + '.csv'
df.to_csv(path_plus_topic, sep='|')

print('...topic model finished in {} seconds.'.format(int(round((time.time()-t_minus_one), 0))))