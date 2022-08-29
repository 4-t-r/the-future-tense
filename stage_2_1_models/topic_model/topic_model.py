import os
import transformers
import pandas as pd
import torch
from huggingface_hub import notebook_login
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel

list_sq = ["AI will be led by companies like Tesla.",
            "AI could be a risk for many workers.",
            "I am sure that social networks like facebook should use AI to counteract hatespeech comments.",
            "In the future, cars will be powered by alternative energy sources such as solar and wind power.",
            "In the future, there will be driverless cars.",
            "In the future, flying cars will be a common form of transport.",
            "In the future, hoverboards will be a common form of transport.",
            "In the future, the use of public transport will increase as it becomes more efficient and affordable.",
            "In the future, the number of people working from home will increase as technology advances.",
            "In the future, there will be more jobs in the service sector as automation increases.",
            "In the future, there will be more jobs in the green economy as we move away from fossil fuels.",
            "In the future, artificial intelligence will lead to the creation of new jobs in fields such as healthcare and finance.",
            "In the future, social media will be used more for business networking and less for personal use.",
            "In the future, there will be more regulation of social media to protect people's privacy.",
            "In the future, virtual reality will be used more for social interaction and entertainment.",
            "In the future, augmented reality will be used"
            ]

## Topics

#topic classifier
classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")
#topic classifier - labeling
candidate_labels = ['Labour market', 'transport', 'social media']
cl = classifier(list_sq, candidate_labels, multi_class=False)

for i in range(len(cl)):
    for k in cl[i]:
        print(cl[i][k])

sequences_list = []
topics_list = []
scores_topics_list = []
for i in range(len(cl)):
    sequences_list.append(cl[i]['sequence'])
    topics_list.append(cl[i]['labels'][0])
    scores_topics_list.append(cl[i]['scores'][0])
#print(sequences_list)
#print(topics_list)
#print(scores_topics_list)
df = pd.DataFrame(data={'sequence':sequences_list, 'labels':topics_list, 'scores':scores_topics_list})

df