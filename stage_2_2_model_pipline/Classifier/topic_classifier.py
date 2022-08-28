import transformers
import pandas as pd
import torch
from huggingface_hub import notebook_login
from transformers import pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os


class TopicClassifier:

    def __init__(self):
        self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        self.candidate_labels = ['machine human interface', 'finance', 'social media', 'search engine', 'computer vision',
                                 'natural language technologiy', 'gaming', 'transhumanism', 'research computing']

    def predict_topic(self, statement_list):
        #return self.classifier(statement_list, self.candidate_labels, multi_class=False)
        classification = self.classifier(statement_list, self.candidate_labels, multi_class=False)
        topics_list = self.get_topics_from_classification(classification)
        return topics_list

    def get_topics_from_classification(self, classification):
        topics_list = []
        for i in range(len(classification)):
            max_idx = classification[i]['scores'].index(max(classification[i]['scores']))
            topics_list.append(classification[i]['labels'][max_idx])
        return topics_list
