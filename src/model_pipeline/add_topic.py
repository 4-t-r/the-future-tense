import transformers
import pandas as pd
import torch
from huggingface_hub import notebook_login
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
import os


class TopicClassifier:

    def __init__(self):
        self.pipeline = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        self.candidate_labels = ['Labour market', 'transport', 'social media']

    def add_topic(self, statement_list):
        return self.pipeline(statement_list, self.candidate_labels, multi_class=False)
