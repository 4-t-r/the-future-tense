from transformers import pipeline
from pysentimiento import SentimentAnalyzer


class SentimentClassifier:

    def __init__(self):
        self.model_path = 'cardiffnlp/twitter-roberta-base-sentiment-latest'
        self.sentiment_pipline = pipeline("sentiment-analysis", model=self.model_path, tokenizer=self.model_path)
        self.sentiment_analyzer = SentimentAnalyzer(lang='en')

    def add_sentiment(self, statement_list):
        return self.sentiment_analyzer.predict(statement_list)
