from Classifier.future_classifier import FutureClassifier
from Classifier.sentiment_classifier import SentimentClassifier
from Classifier.topic_classifier import TopicClassifier


class ClassificationPipeline:

    def __init__(self):
        self.future_model = FutureClassifier()
        self.sentiment_model = SentimentClassifier()
        self.topic_model = TopicClassifier()

    def get_future_model(self):
        return self.future_model

    def get_sentiment_model(self):
        return self.sentiment_model

    def get_topic_model(self):
        return self.topic_model
