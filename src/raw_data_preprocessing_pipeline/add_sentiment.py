
class SentimentClassifier:
    def __init__(self):
        self.classification_pipeline = pipeline('text-classification',
                                                model='distilbert-base-uncased-finetuned-sst-2-english',
                                                device=-1)

    def add_sentiment(self, text):
        outputs = self.classification_pipeline(text)
        return outputs[0]