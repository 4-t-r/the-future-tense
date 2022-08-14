from transformers import pipeline
from pysentimiento import SentimentAnalyzer





class SentimentClassifier:
    def __init__(self):
        '''self.classification_pipeline = pipeline('text-classification',
                                                model='distilbert-base-uncased-finetuned-sst-2-english',
                                                device=-1)'''
        #self.sentiment_analyzer = SentimentAnalyzer(lang = 'en')
        self.model_path = 'cardiffnlp/twitter-roberta-base-sentiment-latest'
        self.sentiment_pipline = pipeline("sentiment-analysis", model=self.model_path, tokenizer=self.model_path)
        self.sentiment_analyzer = SentimentAnalyzer(lang='en')
#sentiment_task("Covid cases are increasing fast!")

    '''def add_sentiment_two_labels(self, statement_list):
        sentiments_list = []
        for i in range(len(statement_list)):
            sentiments_list.append(self.classification_pipeline(statement_list[i])[0])
        return sentiments_list'''

    def add_sentiment_three_labels(self, statement_list):
        #return self.sentiment_pipline(statement_list)
        return self.sentiment_analyzer.predict(statement_list)
        #sentiments_list = []
        #for i in range(len(statement_list)):
        #    sentiments_list.append(self.sentiment_pipline(statement_list[i])[0])
        #return sentiments_list
