from pysentimiento import SentimentAnalyzer


class SentimentClassifier:

    def __init__(self):
        self.sentiment_analyzer = SentimentAnalyzer(lang='en')

    def predict_sentiment(self, statement_list, urls):
        predictions = self.sentiment_analyzer.predict(statement_list)
        sentiments = [sentiment.output for sentiment in predictions]
        probabilities = [sentiment.probas[sentiment.output] for sentiment in predictions]
        sentiments, statements, urls = self.filter_predictions(sentiments, probabilities, statement_list, urls)
        return sentiments, statements, urls

    def filter_predictions(self, sentiments, probabilities, statement_list, urls):
        validated_sentiments = []
        validated_statements = []
        validated_urls = []
        for i in range(len(probabilities)):
            if probabilities[i] > 0.7:
                validated_sentiments.append(sentiments[i])
                validated_statements.append(statement_list[i])
                validated_urls.append(urls[i])
        return validated_sentiments, validated_statements, validated_urls
