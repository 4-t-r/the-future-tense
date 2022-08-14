from satements import list_sq, list_neg
from add_sentiment import SentimentClassifier
import csv

def main():
    sentiment_classifier = SentimentClassifier()
    sentiments = sentiment_classifier.add_sentiment_three_labels(list_neg
        )
    #sentiment = sentiments[0].output
    for i in range(len(sentiments)):
        print(f'{list_neg[i]}| {sentiments[i]}')
        #print(f'{sentiments[i]}')


if __name__ == "__main__":
    main()