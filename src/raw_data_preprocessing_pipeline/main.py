from satements import list_sq, list_neg
from add_sentiment import SentimentClassifier
import csv
import os

test_set_path = os.path.abspath("../../datasets/sentiment_dataset/sentiment_dataset.csv")
sentiment_dict = {
    'NEG': 0,
    'NEU': 1,
    'POS': 2
}




def get_sentiment_class_mapping(sentiments):
    sentiment_class = []
    for sentiment in sentiments:
        sentiment_class.append(sentiment_dict[sentiment.output])


def main():
    statements = list(csv.reader(open(test_set_path, encoding="utf8")))
    sentiment_classifier = SentimentClassifier()
    sentiments = sentiment_classifier.add_sentiment_three_labels(statements)





if __name__ == "__main__":
    main()