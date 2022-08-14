#from satements import list_sq, list_neg
from add_sentiment import SentimentClassifier
import csv
import os

test_set_path = os.path.abspath("../../datasets/sentiment_dataset/sentiment_dataset.csv")
test_set_label_path = os.path.abspath("../../datasets/sentiment_dataset/labels.csv")

sentiment_dict = {
    'NEG': 0,
    'NEU': 1,
    'POS': 2
}


def get_sentiment_classification(statements):
    sentiment_classifier = SentimentClassifier()
    sentiments = sentiment_classifier.add_sentiment_three_labels(statements)


def get_sentiment_class_mapping(sentiments):
    sentiment_class = []
    for sentiment in sentiments:
        sentiment_class.append(sentiment_dict[sentiment.output])


def split_statements_in_chunks(statements, chunksize = 30):
    return [statements[i:i + chunksize] for i in range(0, len(statements), chunksize)]


def get_labeled_test_data():
    statements = list(csv.reader(open(test_set_path, encoding="utf8")))
    labels = list(csv.reader(open(test_set_label_path, encoding="utf8" )))
    return statements, labels


def main():
    statements, y_true = get_labeled_test_data()
    statement_chunks = split_statements_in_chunks(statements)






if __name__ == "__main__":
    main()