#from satements import list_sq, list_neg
from add_sentiment import SentimentClassifier
import csv
import os
from sklearn.metrics import confusion_matrix as cm
import numpy as np

test_set_path = os.path.abspath("sentiment_dataset/sentiment_dataset.csv")
test_set_label_path = os.path.abspath("sentiment_dataset/labels.csv")

sentiment_dict = {
    'NEG': 0,
    'NEU': 1,
    'POS': 2
}


def get_sentiment_classification(statements, sentiment_classifier):
    sentiments = sentiment_classifier.add_sentiment(statements)
    return sentiments


def get_sentiment_class_mapping(sentiments):
    sentiment_class = []
    probabilities = []
    for sentiment in sentiments:
        sentiment_class.append(sentiment_dict[sentiment.output])
        probabilities.append(sentiment.probas[sentiment.output])
    return sentiment_class, probabilities


def label_chunks(statement_chunks, sentiment_classifier):
    y_pred = []
    probas = []
    for chunk in statement_chunks:
        chunk_sentiments = get_sentiment_classification(chunk, sentiment_classifier)
        chunk_sentiments_class, probabilities = get_sentiment_class_mapping(chunk_sentiments)
        y_pred = y_pred + chunk_sentiments_class
        probas = probas + probabilities
        print(y_pred)
    return y_pred, probas




def split_statements_in_chunks(statements, chunksize = 32):
    return [statements[i:i + chunksize] for i in range(0, len(statements), chunksize)]


def get_labeled_test_data():
    statements = list(csv.reader(open(test_set_path, encoding="utf8"), delimiter="|"))
    statements = [x[0] for x in statements]
    labels = list(csv.reader(open(test_set_label_path, encoding="utf8" )))
    labels = [int(x[0]) for x in labels]
    return statements, labels


def get_confusion_matrix(y_true, y_pred):
    #y_true = list(map(int, y_true))
    print(cm(np.asarray(y_true), np.asarray(y_pred)))


def save_wrong_classified(y_true, y_pred, statements, probas):
    negative_by_model = open("negative_by_model.csv", 'w+', newline='')
    neutral_by_model = open("neutral_by_model.csv", 'w+', newline='')
    positive_by_model = open("positive_by_model.csv", 'w+', newline='')

    csv_writer_neg = csv.writer(negative_by_model, delimiter='|',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    csv_writer_neu = csv.writer(neutral_by_model, delimiter='|',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
    csv_writer_pos = csv.writer(positive_by_model, delimiter='|',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)

    for i in range(len(y_true)):
        if y_true[i] != y_pred[i]:
            if y_pred[i] == 0:
                csv_writer_neg.writerow([statements[i], y_true[i], probas[i]])
            elif y_pred[i] == 1:
                csv_writer_neu.writerow([statements[i], y_true[i], probas[i]])
            elif y_pred[i] == 2:
                csv_writer_pos.writerow([statements[i], y_true[i], probas[i]])




def main():
    sentiment_classifier = SentimentClassifier()
    statements, y_true = get_labeled_test_data()
    statement_chunks = split_statements_in_chunks(statements)
    y_pred, probas = label_chunks(statement_chunks, sentiment_classifier)
    get_confusion_matrix(y_true, y_pred)
    save_wrong_classified(y_true, y_pred, statements, probas)






if __name__ == "__main__":
    main()