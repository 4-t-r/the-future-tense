from argparse import ArgumentParser
import pandas as pd
from classification_pipeline import ClassificationPipeline
import numpy as np
import csv
import torch


def get_future_statements(future_model, statements, urls):
    statements_ids, statements_attention = future_model.batch_encode(statements)
    y_pred = future_model.predict_if_future_statement([statements_ids, statements_attention])
    future_statements_idx = np.where(y_pred == 1)[0]

    future_statements = []
    future_urls = []
    for idx in future_statements_idx:
        future_statements.append(statements[idx])
        future_urls.append(urls[idx])

    return future_statements, future_urls


def get_sentiments(sentiment_model, statements, urls):
    sentiments, statements, urls = sentiment_model.predict_sentiment(statements, urls)
    return sentiments, statements, urls


def get_topics(topic_model, statements):
    topics = topic_model.predict_topic(statements)
    return topics


def get_unique_future_statements(statements, urls):
    future_statement_list = []
    future_url_list = []
    for i in range(len(statements)):
        if not statements[i] in future_statement_list:
            future_statement_list.append(statements[i])
            future_url_list.append(urls[i])
    return future_statement_list, future_url_list


def write_header():
    future_statements = open("output/future_statements.csv", 'a+', newline='')

    csv_writer = csv.writer(future_statements, delimiter='|', quotechar='|', quoting=csv.QUOTE_MINIMAL)

    csv_writer.writerow(['statement', 'sentiment', 'topic', 'url'])


def write_statements_to_csv(statements, sentiments, topics, urls):

    future_statements = open("output/future_statements.csv", 'a+', newline='')

    csv_writer = csv.writer(future_statements, delimiter='|', quotechar='|', quoting=csv.QUOTE_MINIMAL)

    for i in range(len(statements)):
        csv_writer.writerow([statements[i], sentiments[i], topics[i], urls[i]])


def main():
    print('Cuda available:',torch.cuda.is_available())

    parser = ArgumentParser(description='A test program.')
    parser.add_argument("-f", "--input_file", help="file containing statements")
    args = parser.parse_args()

    classification_pipeline = ClassificationPipeline()
    cnt = 0

    write_header()

    for chunk in pd.read_csv(args.input_file, on_bad_lines='skip', engine="python", chunksize=30, sep='|'):
        cnt += 1
        print(f'preprocess statement chunk {cnt}')

        statements = chunk['statement'].tolist()
        urls = chunk['url'].tolist()

        statements, urls = get_unique_future_statements(statements, urls)

        print(f'classify future statement')
        statements, urls = get_future_statements(classification_pipeline.get_future_model(), statements, urls)

        if len(statements) > 0:
            print(f'classify sentiment')
            sentiments, statements, urls = get_sentiments(classification_pipeline.get_sentiment_model(), statements, urls)

        if len(statements) > 0:
            print(f'classify topic')
            topics = get_topics(classification_pipeline.get_topic_model(), statements)

            print(f'persist classify statement')
            write_statements_to_csv(statements, sentiments, topics, urls)


if __name__ == "__main__":
    main()
