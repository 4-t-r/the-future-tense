from satements import list_sq
from add_sentiment import SentimentClassifier

def main():
    sentiment_classifier = SentimentClassifier()
    sentiments = sentiment_classifier.add_sentiment_three_labels(['Turkey fully supports the efforts of those secular republics to build pluralistic societies and will assist them in integrating into the world community.'])
    for i in range(len(sentiments)):
        print(f'{list_sq[i]}: {sentiments[i]}')


if __name__ == "__main__":
    main()