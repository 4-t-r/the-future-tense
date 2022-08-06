from transformers import pipeline


if __name__ == '__main__':
    p = pipeline('text-classification', model='distilbert-base-uncased-finetuned-sst-2-english', device = -1)
    outputs = p('AI will be a risk for many workers')
    print(outputs[0])


# See PyCharm help at https://www.jetbrains.com/help/pycharm/

