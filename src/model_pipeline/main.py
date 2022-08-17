from transformers import AutoModel

def main():
    model = AutoModel.from_pretrained("fidsinn/distilbert-base-future")


if __name__ == "__main__":
    main()