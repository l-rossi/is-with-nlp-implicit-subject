import os

import spacy


def main():

    nlp = spacy.load("en_core_web_trf")

    data_dir = "./data/external/synthetic"
    for file_name in os.listdir(data_dir):
        with open(data_dir + "/" + file_name, "r", encoding="utf-8") as f:
            text = f.read()

        doc = nlp(text)

        for sent in doc.sents:
            for tok in sent:
                pass



    pass


if __name__ == "__main__":
    main()
