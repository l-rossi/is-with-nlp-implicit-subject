import spacy


def main():
    # nlp = spacy.load("en_core_web_trf")
    nlp = spacy.load("en_core_web_sm")

    data_dir = "./data/external/synthetic"

    # for file_name in os.listdir(data_dir):
    #     print(file_name)
    #     with open(data_dir + "/" + file_name, "r", encoding="utf-8") as f:

    # with open(data_dir + "/BPI_2020_Challenge.txt", "r", encoding="utf-8") as f:
    #     text = f.read()

    text = """
           As soon as you have an account, log into it.
           """.strip()


    doc = nlp(text)

    print([(tok, tok.dep_, tok.tag_) for sent in doc.sents for tok in sent])



    # passive: VBN without pobj or more strictly even without a by as a pobj might be present, e.g., sent for approval
    # Gerund:  (checking, 'pcomp', 'VBG')
    # Imperative: VB? but that is base form

if __name__ == "__main__":
    main()
