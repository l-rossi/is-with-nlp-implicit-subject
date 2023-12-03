import os

import spacy

from candidate_extraction.CandidateExtractor import CandidateExtractor
from candidate_ranking.PerplexityRanker import PerplexityRanker
from insertion.ImplicitSubjectInserter import ImplicitSubjectInserter
from missing_subject_detection.PassiveDetector import PassiveDetector
from util import get_noun_chunk


def main():
    # tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")

    # mask_filler = pipeline("fill-mask", "distilroberta-base")

    ctx = """
    The various declaration documents all follow a similar process 
    flow. After submission by the employee, the request is sent for
    approval to the travel administration. If approved by the person, the request is
    then forwarded to the budget owner and after that to the supervisor.
    If the budget owner and supervisor are the same person, then only one 
    of the these steps is taken. In some cases, the director also needs to approve
    the request
    """

    nlp = spacy.load("en_core_web_trf")
    doc = nlp(ctx)

    targets = PassiveDetector().detect(doc)
    print("target", targets)

    candidates = CandidateExtractor().extract(doc)

    ranker = PerplexityRanker()


    subjs = [
        get_noun_chunk(ranker.rank(target.predicate, candidates)[0]) for target in targets
    ]

    print(subjs)

    resolved = ImplicitSubjectInserter().insert(doc, targets, subjs)
    print(resolved)

    exit()

    nlp = spacy.load("en_core_web_trf")
    # nlp = spacy.load("en_core_web_sm")

    # data_dir = "./data/external/synthetic"
    data_dir = "./data/external/synthetic"

    chunks = []
    for file_name in os.listdir(data_dir):
        with open(data_dir + "/" + file_name, "r", encoding="utf-8") as f:
            chunks.append(f.read())

    # with open(data_dir + "/BPI_2020_Challenge.txt", "r", encoding="utf-8") as f:
    #     text = f.read()

    text = """
          The setup of your account starts with Blizzard checking whether you have a battle.net account.
           """.strip()

    # doc = nlp(text)
    # displacy.serve(doc, "dep")

    for chunk in chunks[:1]:
        print(chunk)
        doc = nlp(chunk)

        # print(ImperativeDetector().detect(doc))
        print(CandidateExtractor().extract(doc))

    # passive: VBN without pobj or more strictly even without a by as a pobj might be present, e.g., sent for approval
    # Gerund:  (checking, 'pcomp', 'VBG')
    # Imperative: VB? but that is base form


if __name__ == "__main__":
    main()
