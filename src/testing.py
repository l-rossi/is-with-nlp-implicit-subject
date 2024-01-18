import spacy
from dotenv import load_dotenv
from nltk.stem import PorterStemmer
from spacy import displacy

from insertion.pattern.inflect import lexeme
from missing_subject_detection.GerundDetector import GerundDetector
from missing_subject_detection.ImperativeDetector import ImperativeDetector
from missing_subject_detection.NominalizedGerundWordlistDetector import NominalizedGerundWordlistDetector
from missing_subject_detection.PassiveDetector import PassiveDetector
from util import load_gold_standard

load_dotenv()


def main():
    ctx = """
    """
    nlp = spacy.load("en_core_web_trf")

    """detectors = [PassiveDetector(), ImperativeDetector(), GerundDetector(), NominalizedGerundWordlistDetector()]

    for i, (source, target, gs, _, _) in enumerate(list(load_gold_standard())[:]):

        targets = dict()
        for detector in reversed(detectors):
            # (ImplicitSubjectDetection are not hashable, so we use the hashable predicate token as a unique key)
            # Detectors at the front of the list take precedence over those at the back.
            targets.update({
                x.token: x for x in detector.detect(nlp(target)[:])
            })
        targets = list(targets.values())
        print([x.token for x in targets])"""


    # Omitting the verb from the sentence is also possible. Once omitted, it is no longer present.

    txt = """
   If you do not go to the service, you are fined by the police after 30 days.
   The data is kept up to date by me for 30 days.
    """

    """
    As soon as an offer is accepted, all other offers become invalid.
    A job interview can be negotiated.
    Several requests can be submitted independently of each other.
    """

    doc = nlp(txt)

    print(NominalizedGerundWordlistDetector().detect(doc[:]))

    # print(PassiveDetector().detect(doc[:]))
    # print(GerundDetector().detect(doc[:]))
    # print(NominalizedGerundWordlistDetector().detect(doc[:]))

    # print(CandidateExtractorImpl().extract(doc))

    # print(doc.ents)

    stemmer = PorterStemmer()

    print(stemmer.stem("withdrawal"))

    similarity_nlp = spacy.load("en_core_web_lg")
    t1 = "The setup of your account starts with Blizzard checking whether you have a battle.net account."
    t2 = "The setup of your account starts with you checking whether you have a battle.net account."

    print(similarity_nlp(t1).similarity(similarity_nlp(t2)))

    for tok in doc:
        print(tok.text, tok.dep_, tok.tag_, tok.lemma_, tok.pos_)

    print(lexeme("be"))

    displacy.serve(doc, style="dep")


if __name__ == "__main__":
    main()
