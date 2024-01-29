import spacy
from dotenv import load_dotenv
from nltk.stem import PorterStemmer
from spacy import displacy

from insertion.pattern.inflect import lexeme
from missing_subject_detection.NominalizedGerundWordlistDetector import NominalizedGerundWordlistDetector

load_dotenv()


def main():
    """
    This is just a file for messing around mostly with the dependency parser.
    Nothing of value can be found here.
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
Category: Vending and Shipping Title: Ordering in an Online Shop  A customer that logs into an E-Shop system and has to wait for its login confirmation (its credentials are checked by the system). If the login was successful then the customer can continue to select products, else the shopping experience stops. After selecting a product the customer has to add it to a shopping cart, save the product and check if every product was already selected. These steps repeat until all products were selected. Then the order is finished by the shopping system and simultaneously payment and shipment for the order is prepared. For the payment the customer has to enter its payment data and has to wait until the bank confirms the payment. While waiting for the payment confirmation the customer can enter its shipping address (an independent billing address can be entered if the shipping address is not equal to the billing address) . Finally if the address and the payment steps are executed then the order will be finished by the system. 
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

    print(stemmer.stem("eating"))

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
