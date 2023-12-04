import spacy
from spacy import displacy

from missing_subject_detection.PassiveDetector import PassiveDetector


def main():
    ctx = """
    The various declaration documents all follow a similar process 
    flow. After submission by the employee, the request is sent for
    approval to the travel administration. If approved by the person, the request is
    then forwarded to the budget owner and after that to the supervisor.
    If the budget owner and supervisor are the same person, then only one 
    of the these steps is taken. In some cases, the director also needs to approve
    the request. Select an appropriate component.
    """

    txt = """
    Information provided under Articles 13 and 14 and any communication and any actions taken under Articles 15 to 22 and 34 shall be provided by the controller free of charge.
    """

    """
    As soon as an offer is accepted, all other offers become invalid.
    A job interview can be negotiated.
    Several requests can be submitted independently of each other.
    """

    nlp = spacy.load("en_core_web_trf")
    doc = nlp(txt)

    targets = PassiveDetector().detect(doc[:])

    for target in targets:
        subj = "by the subject"

        *_, insertion_point = (x for x in target.predicate.subtree if x.dep_ != "punct")

        list_tokens = list(token.text_with_ws for token in doc)
        print(insertion_point)
        print(insertion_point.text_with_ws)
        list_tokens[insertion_point.i] = insertion_point.text_with_ws + " " + subj
        resolved_text = "".join(list_tokens)
        print(resolved_text)
        print("---")

    displacy.serve(doc, style="dep")


if __name__ == "__main__":
    main()
