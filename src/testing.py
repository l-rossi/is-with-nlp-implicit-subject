import spacy
from spacy import displacy

from missing_subject_detection.PassiveDetector import PassiveDetector


def main():
    txt = """
    Based on your job applications, new potential job offers are sent to you.
    """.strip()

    """
    As soon as an offer is accepted, all other offers become invalid.
    A job interview can be negotiated.
    Several requests can be submitted independently of each other.
    """

    nlp = spacy.load("en_core_web_trf")
    doc = nlp(txt)

    targets = PassiveDetector().detect(doc)

    for target in targets:
        subj = "by the subject"

        # Insert
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
