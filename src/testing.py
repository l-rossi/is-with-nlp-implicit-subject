import spacy
from spacy import displacy

from insertion.pattern.inflect import verbs


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

    # Omitting the verb from the sentence is also possible. Once omitted, it is no longer present.

    txt = """
     Depending on the results, additional questions are asked by you (or not).
    """


    """
    As soon as an offer is accepted, all other offers become invalid.
    A job interview can be negotiated.
    Several requests can be submitted independently of each other.
    """

    nlp = spacy.load("en_core_web_trf")
    doc = nlp(txt)

    print(verbs.find_lexeme("use"))

    # print(PassiveDetector().detect(doc[:]))
    # print(GerundDetector().detect(doc[:]))
    # print(NominalizedGerundWordlistDetector().detect(doc[:]))

    # print(CandidateExtractorImpl().extract(doc))

    # print(doc.ents)

    similarity_nlp = spacy.load("en_core_web_lg")
    t1 = "The setup of your account starts with Blizzard checking whether you have a battle.net account."
    t2 = "The setup of your account starts with you checking whether you have a battle.net account."

    print(similarity_nlp(t1).similarity(similarity_nlp(t2)))

    for tok in doc:
        print(tok.text, tok.dep_, tok.tag_, tok.lemma_)

    displacy.serve(doc, style="dep")


if __name__ == "__main__":
    main()
