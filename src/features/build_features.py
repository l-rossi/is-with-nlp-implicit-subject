import uuid
from typing import List

import spacy
from spacy import Language
from spacy.tokens import Token, Doc

from data.document_parsing.node.node import Node
from data.document_parsing.node.node_traversal import pre_order
from features.phrase_extraction.kg_renderer import nlp_doc
from features.phrase_extraction.sentence_analysing.phrase_extractor import PhraseExtractor
from features.reference_detection.regex_reference_detector import RegexReferenceDetector
from features.reference_resolution.reference_resolver import ReferenceResolver
from data.document_parsing.node.article import Article
from util.parser_util import gdpr_dependency_root
from util.spacy_components import REFERENCE_QUALIFIER_RESOLVER_COMPONENT

import coreferee


# Stolen without shame from https://github.com/CatherineSai/compliance_textual_constraints/blob/master/src/NEW_preprocessing_optionA_reg.ipynb
# def apply_coreference_resolution(text):
#     doc = nlp(text)
#     print(doc._.coref_chains)
#
#     # split text in tokens
#     list_tokens = list(token.text_with_ws for token in doc)
#     for index, token in enumerate(list_tokens):
#         # check if token an identified coreference token
#         resolved = doc._.coref_chains.resolve(doc[index])
#         print(token, resolved)
#         if resolved is not None:
#             new_token = ""
#             # extract those tokens that are identified via index by coreferee and replace with best refrence token
#             for resolved_token in resolved:
#                 new_token = new_token + resolved_token.text + " "
#                 list_tokens[index] = new_token
#
#
#     resolved_text = "".join(list_tokens)
#     return resolved_text


if __name__ == '__main__':
    # fast = False

    # gdpr, document_root = gdpr_dependency_root()
    #
    # gdpr_article30 = gdpr.resolve_loose(pattern=[Article(number=33)])

    spacy.prefer_gpu()

    # We need to use coreferee so that PyCharm does not tidy up the import.
    if not coreferee:
        raise ModuleNotFoundError("Could not import coreferee for anaphora resolution.")

    # if Token.get_extension("reference") is None:
    #     Token.set_extension("reference", default=None)
    #
    # if not Token.get_extension("node"):
    #     Token.set_extension("node", default=None)

    # if fast:
    #     nlp = spacy.load("en_core_web_sm", disable=["ner"])
    # else:
    nlp = spacy.load("en_core_web_trf", disable=["ner"])

    # We setup spaCy with all the necessary pipe components, both custom and from other libraries.
    # Resolves anaphoric references
    # nlp.add_pipe("coreferee", config={}, after="parser")
    nlp.add_pipe("coreferee", config={})
    # Detects references
    # nlp.add_pipe(RegexReferenceDetector.SPACY_COMPONENT_NAME, config={}, after="parser")
    # # Creates reference qualifiers
    # nlp.add_pipe(ReferenceResolver.SPACY_COMPONENT_NAME, config={},
    #              after=RegexReferenceDetector.SPACY_COMPONENT_NAME)
    # # Fully resolves references
    # nlp.add_pipe(REFERENCE_QUALIFIER_RESOLVER_COMPONENT, config={},
    #              after=ReferenceResolver.SPACY_COMPONENT_NAME)
    #
    # # Does some ugly setup of the Doc object and applies nlp
    # doc = nlp_doc(document_root, gdpr_article30[0], nlp)
    #
    # # Phrase Extraction
    # phrases = []
    # phrase_extractor = PhraseExtractor()
    # for sent in doc.sents:
    #     phrases.extend(phrase_extractor.extract_from_sentence(sent))

    text = "A data subject must abide by the law. He or she must send data to the controller."
    #       --------------                        ---------
    # ->  "A data subject must abide by the law. A data subject must send data to the controller."

    doc = nlp(text)

    for token in doc:
        references = doc._.coref_chains.resolve(token)

        if references is not None:
            print(f"replace '{list(token.subtree)}' with references '{[list(it.subtree) for it in references]}'")


            # replace me and my dependents
            pass






    # print(apply_coreference_resolution(text))
