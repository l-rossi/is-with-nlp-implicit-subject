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

if __name__ == '__main__':
    fast = True

    gdpr, document_root = gdpr_dependency_root()

    gdpr_article30 = gdpr.resolve_loose(pattern=[Article(number=33)])

    spacy.prefer_gpu()

    # We need to use coreferee so that PyCharm does not tidy up the import.
    if not coreferee:
        raise ModuleNotFoundError("Could not import coreferee for anaphora resolution.")

    if Token.get_extension("reference") is None:
        Token.set_extension("reference", default=None)

    if not Token.get_extension("node"):
        Token.set_extension("node", default=None)

    if fast:
        nlp = spacy.load("en_core_web_sm", disable=["ner"])
    else:
        nlp = spacy.load("en_core_web_trf", disable=["ner"])

    # We setup spaCy with all the necessary pipe components, both custom and from other libraries.
    # Resolves anaphoric references
    nlp.add_pipe("coreferee", config={}, after="parser")
    # Detects references
    nlp.add_pipe(RegexReferenceDetector.SPACY_COMPONENT_NAME, config={}, after="parser")
    # Creates reference qualifiers
    nlp.add_pipe(ReferenceResolver.SPACY_COMPONENT_NAME, config={},
                 after=RegexReferenceDetector.SPACY_COMPONENT_NAME)
    # Fully resolves references
    nlp.add_pipe(REFERENCE_QUALIFIER_RESOLVER_COMPONENT, config={},
                 after=ReferenceResolver.SPACY_COMPONENT_NAME)

    # Does some ugly setup of the Doc object and applies nlp
    doc = nlp_doc(document_root, gdpr_article30[0], nlp)

    # Phrase Extraction
    phrases = []
    phrase_extractor = PhraseExtractor()
    for sent in doc.sents:
        phrases.extend(phrase_extractor.extract_from_sentence(sent))


    print(phrases)
