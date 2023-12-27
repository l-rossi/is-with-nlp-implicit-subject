from typing import List

from spacy.tokens import Span

from missing_subject_detection.ImplicitSubjectDetection import ImplicitSubjectDetection, ImplicitSubjectType
from missing_subject_detection.ImplicitSubjectDetector import ImplicitSubjectDetector


class NominalizedGerundWordlistDetector(ImplicitSubjectDetector):

    def __init__(self):
        # wordlist ripped from Ubuntu
        with open("./data/external/wordlist", 'r', encoding="utf-8") as wf:
            words = set(x.removesuffix("\n") for x in wf.readlines())

        # TODO maybe check if word withou -ing is a verb
        ing_words = [x for x in words if x.endswith("ing")]
        self.word_list = set(x for x in ing_words if x.removesuffix("ing") in words)

    def detect(self, span: Span) -> List[ImplicitSubjectDetection]:
        """
        Detects nouns that look like gerunds, for example processing, packing, trading...
        Does not detect all nouns referring to an action.
        """

        # This is dumb
        return [
            ImplicitSubjectDetection(predicate=tok, type=ImplicitSubjectType.NOMINALIZED_VERB) for tok in span if
            tok.pos_ == "NOUN" and tok.text.lower() in self.word_list
        ]
