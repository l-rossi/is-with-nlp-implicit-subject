from typing import List

from spacy.tokens import Token, Span

from missing_subject_detection.ImplicitSubjectDetection import ImplicitSubjectDetection, ImplicitSubjectType
from missing_subject_detection.ImplicitSubjectDetector import ImplicitSubjectDetector


class ImperativeDetector(ImplicitSubjectDetector):
    """
    Detects imperatives.
    """

    @staticmethod
    def _has_aux(token: Token):
        return any(c.dep_ == "aux" for c in token.children)

    def detect(self, span: Span) -> List[ImplicitSubjectDetection]:
        # TODO make sure verb is not linked by conj to non imperative
        return [ImplicitSubjectDetection(predicate=token, type=ImplicitSubjectType.IMPERATIVE) for sent in span.sents for
                token in sent if token.tag_ == "VB" and not self._has_aux(token) and token.pos_ == "VERB"]
