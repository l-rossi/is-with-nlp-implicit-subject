from typing import List

from spacy.tokens import Token, Span

from missing_subject_detection.ImplicitSubjectDetection import ImplicitSubjectDetection, ImplicitSubjectType
from missing_subject_detection.ImplicitSubjectDetector import ImplicitSubjectDetector
from util import AUX_DEPS


class ImperativeDetector(ImplicitSubjectDetector):
    """
    Detects imperatives.
    """

    @staticmethod
    def _has_aux(token: Token):
        return any(c.dep_ in {"aux", "auxpass"} for c in token.children)

    def detect(self, span: Span) -> List[ImplicitSubjectDetection]:
        """
        Detects imperatives.
        """
        return [ImplicitSubjectDetection(token=token, type=ImplicitSubjectType.IMPERATIVE) for token in span if
                token.tag_ == "VB" and not self._has_aux(token) and token.dep_ not in AUX_DEPS]
