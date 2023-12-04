from typing import List

from spacy.tokens import Doc, Token, Span

from missing_subject_detection.ImplicitSubjectDetection import ImplicitSubjectDetection, ImplicitSubjectType
from missing_subject_detection.ImplicitSubjectDetector import ImplicitSubjectDetector


class PassiveDetector(ImplicitSubjectDetector):

    @staticmethod
    def _has_aux_pass(token: Token):
        """
        TODO This is not actually a great way of detecting of the predicate 'needs' a subject as
        for example: Impaled on a pike [by loyalists], Oliver Cromwell's head was paraded around London.
        It seems to be a good heuristic though :/
        (Sorry for the example)
        """
        return "auxpass" in [x.dep_ for x in token.children]

    @staticmethod
    def _has_explicit_subject(token: Token):
        return any(c.dep_ == "agent" for c in token.children)

    def detect(self, span: Span) -> List[ImplicitSubjectDetection]:
        return [ImplicitSubjectDetection(predicate=tok, type=ImplicitSubjectType.PASSIVE) for tok in span if
                tok.tag_ == "VBN" and not self._has_explicit_subject(tok) and self._has_aux_pass(tok)]
