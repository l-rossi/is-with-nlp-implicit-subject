from typing import List, Set

from spacy.tokens import Span

from missing_subject_detection.ImplicitSubjectDetection import ImplicitSubjectDetection, ImplicitSubjectType
from missing_subject_detection.ImplicitSubjectDetector import ImplicitSubjectDetector
from missing_subject_detection.util import has_explicit_subject
from util import AUX_DEPS


class PassiveDetector(ImplicitSubjectDetector):
    """
    Detects passive verbs.
    """

    def __init__(self, blacklist: Set[str] = None):
        """
        :param blacklist: A list of predicates that should simply be ignored
        """
        if blacklist is None:
            blacklist = {"based", "referred"}
        self._blacklist = blacklist

    def detect(self, span: Span) -> List[ImplicitSubjectDetection]:
        # TODO the GS usually ignores the detection is an aux_pass is present -> and not has_aux_pass(tok)
        return [ImplicitSubjectDetection(token=tok, type=ImplicitSubjectType.PASSIVE) for tok in span if
                tok.tag_ == "VBN" and not has_explicit_subject(
                    tok) and tok.dep_ not in AUX_DEPS and not tok.dep_ == "amod" and tok.text.lower() not in self._blacklist]
