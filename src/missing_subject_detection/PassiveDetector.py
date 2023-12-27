from typing import List

from spacy.tokens import Span

from missing_subject_detection.ImplicitSubjectDetection import ImplicitSubjectDetection, ImplicitSubjectType
from missing_subject_detection.ImplicitSubjectDetector import ImplicitSubjectDetector
from missing_subject_detection.util import has_explicit_subject, has_aux_pass


class PassiveDetector(ImplicitSubjectDetector):
    def detect(self, span: Span) -> List[ImplicitSubjectDetection]:
        return [ImplicitSubjectDetection(predicate=tok, type=ImplicitSubjectType.PASSIVE) for tok in span if
                tok.tag_ == "VBN" and not has_explicit_subject(tok) and has_aux_pass(tok)]
