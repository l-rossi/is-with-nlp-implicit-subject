from typing import List

from spacy.tokens import Span

from missing_subject_detection.ImplicitSubjectDetection import ImplicitSubjectDetection, ImplicitSubjectType
from missing_subject_detection.ImplicitSubjectDetector import ImplicitSubjectDetector
from missing_subject_detection.util import has_explicit_subject


class GerundDetector(ImplicitSubjectDetector):
    """
    Detects gerunds which are missing a 'subject', i.e., a by phrase.
    """

    def detect(self, span: Span) -> List[ImplicitSubjectDetection]:
        return [ImplicitSubjectDetection(predicate=tok, type=ImplicitSubjectType.GERUND) for tok in span if
                tok.tag_ == "VBG" and not has_explicit_subject(tok) and not tok.dep_ == "amod"]
