from typing import List

from spacy.tokens import Doc

from missing_subject_detection.ImplicitSubjectDetection import ImplicitSubjectDetection, ImplicitSubjectType
from missing_subject_detection.ImplicitSubjectDetector import ImplicitSubjectDetector


class ImperativeDetector(ImplicitSubjectDetector):
    """
    Detects imperatives.
    """

    def detect(self, doc: Doc) -> List[ImplicitSubjectDetection]:
        return [ImplicitSubjectDetection(predicate=token, type=ImplicitSubjectType.IMPERATIVE) for sent in doc.sents for
                token in sent if token.tag_ == "VB"]
