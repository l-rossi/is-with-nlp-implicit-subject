from abc import ABC
from typing import List

from spacy.tokens import Doc

from missing_subject_detection.ImplicitSubjectDetection import ImplicitSubjectDetection


class ImplicitSubjectDetector(ABC):
    """
    Abstract base class for subject detection.
    """

    def detect(self, doc: Doc) -> List[ImplicitSubjectDetection]:
        """
        Detects verbs without an accompanying subject.
        """
        raise NotImplementedError()
