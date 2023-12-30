from abc import ABC
from typing import List

from spacy.tokens import Token

from missing_subject_detection.ImplicitSubjectDetection import ImplicitSubjectDetection


class CandidateFilter(ABC):
    """
    Filters the documents. If no meaningful filtering can be achieved, return all the candidates.
    """

    def filter(self, target: ImplicitSubjectDetection, candidates: List[Token]) -> List[Token]:
        """
        Filters the documents. If no meaningful filtering can be achieved, return all the candidates.

        :param target: The predicate or other token for which we are finding the subject.
        :param candidates: The potential candidates.
        """

        raise NotImplementedError()
