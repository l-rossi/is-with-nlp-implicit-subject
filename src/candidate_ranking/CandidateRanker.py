from abc import ABC
from typing import List

from spacy.tokens import Token


class CandidateRanker(ABC):
    """
    Ranks a list of candidates according to some criteria.
    """

    def rank(self, target: Token,  candidates: List[Token]) -> List[Token]:
        """
        Ranks a list of candidates.

        :param target: The predicate or other token for which we are finding the subject.
        :param candidates: The potential candidates.
        """

        raise NotImplementedError()
