from typing import List

from spacy.tokens import Token, Span

from candidate_filtering.CandidateFilter import CandidateFilter
from missing_subject_detection.ImplicitSubjectDetection import ImplicitSubjectDetection


class PartOfSpeechFilter(CandidateFilter):
    """
    Filters the candidates based on their POS.
    """

    def filter(self, target: ImplicitSubjectDetection, candidates: List[Token], context: Span) -> List[Token]:
        """
        Filters everything but nouns and 'you'.
        """
        return [c for c in candidates if c.pos_ == "NOUN" or c.text.lower() == "you"] or candidates