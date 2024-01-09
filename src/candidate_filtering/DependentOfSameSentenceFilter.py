from typing import List

from spacy.tokens import Token, Span

from candidate_filtering.CandidateFilter import CandidateFilter
from missing_subject_detection.ImplicitSubjectDetection import ImplicitSubjectDetection
from util import search_for_head, search_for_head_block_nouns


class DependentOfSameSentenceFilter(CandidateFilter):
    """
    Filters candidates that are already a dependency of the target,
    """

    def filter(self, target: ImplicitSubjectDetection, candidates: List[Token], context: Span) -> List[Token]:
        dependent_lemma = {tok.lemma_ for tok in candidates if search_for_head_block_nouns(tok) == target.token}
        return [c for c in candidates if c.lemma_ not in dependent_lemma] or candidates
