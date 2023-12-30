from typing import List

import nltk
from spacy.tokens import Token

from candidate_filtering.CandidateFilter import CandidateFilter
from missing_subject_detection.ImplicitSubjectDetection import ImplicitSubjectDetection
from util import AUX_DEPS


class SemanticSimilarityFilter(CandidateFilter):

    def filter(self, target: ImplicitSubjectDetection, candidates: List[Token]) -> List[Token]:
        """
        Ranks a list of candidates.

        :param target: The predicate or other token for which we are finding the subject.
        :param candidates: The potential candidates.
        """
        # raise NotImplementedError()
        ps = nltk.stem.PorterStemmer()

        target_stem = ps.stem(target.predicate.text)
        candidates_with_same_stem = [x for x in candidates if ps.stem(x.text) == target_stem]

        candidates_with_same_verb = [(x, SemanticSimilarityFilter._search_for_head(x)) for x in candidates]
        candidates_with_same_verb = [(x, y) for x, y in candidates_with_same_verb if y.lemma_ == target.predicate.lemma_]

        print(target, candidates_with_same_stem, target.predicate.lemma_, candidates_with_same_verb)

        return candidates

    @staticmethod
    def _search_for_head(tok: Token):
        head = tok.head
        while head.dep_ in AUX_DEPS:
            head = head.head
        return head
