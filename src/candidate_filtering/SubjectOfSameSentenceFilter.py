from typing import List

from spacy.tokens import Token

from candidate_filtering.CandidateFilter import CandidateFilter
from missing_subject_detection.ImplicitSubjectDetection import ImplicitSubjectDetection
from util import search_for_head


class SubjectIfSameSentenceFilter(CandidateFilter):

    def filter(self, target: ImplicitSubjectDetection, candidates: List[Token]) -> List[Token]:
        candidate_in_sane_sent = [tok for tok in candidates if
                              tok.sent == target.predicate.sent and search_for_head(tok) != target.predicate]
        print(candidate_in_sane_sent)
        return candidate_in_sane_sent or candidates
