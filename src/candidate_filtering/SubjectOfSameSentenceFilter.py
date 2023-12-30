from typing import List

from spacy.tokens import Token

from candidate_filtering.CandidateFilter import CandidateFilter
from util import SUBJ_DEPS, OBJ_DEPS


class SubjectIfSameSentenceFilter(CandidateFilter):

    def filter(self, target: Token, candidates: List[Token]) -> List[Token]:
        subject_in_same_sent = [tok for tok in candidates if tok.sent == target.sent]

        target

        return []
