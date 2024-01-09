from typing import List

from spacy.lang.en import English
from spacy.tokens import Token, Span

from candidate_filtering.CandidateFilter import CandidateFilter
from missing_subject_detection.ImplicitSubjectDetection import ImplicitSubjectType, ImplicitSubjectDetection


class ImperativeFilter(CandidateFilter):

    def filter(self, target: ImplicitSubjectDetection, candidates: List[Token], context: Span) -> List[Token]:
        if target.type == ImplicitSubjectType.IMPERATIVE:
            return [c for c in candidates if c.text.lower() == "you"][:1] or ImperativeFilter._you_token()

        return candidates

    @staticmethod
    def _you_token():
        nlp = English()
        return nlp("you")[:]
