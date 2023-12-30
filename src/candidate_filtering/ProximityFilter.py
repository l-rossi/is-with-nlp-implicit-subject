from typing import List

from candidate_filtering.CandidateFilter import CandidateFilter
from spacy.tokens import Token

from missing_subject_detection.ImplicitSubjectDetection import ImplicitSubjectDetection


class ProximityFilter(CandidateFilter):
    """
    Ranks according to how close the candidate is to the target.
    """

    CATAPHORIC_PENALTY = 20

    DEPENDANT_PENALTY = 999

    def filter(self, target: ImplicitSubjectDetection, candidates: List[Token]) -> List[Token]:
        """
        Returns the physically closest candidate. At most one candidate is selected so this can be used
        at the end of a pipeline.
        """

        if not candidates:
            return []

        assert target.predicate.doc == candidates[
            0].doc, "The ProximityRanker requires targets and candidates to be from the same doc."

        target_children = set(target.predicate.children) | {tok for c in target.predicate.children for tok in c.children if
                                                  c.dep_ == "auxpass"}
        return [min(candidates,
                    key=lambda c:
                    abs(c.i - target.predicate.i) +
                    (self.CATAPHORIC_PENALTY if c.i > target.predicate.i else 0) +
                    (self.DEPENDANT_PENALTY if c in target_children else 0)
                    )]
