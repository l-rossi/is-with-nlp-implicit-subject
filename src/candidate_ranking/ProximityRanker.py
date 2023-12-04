from typing import List

from spacy.tokens import Token

from candidate_ranking.CandidateRanker import CandidateRanker


class ProximityRanker(CandidateRanker):
    """
    Ranks according to how close the candidate is to the target.
    """

    CATAPHORIC_PENALTY = 20

    DEPENDANT_PENALTY = 999

    def rank(self, target: Token, candidates: List[Token]) -> List[Token]:
        if not candidates:
            return []

        assert target.doc == candidates[
            0].doc, "The ProximityRanker requires targets and candidates to be from the same doc."

        target_children = set(target.children) | {tok for c in target.children for tok in c.children if
                                                  c.dep_ == "auxpass"}
        return sorted(candidates,
                      key=lambda c:
                      abs(c.i - target.i) +
                      (self.CATAPHORIC_PENALTY if c.i > target.i else 0) +
                      (self.DEPENDANT_PENALTY if c in target_children else 0)
                      )
