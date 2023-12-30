from typing import List

import evaluate
from spacy.tokens import Token

from candidate_filtering.CandidateFilter import CandidateFilter
from missing_subject_detection.ImplicitSubjectDetection import ImplicitSubjectDetection
from util import get_noun_chunk


class PerplexityFilter(CandidateFilter):
    """

    Ranks candidates on the their perplexity when concatenated with the target predicate.

    This is just a demonstration of a possible ranker and should not be used as it disregards context.

    # Correct: 1/77 (1.30%)
    """

    def __init__(self, model_id="gpt2", cutoff=0.6):
        self.perplexity = evaluate.load("perplexity", module_type="metric")
        self.model_id = model_id
        self.cutoff = cutoff

    def filter(self, target: ImplicitSubjectDetection, candidates: List[Token]) -> List[Token]:
        """
        Permutes the list of candidates according to the perplexity of the candidate appended to the target.
        This is straight up dumb and just a demonstration.
        """
        if not candidates:
            return []

        target = target.predicate

        candidate_chunks = [get_noun_chunk(x) for x in candidates]

        input_texts = [
            f"{str(target)} by {str(x)}" for x in candidate_chunks
        ]

        results = self.perplexity.compute(model_id=self.model_id,
                                          add_start_token=True,
                                          predictions=input_texts)

        # TODO get proper cutoff
        return [x for x, p in zip(candidates, results["perplexities"]) if p <= self.cutoff]
