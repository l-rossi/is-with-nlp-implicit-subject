from typing import List

import evaluate
from spacy.tokens import Token

from candidate_filtering.CandidateFilter import CandidateFilter
from insertion.ImplicitSubjectInserterImpl import ImplicitSubjectInserterImpl
from missing_subject_detection.ImplicitSubjectDetection import ImplicitSubjectDetection


class PerplexityFilter(CandidateFilter):
    """
    Ranks candidates based on the perplexity of the generated sentence using a LLM.

    Disregards the larger context but good for filtering out semantically nonsensical candidates, e.g.,
    The letter is sent by [the rotation around the sun].
    """

    def __init__(self, model_id="gpt2", perplexity_buffer=1.5, missing_subject_inserter=None, max_returned=5):
        self._perplexity = evaluate.load("perplexity", module_type="metric")
        self._model_id = model_id
        self._perplexity_buffer = perplexity_buffer
        self._missing_subject_inserter = missing_subject_inserter or ImplicitSubjectInserterImpl()
        self._max_returned = max_returned

    def filter(self, target: ImplicitSubjectDetection, candidates: List[Token]) -> List[Token]:
        """
        Filters out candidates based the sentences perplexity when compared to the complexity of the
        sentence without the inserted candidate.
        """
        if not candidates:
            return []

        input_texts = [str(target.predicate.sent)] + [
            self._missing_subject_inserter.insert(target.predicate.sent, [target], [x]) for x in
            candidates
        ]

        baseline, *results = self._perplexity.compute(model_id=self._model_id,
                                                      add_start_token=True,
                                                      predictions=input_texts)["perplexities"]
        _, *input_texts = input_texts

        print(baseline, list(zip(input_texts, candidates, results)))

        # We allow for slightly more perplexity than the baseline (i.e., no subject inserted)
        output = [(x, p) for x, p in zip(candidates, results) if
                  p <= self._perplexity_buffer * baseline]

        output.sort(key=lambda x: x[1])

        return [x for x, _ in output][:self._max_returned] or candidates
