from typing import List, Optional

import spacy

from candidate_extraction.CandidateExtractor import CandidateExtractor
from candidate_ranking.CandidateRanker import CandidateRanker
from insertion.ImplicitSubjectInserter import ImplicitSubjectInserter
from missing_subject_detection.ImplicitSubjectDetector import ImplicitSubjectDetector
from util import get_noun_chunk


class ImplicitSubjectPipeline:
    """
    Manages the different components of the implicit subject process.
    """

    def __init__(self,
                 missing_subject_detectors: List[ImplicitSubjectDetector],
                 candidate_extractor: CandidateExtractor,
                 candidate_ranker: CandidateRanker,
                 missing_subject_inserter: ImplicitSubjectInserter,
                 verbose: bool = False,
                 fast: bool = False):
        self._candidate_ranker = candidate_ranker
        self._missing_subject_detectors = missing_subject_detectors
        self._candidate_extractor = candidate_extractor
        self._missing_subject_inserter = missing_subject_inserter
        self._verbose = verbose
        self._nlp = spacy.load("en_core_web_sm" if fast else "en_core_web_trf")

    def _debug(self, *msg, **kwargs):
        if self._verbose:
            print(*msg, **kwargs)

    def apply(self, inspected_text: str, context: Optional[str] = None) -> str:
        """
        Runs the pipeline using the components defined in the constructor.
        """
        if not context:
            context = inspected_text

        inspected_text_doc = self._nlp(inspected_text)
        context_doc = self._nlp(context)

        targets = dict()
        for detector in reversed(self._missing_subject_detectors):
            # (ImplicitSubjectDetection are not hashable so we use the hashable predicate token as a unique key)
            # Detectors at the front of the list take precedence over those at the back.
            targets.update({
                x.predicate: x for x in detector.detect(inspected_text_doc)
            })
        targets = list(targets.values())

        self._debug("Detected the following targets:\n", targets, sep="")
        self._debug("-----")

        candidates = self._candidate_extractor.extract(context_doc)

        self._debug("Extracted the following candidates:\n", candidates, sep="")
        self._debug("-----")

        subjects_for_insertion = [
            get_noun_chunk(self._candidate_ranker.rank(target.predicate, candidates)[0]) for target in targets
        ]

        self._debug("Picked the following subjects for insertion:\n", *zip(targets, subjects_for_insertion), sep="")

        resolved = self._missing_subject_inserter.insert(inspected_text_doc, targets, subjects_for_insertion)

        return resolved
