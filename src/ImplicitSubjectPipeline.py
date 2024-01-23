import warnings
from functools import reduce
from typing import List, Optional, Tuple

import spacy
from spacy.tokens import Token, Span

from candidate_extraction.CandidateExtractor import CandidateExtractor
from candidate_filtering.CandidateFilter import CandidateFilter
from insertion.ImplicitSubjectInserter import ImplicitSubjectInserter
from missing_subject_detection.ImplicitSubjectDetection import ImplicitSubjectDetection
from missing_subject_detection.ImplicitSubjectDetector import ImplicitSubjectDetector


class ImplicitSubjectPipeline:
    """
    Manages the different components of the implicit subject process.
    """

    def __init__(self,
                 missing_subject_detectors: List[ImplicitSubjectDetector],
                 candidate_extractor: CandidateExtractor,
                 candidate_rankers: List[CandidateFilter],
                 missing_subject_inserter: ImplicitSubjectInserter,
                 verbose: bool = False,
                 fast: bool = False):
        self._candidate_rankers = candidate_rankers
        self._missing_subject_detectors = missing_subject_detectors
        self._candidate_extractor = candidate_extractor
        self._missing_subject_inserter = missing_subject_inserter
        self._verbose = verbose
        self._nlp = spacy.load("en_core_web_sm" if fast else "en_core_web_trf")
        self._last_detections = None
        self._last_selected_candidates = None
        self._last_log = None

    def last_selected_candidates(self):
        """
        Returns the last selected candidates. Useful for evaluation.
        """
        return self._last_selected_candidates

    def last_detections(self):
        """
        Returns the last implicit subject detections. Useful for evaluation.
        """
        return self._last_detections

    def last_filter_log(self):
        """
        Returns a log of the candidate filter mechanism. Useful for evaluation.
        """
        return self._last_log

    def _debug(self, *msg, **kwargs):
        if self._verbose:
            print(*msg, **kwargs)

    def _apply_candidate_filters(self, targets: List[ImplicitSubjectDetection], candidates: List[Token], context: Span):
        def _apply_filter(acc: List[Token], f: CandidateFilter):
            prev_len = len(acc)
            if prev_len == 1:
                self._debug(f"Short circuiting  {f.__class__.__name__} with {acc}.")
                return acc

            intermediate_result = f.filter(target, acc, context)
            self._debug(
                f"Applied {f.__class__.__name__}, filtered {100 - 100 * len(intermediate_result) / prev_len :.2f}% of "
                f"candidates and returned {intermediate_result}")
            return intermediate_result

        def _logged_apply(acc: Tuple[List[Tuple[Optional[CandidateFilter], List[Token]]], List[Token]],
                          f: CandidateFilter):
            _log, _acc = acc
            _res = _apply_filter(_acc, f)
            _log.append((f, _res))  # very functional :p
            return _log, _res

        for target in targets:
            log, res = reduce(_logged_apply, self._candidate_rankers, ([(None, candidates)], candidates))
            tok = res[0] if res else candidates[0]
            yield tok, log

    def apply(self, inspected_text: str, context: Optional[str] = None) -> str:
        """
        Runs the pipeline using the components defined in the constructor.
        """
        if not context:
            context = inspected_text

        if inspected_text not in context:
            warnings.warn(f"Could not find inspected text in context. Using separate docs. ('{inspected_text}')")
            inspected_text_span = self._nlp(inspected_text)[:]
            context_doc = self._nlp(context)
        else:
            context_doc = self._nlp(context)
            ind = context.index(inspected_text)
            inspected_text_span = context_doc.char_span(ind, ind + len(inspected_text))
            assert inspected_text == str(inspected_text_span), "Could not extract the inspected text from the context."

        targets = dict()
        for detector in reversed(self._missing_subject_detectors):
            # (ImplicitSubjectDetection are not hashable, so we use the hashable predicate token as a unique key)
            # Detectors at the front of the list take precedence over those at the back.
            targets.update({
                x.token: x for x in detector.detect(inspected_text_span)
            })
        targets = list(targets.values())
        self._last_detections = targets

        self._debug("Detected the following targets:\n", targets, sep="")
        self._debug("-----")

        candidates = self._candidate_extractor.extract(context_doc)

        self._debug("Extracted the following candidates:\n", candidates, sep="")
        self._debug("-----")

        filter_res = self._apply_candidate_filters(targets, candidates, context_doc[:])
        log = []
        subjects_for_insertion = []
        for t, l in filter_res:
            log.append(l)
            subjects_for_insertion.append(t)

        self._last_log = list(zip(targets, log))

        self._debug("Picked the following subjects for insertion:\n", *zip(targets, subjects_for_insertion), sep="")
        self._last_selected_candidates = subjects_for_insertion

        resolved = self._missing_subject_inserter.insert(inspected_text_span, targets, subjects_for_insertion)

        return resolved
