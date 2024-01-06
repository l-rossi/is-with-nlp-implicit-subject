from typing import List

from spacy.tokens import Span, Token

from insertion.ImplicitSubjectInserter import ImplicitSubjectInserter
from missing_subject_detection.ImplicitSubjectDetection import ImplicitSubjectDetection, ImplicitSubjectType
from util import get_noun_chunk


class ImplicitSubjectInserterImpl(ImplicitSubjectInserter):
    """
    Basic implementation of the subject insertion step.
    """

    def insert(self, span: Span, targets: List[ImplicitSubjectDetection], subjects: List[Token]) -> str:
        """
        Creates a new string from the doc with the subjects inserted at the targets location.
        """

        if len(subjects) != len(targets):
            raise ValueError("subjects and targets must have the same length")

        list_tokens = list(token.text_with_ws for token in span)

        for target, subj in zip(targets, subjects):
            if target.type == ImplicitSubjectType.IMPERATIVE:
                if target.predicate.is_sent_start:
                    list_tokens[
                        target.predicate.i - span.start] = "You " + ImplicitSubjectInserterImpl._lower_case_first(
                        target.predicate.text_with_ws)
                else:
                    list_tokens[target.predicate.i - span.start] = "you " + target.predicate.text_with_ws

            elif target.type == ImplicitSubjectType.GERUND:
                cleaned_subj = ImplicitSubjectInserterImpl._clean_subject(subj)
                if target.predicate.is_sent_start:
                    list_tokens[target.predicate.i - span.start] = ImplicitSubjectInserterImpl._upper_case_first(
                        cleaned_subj) + " " + ImplicitSubjectInserterImpl._lower_case_first(
                        target.predicate.text_with_ws)
                else:
                    list_tokens[target.predicate.i - span.start] = ImplicitSubjectInserterImpl._lower_case_first(
                        cleaned_subj).rstrip() + " " + ImplicitSubjectInserterImpl._lower_case_first(
                        target.predicate.text_with_ws)

            else:
                # We insert the subject after the last element in the target verbs dependency subtree
                # This is based purely on intuition as I could not find a source for where it actually belongs.
                *_, insertion_point = (x for x in target.predicate.subtree if x.dep_ != "punct")
                cleaned_subj = ImplicitSubjectInserterImpl._clean_subject(subj)
                list_tokens[
                    insertion_point.i - span.start] = insertion_point.text + " by " + cleaned_subj.strip() + insertion_point.whitespace_

        resolved_text = "".join(list_tokens)

        return resolved_text

    @staticmethod
    def _clean_subject(subj: Token) -> str:
        chunky_boi = get_noun_chunk(subj)
        return ImplicitSubjectInserterImpl._lower_case_first(str(chunky_boi.text_with_ws)) if not chunky_boi[
                                                                                         0].pos_ == "PROPN" else str(
            chunky_boi)

    @staticmethod
    def _lower_case_first(s: str) -> str:
        return s[0].lower() + s[1:] if s else s

    @staticmethod
    def _upper_case_first(s: str) -> str:
        return s[0].upper() + s[1:] if s else s
