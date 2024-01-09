from typing import List

from spacy.tokens import Token, Span

from insertion.SpecializedInserter import SpecializedInserter
from missing_subject_detection.ImplicitSubjectDetection import ImplicitSubjectDetection, ImplicitSubjectType


class GerundInserter(SpecializedInserter):
    """
    Functional decomposition is a bane on my existence.
    """

    def accepts(self, subject_type: ImplicitSubjectType):
        """
        Accepts gerunds.
        """
        return subject_type == ImplicitSubjectType.GERUND

    def insert(self, subj: Token, list_tokens: List[str], target: ImplicitSubjectDetection, span: Span):
        """
        Feel free to guess.
        """

        # TODO
        # Better morphology, I mean why did I rip pattern if not for this?
        # Expected: While the customer waits for the payment confirmation the customer can enter its shipping address
        # Actual:   While the customer waiting for the payment confirmation the customer can enter its shipping address

        # Keep gerunds as -ing if they are a pcomp else inflect them to fit the inserted candidate

        cleaned_subj = SpecializedInserter._clean_subject(subj)
        if target.token.is_sent_start:
            list_tokens[target.token.i - span.start] = SpecializedInserter._upper_case_first(
                cleaned_subj) + " " + SpecializedInserter._lower_case_first(
                target.token.text_with_ws)
        else:
            list_tokens[target.token.i - span.start] = SpecializedInserter._lower_case_first(
                cleaned_subj).rstrip() + " " + SpecializedInserter._lower_case_first(
                target.token.text_with_ws)
