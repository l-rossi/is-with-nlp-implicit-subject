from typing import List

from spacy.tokens import Token, Span

from insertion.SpecializedInserter import SpecializedInserter
from missing_subject_detection.ImplicitSubjectDetection import ImplicitSubjectDetection, ImplicitSubjectType


class ImperativeInserter(SpecializedInserter):
    """
    Inserts imperatives.
    """

    def accepts(self, subject_type: ImplicitSubjectType):
        """
        Accepts imperatives.
        """
        return subject_type == ImplicitSubjectType.IMPERATIVE

    def insert(self, subj: Token, list_tokens: List[str], target: ImplicitSubjectDetection, span: Span):
        """
        Do the insert.
        """
        # To the left of advmod if part of same phrase, e.g.,
        # "[You] [a]lways use the cheapest parts"
        # But ignore if the advmod if it is its own clause, e.g.,
        # As soon as you have an account, [you] log into it.
        insertion_point = target.token
        while pots := [x for x in insertion_point.lefts if
                       x.dep_ == "advmod" and all(y.dep_ != "advcl" for y in x.children)]:
            # I am just assuming that the advmod tree is in-order
            insertion_point = min(pots, key=lambda x: x.i)

        if insertion_point.is_sent_start:
            list_tokens[
                insertion_point.i - span.start] = "You " + SpecializedInserter._lower_case_first(
                insertion_point.text_with_ws)
        else:
            list_tokens[insertion_point.i - span.start] = "you " + insertion_point.text_with_ws
