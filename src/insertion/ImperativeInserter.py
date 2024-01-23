from typing import List

from spacy.tokens import Token, Span

from insertion.SpecializedInserter import SpecializedInserter
from insertion.pattern.inflect import conjugate
from insertion.pattern.inflect_global import PRESENT, SINGULAR
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
                       x.dep_ == "advmod" and x.tag_ == "ADV" and all(y.dep_ != "advcl" for y in x.children)]:
            # I am just assuming that the advmod tree is in-order
            insertion_point = min(pots, key=lambda x: x.i)

        # Conjugating the verb is only necessary if the verb is highly irregular, e.g., Be better -> You are better.
        conjugated_verb = SpecializedInserter._lower_case_first(conjugate(target.token.lemma_, PRESENT, 2, SINGULAR))

        if insertion_point == target.token:
            if insertion_point.is_sent_start:
                list_tokens[
                    insertion_point.i - span.start] = "You " + SpecializedInserter._lower_case_first(
                    conjugated_verb) + insertion_point.whitespace_
            else:
                list_tokens[insertion_point.i - span.start] = "you " + conjugated_verb + insertion_point.whitespace_
        else:
            if insertion_point.is_sent_start:
                list_tokens[
                    insertion_point.i - span.start] = "You " + SpecializedInserter._lower_case_first(
                    insertion_point.text) + insertion_point.whitespace_
            else:
                list_tokens[
                    insertion_point.i - span.start] = "you " + insertion_point.text + insertion_point.whitespace_
            list_tokens[target.token.i - span.start] = conjugated_verb + target.token.whitespace_
