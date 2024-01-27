from typing import List

from spacy.tokens import Token, Span, MorphAnalysis

from insertion.SpecializedInserter import SpecializedInserter
from insertion.pattern.inflect import conjugate
from insertion.pattern.inflect_global import PRESENT, SINGULAR, PLURAL
from missing_subject_detection.ImplicitSubjectDetection import ImplicitSubjectDetection, ImplicitSubjectType


class GerundInserter(SpecializedInserter):
    """
    Functional decomposition is a bane on my existence.
    """

    # TODO one should probably also look for temporal signal words like "when" and "during"
    PREPOSITIONS_TAKING_GERUND = {"of", "with", "for", "at", "about", "against", "up", "to"}

    def accepts(self, subject_type: ImplicitSubjectType):
        """
        Accepts gerunds.
        """
        return subject_type == ImplicitSubjectType.GERUND

    @staticmethod
    def _conjugate(verb: Token, morph: MorphAnalysis) -> str:
        map_num = {
            "Sing": SINGULAR,
            "Plur": PLURAL
        }

        pers = (morph.get("Person", ["3"]))[0]
        num = (morph.get("Number", ["Sing"]))[0]
        c = conjugate(verb.lemma_, PRESENT, int(pers), map_num[num])
        return c

    def insert(self, subj: Token, list_tokens: List[str], target: ImplicitSubjectDetection, span: Span):
        """
        Feel free to guess.
        """

        insertion_point = target.token
        while pots := [x for x in insertion_point.lefts if
                       x.dep_ == "advmod" and x.pos_ == "ADV" and all(y.dep_ != "advcl" for y in x.children)]:
            # I am just assuming that the advmod tree is in-order
            insertion_point = min(pots, key=lambda x: x.i)

        cleaned_subj = SpecializedInserter._clean_subject(subj)

        if target.token.head.text in GerundInserter.PREPOSITIONS_TAKING_GERUND:
            target_replacement = target.token.text
        else:
            target_replacement = GerundInserter._conjugate(target.token, subj.morph)

        if insertion_point == target.token:
            if insertion_point.is_sent_start:
                list_tokens[
                    insertion_point.i - span.start] = SpecializedInserter._upper_case_first(
                    cleaned_subj) + SpecializedInserter._lower_case_first(
                    target_replacement) + insertion_point.whitespace_
            else:
                list_tokens[insertion_point.i - span.start] = SpecializedInserter._lower_case_first(
                    cleaned_subj).rstrip() + " " + target_replacement + insertion_point.whitespace_
        else:
            if insertion_point.is_sent_start:
                list_tokens[
                    insertion_point.i - span.start] = SpecializedInserter._upper_case_first(
                    cleaned_subj) + " " + SpecializedInserter._lower_case_first(
                    insertion_point.text) + insertion_point.whitespace_
            else:
                list_tokens[
                    insertion_point.i - span.start] = SpecializedInserter._lower_case_first(
                    target_replacement) + " " + insertion_point.text + insertion_point.whitespace_
            list_tokens[target.token.i - span.start] = target_replacement + target.token.whitespace_

        """



        if target.token.is_sent_start:
            list_tokens[target.token.i - span.start] = SpecializedInserter._upper_case_first(
                cleaned_subj) + " " + SpecializedInserter._lower_case_first(
                target_replacement)
        else:
            list_tokens[target.token.i - span.start] = SpecializedInserter._lower_case_first(
                cleaned_subj).rstrip() + " " + SpecializedInserter._lower_case_first(
                target_replacement)"""
