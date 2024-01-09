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

        cleaned_subj = SpecializedInserter._clean_subject(subj)

        # TODO put this list somewhere else and find actual reasoning behind this
        if target.token.head.text in {"of", "with", "for", "at", "about", "against", "up"}:
            target_replacement = target.token.text_with_ws
        else:
            target_replacement = GerundInserter._conjugate(target.token, subj.morph) + target.token.whitespace_

        if target.token.is_sent_start:
            list_tokens[target.token.i - span.start] = SpecializedInserter._upper_case_first(
                cleaned_subj) + " " + SpecializedInserter._lower_case_first(
                target_replacement)
        else:
            list_tokens[target.token.i - span.start] = SpecializedInserter._lower_case_first(
                cleaned_subj).rstrip() + " " + SpecializedInserter._lower_case_first(
                target_replacement)
