from typing import List

from spacy.tokens import Doc

from insertion.ImplicitSubjectInserter import ImplicitSubjectInserter
from missing_subject_detection.ImplicitSubjectDetection import ImplicitSubjectDetection


class ImplicitSubjectInserterImpl(ImplicitSubjectInserter):
    """
    Basic implementation of the subject insertion step.
    """

    def insert(self, doc: Doc, targets: List[ImplicitSubjectDetection], subjects: List[str]) -> str:
        """
        Creates a new string from the doc with the subjects inserted at the targets location.
        """

        if len(subjects) != len(targets):
            raise ValueError("subjects and targets must have the same length")

        # TODO morphology
        list_tokens = list(token.text_with_ws for token in doc)

        for target, subj in zip(targets, subjects):
            # We insert the subject after the last element in the target verbs dependency subtree
            # This is based purely on intuition as I could not find a source for where it actually belongs.
            *_, insertion_point = (x for x in target.predicate.subtree if x.dep_ != "punct")
            list_tokens[insertion_point.i] = insertion_point.text_with_ws + " by " + subj

        resolved_text = "".join(list_tokens)

        return resolved_text
