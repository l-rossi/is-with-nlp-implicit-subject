from typing import List

from spacy.tokens import Doc

from missing_subject_detection.ImplicitSubjectDetection import ImplicitSubjectDetection


class ImplicitSubjectInserter:

    def insert(self, doc: Doc, targets: List[ImplicitSubjectDetection], subjects: List[str]) -> str:
        if len(subjects) != len(targets):
            raise ValueError("subjects and targets must have the same length")

        # TODO morphology
        list_tokens = list(token.text_with_ws for token in doc)

        for target, subj in zip(targets, subjects):
            # Insert
            *_, insertion_point = (x for x in target.predicate.subtree if x.dep_ != "punct")

            print(insertion_point)
            print(insertion_point.text_with_ws)
            list_tokens[insertion_point.i] = insertion_point.text_with_ws + " by " + subj

        resolved_text = "".join(list_tokens)

        return resolved_text
