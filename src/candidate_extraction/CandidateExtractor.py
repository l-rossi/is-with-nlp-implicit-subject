from typing import List

from spacy.tokens import Doc, Token

from util import SUBJ_DEPS, OBJ_DEPS


class CandidateExtractor:
    """
    Extracts all possible candidates from the context
    """


    def extract(self, context: Doc) -> List[Token]:
        """
        Takes in a context and extracts every possible subject candidate from it.
        """
        return [tok for tok in context if tok.dep_ in SUBJ_DEPS or tok.dep_ in OBJ_DEPS]
