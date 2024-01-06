from spacy.tokens import Token

from util import ACTIVE_VOICE_SUBJ_DEPS


def has_aux_pass(token: Token):
    """
    TODO This is not actually a great way of detecting of the predicate 'needs' a subject as
    for example: Impaled on a pike [by loyalists], Oliver Cromwell's head was paraded around London.
    It seems to be a good heuristic though :/
    (Sorry for the example)
    """
    return "auxpass" in [x.dep_ for x in token.children]


def has_explicit_subject(token: Token):
    return any(c.dep_ in ACTIVE_VOICE_SUBJ_DEPS for c in token.children)
