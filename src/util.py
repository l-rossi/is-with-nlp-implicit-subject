from spacy.tokens import Token

SUBJ_DEPS = {"agent", "csubj", "csubjpass", "expl", "nsubj", "nsubjpass"}
OBJ_DEPS = {"attr", "dobj", "dative", "oprd", "pobj"}


def get_noun_chunk(token: Token):
    for chunk in token.doc.noun_chunks:
        if token in chunk:
            return chunk.text
    return token
