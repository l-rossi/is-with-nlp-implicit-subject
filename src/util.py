import csv
from typing import Tuple, Iterable

from spacy.tokens import Token

SUBJ_DEPS = {"agent", "csubj", "csubjpass", "expl", "nsubj", "nsubjpass"}
OBJ_DEPS = {"attr", "dobj", "dative", "oprd", "pobj"}


def get_noun_chunk(token: Token):
    for chunk in token.doc.noun_chunks:
        if token in chunk:
            return chunk.text
    return token


def load_gold_standard(file_name='./data/evaluation/gold_standard.csv') -> Iterable[Tuple[str, str, str]]:
    """
    Loads the gold standard triplets.
    """
    with open(file_name, 'r', encoding="utf-8") as file:
        reader = csv.reader(file)
        next(reader, None)
        for source, target, gs in reader:
            with open(f"./data/external{source}", 'r', encoding="utf-8") as source_file:
                source_txt = source_file.read().replace("\n", " ").replace("  ", " ")

            yield source_txt, target, gs
