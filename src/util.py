import csv
from typing import Tuple, Iterable, Set, List

from spacy.tokens import Token, Span

# Constants based on textacy's constants
SUBJ_DEPS = {"agent", "csubj", "csubjpass", "expl", "nsubj", "nsubjpass"}
OBJ_DEPS = {"attr", "dobj", "dative", "oprd", "pobj"}
AUX_DEPS = {"aux", "auxpass", "neg"}
NOMINAL_SUBJ_DEPS = {"agent", "expl", "nsubj", "nsubjpass"}
CLAUSAL_SUBJ_DEPS = {"csubj", "csubjpass"}

# Adapted from Mayfield Electronic Handbook of Technical & Scientific Writing
# by Leslie C. Perelman, Edward Barrett, and James Paradis, available at
# https://www.mit.edu/course/21/21.guide/cnj-sub.htm#:~:text=Some%20common%20subordinating%20conjunctions%20are,whereas%2C%20whether%2C%20and%20while.
CONDITIONAL_SUBORDINATE_CONJUNCTIONS = {"if", "unless", "when", "where", "while"}


def get_main_verbs_of_sent(sent: Span) -> List[List[Token]]:
    """
    Based on 'get_main_verbs_of_sent' in textacy, but also groups verbs linked by conjunctions

    Return the main (non-auxiliary) verbs in a sentence.
    """

    verbs = [
        set([tok] + get_conjuncts(tok, {"VERB", "AUX"})) for tok in sent if
        tok.pos_ in {"VERB", "AUX"} and tok.dep_ not in AUX_DEPS
    ]

    verbs_out = []

    for vg in verbs:
        for vo in verbs_out:
            if not vo.isdisjoint(vg):
                vo.update(vg)
                break
        else:
            verbs_out.append(vg)

    return [list(x) for x in verbs_out]


def extract_prepositions(verb: Token):
    """
    Finds all children and children's children for which there is a direct path of prep/agent
    dependencies to the root.
    :param verb: The root from which to search for prepositions.
    :return: A list of prepositions.
    """
    out = []
    preps = [verb]
    # Tree search for all paths of prepositions as to catch constructs with multiple prepositions.
    while preps:
        p = preps.pop()
        new_preps = [tok for tok in p.rights if tok.dep_ in {"prep", "agent", "acomp"}]
        preps.extend(new_preps)
        out.extend(new_preps)
    return out


def get_objects_of_verb_consider_preposition(verbs: List[Token]) -> List[Token]:
    """
    Adapted from 'get_objects_of_verb' from textacy.

    Return all objects of a verb according to the dependency parse,
    including open clausal complements.
    """

    objs = []
    for verb in verbs:
        verb_and_prep = [verb] + extract_prepositions(verb)
        objs.extend(tok for v in verb_and_prep for tok in v.rights if tok.dep_ in OBJ_DEPS)
        objs.extend(tok for tok in verb.rights if tok.dep_ in {"acomp", "advmod"})
        objs.extend(tok for tok in verb.rights if tok.dep_ == "xcomp" and tok.pos_ != "VERB")
        objs.extend(tok for obj in objs for tok in get_conjuncts(obj, {obj.pos_}))

    return objs


def get_conjuncts(tok: Token, allowed_pos: Set[str] = None) -> List[Token]:
    """
    Adapted from '_get_conjuncts' from textacy.

    We also treat appositions as conjunctions. Though technically not correct,
    this helps with enumerations.

    Return conjunct dependents of the leftmost conjunct in a coordinated phrase,
    e.g. "Burton, [Dan], and [Josh] ...".
    """

    return [right for right in tok.rights if
            right.dep_ in {"conj", "appos"} and (not allowed_pos or right.pos_ in allowed_pos)]


def is_acl_without_subj(tok: Token):
    """
    Check if a phrase is an adnominal phrase without a subject (and thus the subject should be taken
    from higher up)
    """
    return tok.dep_ == "acl" and not any(x.dep_ in SUBJ_DEPS for x in tok.children)


def get_noun_chunk(token: Token):
    """
    Gets the noun chunk the token is contained in.
    """
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
